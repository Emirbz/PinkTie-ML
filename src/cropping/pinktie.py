
import argparse
import collections as col
import os
import random
import shutil as sh
from functools import partial as ps
from itertools import repeat
from multiprocessing import Pool
import numpy as np
import pandas as pd
import scipy.ndimage
import torch
import torch.nn.functional as F
import tqdm
from flask import Flask, request
from flask import jsonify
from flask_cors import CORS
import src.data_loading.loading as loading
import src.heatmaps.models as models
import src.optimal_centers.calc_optimal_centers as calc_optimal_centers
import src.utilities.data_handling as data_handling
import src.utilities.pickling as pickling
import src.utilities.reading_images as reading_images
import src.utilities.saving_images as saving_images
import src.utilities.tools as tools
from src.constants import INPUT_SIZE_DICT
from src.constants import VIEWS, VIEWANGLES, LABELS
app = Flask(__name__, template_folder="templates")
CORS(app)
############# CROP MAMMOGRAM END ---1--- #####################



def get_masks_and_sizes_of_connected_components(img_mask):
    mask, num_labels = scipy.ndimage.label(img_mask)

    mask_pixels_dict = {}
    for i in range(num_labels + 1):
        this_mask = (mask == i)
        if img_mask[this_mask][0] != 0:
            # Exclude the 0-valued mask
            mask_pixels_dict[i] = np.sum(this_mask)

    return mask, mask_pixels_dict


def get_mask_of_largest_connected_component(img_mask):
    """
    Finds the largest connected component from the mask of the image
    """
    mask, mask_pixels_dict = get_masks_and_sizes_of_connected_components(img_mask)
    largest_mask_index = pd.Series(mask_pixels_dict).idxmax()
    largest_mask = mask == largest_mask_index
    return largest_mask


def get_edge_values(img, largest_mask, axis):
    """
    Finds the bounding box for the largest connected component
    """
    assert axis in ["x", "y"]
    has_value = np.any(largest_mask, axis=int(axis == "y"))
    edge_start = np.arange(img.shape[int(axis == "x")])[has_value][0]
    edge_end = np.arange(img.shape[int(axis == "x")])[has_value][-1] + 1
    return edge_start, edge_end


def get_bottommost_pixels(img, largest_mask, y_edge_bottom):
    """
    Gets the bottommost nonzero pixels of dilated mask before cropping.
    """
    bottommost_nonzero_y = y_edge_bottom - 1
    bottommost_nonzero_x = np.arange(img.shape[1])[largest_mask[bottommost_nonzero_y, :] > 0]
    return bottommost_nonzero_y, bottommost_nonzero_x


def get_distance_from_starting_side(img, mode, x_edge_left, x_edge_right):
    """
    If we fail to recover the original shape as a result of erosion-dilation
    on the side where the breast starts to appear in the image,
    we record this information.
    """
    if mode == "left":
        return img.shape[1] - x_edge_right
    else:
        return x_edge_left


def include_buffer_y_axis(img, y_edge_top, y_edge_bottom, buffer_size):
    """
    Includes buffer in all sides of the image in y-direction
    """
    if y_edge_top > 0:
        y_edge_top -= min(y_edge_top, buffer_size)
    if y_edge_bottom < img.shape[0]:
        y_edge_bottom += min(img.shape[0] - y_edge_bottom, buffer_size)
    return y_edge_top, y_edge_bottom


def include_buffer_x_axis(img, mode, x_edge_left, x_edge_right, buffer_size):
    """
    Includes buffer in only one side of the image in x-direction
    """
    if mode == "left":
        if x_edge_left > 0:
            x_edge_left -= min(x_edge_left, buffer_size)
    else:
        if x_edge_right < img.shape[1]:
            x_edge_right += min(img.shape[1] - x_edge_right, buffer_size)
    return x_edge_left, x_edge_right


def convert_bottommost_pixels_wrt_cropped_image(mode, bottommost_nonzero_y, bottommost_nonzero_x,
                                                y_edge_top, x_edge_right, x_edge_left):
    """
    Once the image is cropped, adjusts the bottommost pixel values which was originally w.r.t. the original image
    """
    bottommost_nonzero_y -= y_edge_top
    if mode == "left":
        bottommost_nonzero_x = x_edge_right - bottommost_nonzero_x  # in this case, not in sorted order anymore.
        bottommost_nonzero_x = np.flip(bottommost_nonzero_x, 0)
    else:
        bottommost_nonzero_x -= x_edge_left
    return bottommost_nonzero_y, bottommost_nonzero_x


def get_rightmost_pixels_wrt_cropped_image(mode, largest_mask_cropped, find_rightmost_from_ratio):
    """
    Ignores top find_rightmost_from_ratio of the image and searches the rightmost nonzero pixels
    of the dilated mask from the bottom portion of the image.
    """
    ignore_height = int(largest_mask_cropped.shape[0] * find_rightmost_from_ratio)
    rightmost_pixel_search_area = largest_mask_cropped[ignore_height:, :]
    rightmost_pixel_search_area_has_value = np.any(rightmost_pixel_search_area, axis=0)
    rightmost_nonzero_x = np.arange(rightmost_pixel_search_area.shape[1])[
        rightmost_pixel_search_area_has_value][-1 if mode == 'right' else 0]
    rightmost_nonzero_y = np.arange(rightmost_pixel_search_area.shape[0])[
                              rightmost_pixel_search_area[:, rightmost_nonzero_x] > 0] + ignore_height

    # rightmost pixels are already found w.r.t. newly cropped image, except that we still need to
    #   reflect horizontal_flip
    if mode == "left":
        rightmost_nonzero_x = largest_mask_cropped.shape[1] - rightmost_nonzero_x

    return rightmost_nonzero_y, rightmost_nonzero_x


def crop_img_from_largest_connected(img, mode, erode_dialate=True, iterations=100,
                                    buffer_size=50, find_rightmost_from_ratio=1 / 3):
    """
    Performs erosion on the mask of the image, selects largest connected component,
    dialates the largest connected component, and draws a bounding box for the result
    with buffers

    input:
        - img:   2D numpy array
        - mode:  breast pointing left or right

    output: a tuple of (window_location, rightmost_points,
                        bottommost_points, distance_from_starting_side)
        - window_location: location of cropping window w.r.t. original dicom image so that segmentation
           map can be cropped in the same way for training.
        - rightmost_points: rightmost nonzero pixels after correctly being flipped in the format of
                            ((y_start, y_end), x)
        - bottommost_points: bottommost nonzero pixels after correctly being flipped in the format of
                             (y, (x_start, x_end))
        - distance_from_starting_side: number of zero columns between the start of the image and start of
           the largest connected component w.r.t. original dicom image.
    """
    assert mode in ("left", "right")

    img_mask = img > 0

    # Erosion in order to remove thin lines in the background
    if erode_dialate:
        img_mask = scipy.ndimage.morphology.binary_erosion(img_mask, iterations=iterations)

    # Select mask for largest connected component
    largest_mask = get_mask_of_largest_connected_component(img_mask)

    # Dilation to recover the original mask, excluding the thin lines
    if erode_dialate:
        largest_mask = scipy.ndimage.morphology.binary_dilation(largest_mask, iterations=iterations)

    # figure out where to crop
    y_edge_top, y_edge_bottom = get_edge_values(img, largest_mask, "y")
    x_edge_left, x_edge_right = get_edge_values(img, largest_mask, "x")

    # extract bottommost pixel info
    bottommost_nonzero_y, bottommost_nonzero_x = get_bottommost_pixels(img, largest_mask, y_edge_bottom)

    # include maximum 'buffer_size' more pixels on both sides just to make sure we don't miss anything
    y_edge_top, y_edge_bottom = include_buffer_y_axis(img, y_edge_top, y_edge_bottom, buffer_size)

    # If cropped image not starting from corresponding edge, they are wrong. Record the distance, will reject if not 0.
    distance_from_starting_side = get_distance_from_starting_side(img, mode, x_edge_left, x_edge_right)

    # include more pixels on either side just to make sure we don't miss anything, if the next column
    #   contains non-zero value and isn't noise
    x_edge_left, x_edge_right = include_buffer_x_axis(img, mode, x_edge_left, x_edge_right, buffer_size)

    # convert bottommost pixel locations w.r.t. newly cropped image. Flip if necessary.
    bottommost_nonzero_y, bottommost_nonzero_x = convert_bottommost_pixels_wrt_cropped_image(
        mode,
        bottommost_nonzero_y,
        bottommost_nonzero_x,
        y_edge_top,
        x_edge_right,
        x_edge_left
    )

    # calculate rightmost point from bottom portion of the image w.r.t. cropped image. Flip if necessary.
    rightmost_nonzero_y, rightmost_nonzero_x = get_rightmost_pixels_wrt_cropped_image(
        mode,
        largest_mask[y_edge_top: y_edge_bottom, x_edge_left: x_edge_right],
        find_rightmost_from_ratio
    )

    # save window location in medical mode, but everything else in training mode
    return (y_edge_top, y_edge_bottom, x_edge_left, x_edge_right), \
           ((rightmost_nonzero_y[0], rightmost_nonzero_y[-1]), rightmost_nonzero_x), \
           (bottommost_nonzero_y, (bottommost_nonzero_x[0], bottommost_nonzero_x[-1])), \
           distance_from_starting_side


def image_orientation(horizontal_flip, side):
    """
    Returns the direction where the breast should be facing in the original image
    This information is used in cropping.crop_img_horizontally_from_largest_connected
    """
    assert horizontal_flip in ['YES', 'NO'], "Wrong horizontal flip"
    assert side in ['L', 'R'], "Wrong side"
    if horizontal_flip == 'YES':
        if side == 'R':
            return 'right'
        else:
            return 'left'
    else:
        if side == 'R':
            return 'left'
        else:
            return 'right'


@app.route('/run')
def crop_mammogram():
    id_patient = request.args.get('id');
    input_data_folder = 'C:/Users/ultra/PycharmProjects/BIRADS_classifier-master/images/' + id_patient
    # exam_list_path='C:/Users/ultra/PycharmProjects/breast_cancer_classifier/sample_data/exam_list_before_cropping.pkl'
    exam_list_path = 'C:/Users/ultra/PycharmProjects/breast_cancer_classifier/sample_data/filename.pkl'
    cropped_exam_list_path = 'C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/' + id_patient + '/' + id_patient + '.pkl'
    output_data_folder = 'C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/' + id_patient
    num_processes = 10
    num_iterations = 100
    buffer_size = 50

    exam_list = pickling.unpickle_from_file(exam_list_path)

    image_list = data_handling.unpack_exam_into_images(exam_list)

    if os.path.exists(output_data_folder):
        # Prevent overwriting to an existing directory
        print("Error: the directory to save cropped images already exists.")
        sh.rmtree('C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/'+id_patient)
        os.makedirs(output_data_folder)

    crop_mammogram_one_image_func = ps(
        crop_mammogram_one_image,
        input_data_folder=input_data_folder,
        output_data_folder=output_data_folder,
        num_iterations=num_iterations,
        buffer_size=buffer_size,
    )
    with Pool(num_processes) as pool:
        cropped_image_info = pool.map(crop_mammogram_one_image_func, image_list)

    window_location_dict = dict([x[0] for x in cropped_image_info])
    rightmost_points_dict = dict([x[1] for x in cropped_image_info])
    bottommost_points_dict = dict([x[2] for x in cropped_image_info])
    distance_from_starting_side_dict = dict([x[3] for x in cropped_image_info])

    data_handling.add_metadata(exam_list, "window_location", window_location_dict)
    data_handling.add_metadata(exam_list, "rightmost_points", rightmost_points_dict)
    data_handling.add_metadata(exam_list, "bottommost_points", bottommost_points_dict)
    data_handling.add_metadata(exam_list, "distance_from_starting_side", distance_from_starting_side_dict)

    pickling.pickle_to_file(cropped_exam_list_path, exam_list)

            ###NEXT####


    cropped_exam_list_path = 'C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/' + id_patient + '/' + id_patient + '.pkl'
    data_prefix = 'C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/' + id_patient
    output_exam_list_path = 'C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/optimal_centers/sample_output/' + id_patient + '.pkl'
    num_processes = 10
    id_patient = request.args.get('id');
    exam_list = pickling.unpickle_from_file(cropped_exam_list_path)
    data_list = data_handling.unpack_exam_into_images(exam_list, cropped=True)
    optimal_centers = get_optimal_centers(
        data_list=data_list,
        data_prefix=data_prefix,
        num_processes=num_processes
    )
    data_handling.add_metadata(exam_list, "best_center", optimal_centers)
    os.makedirs(os.path.dirname(output_exam_list_path), exist_ok=True)
    pickling.pickle_to_file(output_exam_list_path, exam_list)


     ###NEXT #####
    parser = argparse.ArgumentParser(description='Run image-only model or image+heatmap model')
    # parser.add_argument('--model-path', required=True)
    # parser.add_argument('--data-path', required=True)
    # parser.add_argument('--image-path', required=True)
    # parser.add_argument('--output-path', required=True)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--use-heatmaps', action="store_true")
    parser.add_argument('--heatmaps-path')
    parser.add_argument('--use-augmentation', action="store_true")
    parser.add_argument('--use-hdf5', action="store_true")
    # parser.add_argument('--num-epochs', default=1, type=int)
    # parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
    # parser.add_argument("--gpu-number", type=int, default=0)
    args = parser.parse_args()

    parameters = {
        "device_type": 'gpu',
        "gpu_number": 0,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": 'C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/' + id_patient,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "augmentation": args.use_augmentation,
        "num_epochs": 10,
        "use_heatmaps": args.use_heatmaps,
        "heatmaps_path": args.heatmaps_path,
        #   "heatmaps_path": 'C:/Users/ultra/jPycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_'+id_patient,
        "use_hdf5": args.use_hdf5
    }
    # load_run_save(
    #    model_path='C:/Users/ultra/PycharmProjects/breast_cancer_classifier/models/sample_image_model.p',
    #    data_path='C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/optimal_centers/sample_output/' + id_patient + '.pkl',
    #   # output_path='C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/optimal_centers/sample_output/image_predictions.csv',
    #   output_path='C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/optimal_centers/sample_output/' + id_patient + '.csv',
    #   parameters=parameters,
    # )
    return load_run_save(
        model_path='C:/Users/ultra/PycharmProjects/breast_cancer_classifier/models/sample_image_model.p',
        data_path='C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/optimal_centers/sample_output/' + id_patient + '.pkl',
        # output_path='C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/optimal_centers/sample_output/image_predictions.csv',
        output_path='C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/optimal_centers/sample_output/' + id_patient + '.csv',
        parameters=parameters,
    )







def crop_mammogram_one_image(scan, input_data_folder, output_data_folder, num_iterations, buffer_size):
    """
    Crops a mammogram, saves as png file, includes the following additional information:
        - window_location: location of cropping window w.r.t. original dicom image so that segmentation
           map can be cropped in the same way for training.
        - rightmost_points: rightmost nonzero pixels after correctly being flipped
        - bottommost_points: bottommost nonzero pixels after correctly being flipped
        - distance_from_starting_side: number of zero columns between the start of the image and start of
           the largest connected component w.r.t. original dicom image.
    """
    full_file_path = os.path.join(input_data_folder, scan['short_file_path'] + '.png')
    image = reading_images.read_image_png(full_file_path)
    try:
        # error detection using erosion. Also get cropping information for this image.
        cropping_info = crop_img_from_largest_connected(
            image,
            image_orientation(scan['horizontal_flip'], scan['side']),
            True,
            num_iterations,
            buffer_size,
            1 / 3
        )
    except Exception as error:
        print(full_file_path, "\n\tFailed to crop image because image is invalid.", str(error))
    else:
        success_image_info = list(zip([scan['short_file_path']] * 4, cropping_info))

        top, bottom, left, right = cropping_info[0]

        full_target_file_path = os.path.join(output_data_folder, scan['short_file_path'] + '.png')
        target_parent_dir = os.path.split(full_target_file_path)[0]
        if not os.path.exists(target_parent_dir):
            os.makedirs(target_parent_dir)

        try:
            saving_images.save_image_as_png(image[top:bottom, left:right], full_target_file_path)
        except Exception as error:
            print(full_file_path, "\n\tError while saving image.", str(error))

        return success_image_info


############# CROP MAMMOGRAM END ---1--- #####################

############# EXTRACT CENTERS ---2--- #####################





def extract_center(datum, image):
    """
    Compute the optimal center for an image
    """
    image = loading.flip_image(image, datum["full_view"], datum['horizontal_flip'])
    if datum["view"] == "MLO":
        tl_br_constraint = calc_optimal_centers.get_bottomrightmost_pixel_constraint(
            rightmost_x=datum["rightmost_points"][1],
            bottommost_y=datum["bottommost_points"][0],
        )
    elif datum["view"] == "CC":
        tl_br_constraint = calc_optimal_centers.get_rightmost_pixel_constraint(
            rightmost_x=datum["rightmost_points"][1]
        )
    else:
        raise RuntimeError(datum["view"])
    optimal_center = calc_optimal_centers.get_image_optimal_window_info(
        image,
        com=np.array(image.shape) // 2,
        window_dim=np.array(INPUT_SIZE_DICT[datum["full_view"]]),
        tl_br_constraint=tl_br_constraint,
    )
    return optimal_center["best_center_y"], optimal_center["best_center_x"]


def load_and_extract_center(datum, data_prefix):
    """
    Load image and computer optimal center
    """
    full_image_path = os.path.join(data_prefix, datum["short_file_path"] + '.png')
    image = reading_images.read_image_png(full_image_path)
    return datum["short_file_path"], extract_center(datum, image)


def get_optimal_centers(data_list, data_prefix, num_processes=1):
    """
    Compute optimal centers for each image in data list
    """
    pool = Pool(num_processes)
    result = pool.starmap(load_and_extract_center, zip(data_list, repeat(data_prefix)))
    return dict(result)





############# EXTRACT CENTERS  END ---2--- #####################



############# GENERATE HEATMAPS   ---3--- #####################


# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
# tf.logging.set_verbosity(tf.logging.ERROR)


def stride_list_generator(img_width, patch_size, more_patches=0, stride_fixed=-1):
    """
    Determines how an image should be split up into patches
    """
    if stride_fixed != -1:
        patch_num_lower_bound = (img_width - patch_size) // stride_fixed + 1
        pixel_left = (img_width - patch_size) % stride_fixed
        more_patches = 0
    else:
        patch_num_lower_bound = img_width // patch_size
        pixel_left = img_width % patch_size
        stride_fixed = patch_size

    if pixel_left == 0 and more_patches == 0:
        stride = stride_fixed
        patch_num = patch_num_lower_bound
        sliding_steps = patch_num - 1
        stride_list = [stride] * sliding_steps
    else:
        pixel_overlap = stride_fixed - pixel_left + more_patches * stride_fixed
        patch_num = patch_num_lower_bound + 1 + more_patches
        sliding_steps = patch_num - 1

        stride_avg = stride_fixed - pixel_overlap // sliding_steps

        sliding_steps_smaller = pixel_overlap % sliding_steps
        stride_smaller = stride_avg - 1

        stride_list = [stride_avg] * sliding_steps

        for step in random.sample(range(sliding_steps), sliding_steps_smaller):
            stride_list[step] = stride_smaller

    return stride_list


def prediction_by_batch(minibatch_patches, model, device, parameters):
    """
    Puts patches into a batch and gets predictions of patch classifier.
    """
    minibatch_x = np.stack((minibatch_patches,) * parameters['input_channels'], axis=-1).reshape(
        -1, parameters['patch_size'], parameters['patch_size'], parameters['input_channels']
    ).transpose(0, 3, 1, 2)

    with torch.no_grad():
        output = F.softmax(model(torch.FloatTensor(minibatch_x).to(device)), dim=1).cpu().detach().numpy()
    return output


def ori_image_prepare(short_file_path, view, horizontal_flip, parameters):
    """
    Loads an image and creates stride_lists
    """
    orginal_image_path = parameters['orginal_image_path']
    patch_size = parameters['patch_size']
    more_patches = parameters['more_patches']
    stride_fixed = parameters['stride_fixed']

    image_extension = '.hdf5' if parameters['use_hdf5'] else '.png'
    image_path = os.path.join(orginal_image_path, short_file_path + image_extension)
    image = loading.load_image(image_path, view, horizontal_flip)
    image = image.astype(float)
    loading.standard_normalize_single_image(image)

    img_width, img_length = image.shape
    width_stride_list = stride_list_generator(img_width, patch_size, more_patches, stride_fixed)
    length_stride_list = stride_list_generator(img_length, patch_size, more_patches, stride_fixed)

    return image, width_stride_list, length_stride_list


def patch_batch_prepare(image, length_stride_list, width_stride_list, patch_size):
    """
    Samples patches from an image according to stride_lists
    """
    min_x, min_y = 0, 0
    minibatch_patches = []
    img_width, img_length = image.shape

    for stride_y in length_stride_list + [0]:
        for stride_x in width_stride_list + [-(img_width - patch_size)]:
            patch = image[min_x:min_x + patch_size, min_y:min_y + patch_size]
            minibatch_patches.append(np.expand_dims(patch, axis=2))
            min_x += stride_x
        min_y += stride_y

    return minibatch_patches


def probabilities_to_heatmap(patch_counter, all_prob, image_shape, length_stride_list, width_stride_list,
                             patch_size, heatmap_type):
    """
    Generates heatmaps using output of patch classifier
    """
    min_x, min_y = 0, 0

    prob_map = np.zeros(image_shape, dtype=np.float32)
    count_map = np.zeros(image_shape, dtype=np.float32)

    img_width, img_length = image_shape

    for stride_y in length_stride_list + [0]:
        for stride_x in width_stride_list + [-(img_width - patch_size)]:
            prob_map[min_x:min_x + patch_size, min_y:min_y + patch_size] += all_prob[
                patch_counter, heatmap_type
            ]
            count_map[min_x:min_x + patch_size, min_y:min_y + patch_size] += 1
            min_x += stride_x
            patch_counter += 1
        min_y += stride_y

    heatmap = prob_map / count_map

    return heatmap, patch_counter


def get_all_prob(all_patches, minibatch_size, model, device, parameters):
    """
    Gets predictions for all sampled patches
    """
    all_prob = np.zeros((len(all_patches), parameters['number_of_classes']))

    for i, minibatch in enumerate(tools.partition_batch(all_patches, minibatch_size)):
        minibatch_prob = prediction_by_batch(minibatch, model, device, parameters)
        all_prob[i * minibatch_size: i * minibatch_size + minibatch_prob.shape[0]] = minibatch_prob

    return all_prob.astype(np.float32)


def save_heatmaps(heatmap_malignant, heatmap_benign, short_file_path, view, horizontal_flip, parameters):
    """
    Saves the heatmaps after flipping back to the original direction
    """
    heatmap_malignant = loading.flip_image(heatmap_malignant, view, horizontal_flip)
    heatmap_benign = loading.flip_image(heatmap_benign, view, horizontal_flip)

    heatmap_save_path_malignant = os.path.join(
        parameters['save_heatmap_path'][0],
        short_file_path + '.hdf5'
    )
    saving_images.save_image_as_hdf5(heatmap_malignant, heatmap_save_path_malignant)

    heatmap_save_path_benign = os.path.join(
        parameters['save_heatmap_path'][1],
        short_file_path + '.hdf5'
    )
    saving_images.save_image_as_hdf5(heatmap_benign, heatmap_save_path_benign)


def sample_patches(exam, parameters):
    """
    Samples patches for one exam
    """
    all_patches = []
    all_cases = []
    for view in VIEWS.LIST:
        for short_file_path in exam[view]:
            image, width_stride_list, length_stride_list = ori_image_prepare(
                short_file_path,
                view,
                exam['horizontal_flip'],
                parameters
            )

            all_patches.extend(
                patch_batch_prepare(
                    image,
                    length_stride_list,
                    width_stride_list,
                    parameters['patch_size']
                )
            )
            all_cases.append(
                (
                    short_file_path,
                    image.shape,
                    view,
                    exam['horizontal_flip'],
                    width_stride_list,
                    length_stride_list
                )
            )
    return all_patches, all_cases


def making_heatmap_with_large_minibatch_potential(parameters, model, exam_list, device):
    """
    Samples patches for each exam, gets batch prediction, creates and saves heatmaps
    """
    minibatch_size = parameters['minibatch_size']

    os.makedirs(parameters['save_heatmap_path'][0], exist_ok=True)
    os.makedirs(parameters['save_heatmap_path'][1], exist_ok=True)

    for exam in tqdm.tqdm(exam_list):

        # create patches and other information with the images
        all_patches, all_cases = sample_patches(exam, parameters)

        if len(all_patches) != 0:
            all_prob = get_all_prob(
                all_patches,
                minibatch_size,
                model,
                device,
                parameters
            )

            del all_patches

            patch_counter = 0

            for (short_file_path, image_shape, view, horizontal_flip, width_stride_list, length_stride_list) \
                    in all_cases:
                heatmap_malignant, _ = probabilities_to_heatmap(
                    patch_counter,
                    all_prob,
                    image_shape,
                    length_stride_list,
                    width_stride_list,
                    parameters['patch_size'],
                    parameters['heatmap_type'][0]
                )
                heatmap_benign, patch_counter = probabilities_to_heatmap(
                    patch_counter,
                    all_prob,
                    image_shape,
                    length_stride_list,
                    width_stride_list,
                    parameters['patch_size'],
                    parameters['heatmap_type'][1]
                )
                save_heatmaps(
                    heatmap_malignant,
                    heatmap_benign,
                    short_file_path,
                    view,
                    horizontal_flip,
                    parameters
                )

                del heatmap_malignant, heatmap_benign

            del all_prob, all_cases


def load_model_and_produce_heatmaps(parameters):
    """
    Loads trained patch classifier and generates heatmaps for all exams
    """
    # set random seed at the beginning of program
    random.seed(parameters['seed'])

    if (parameters["device_type"] == "gpu") and torch.has_cudnn:
        device = torch.device("cuda:{}".format(parameters["gpu_number"]))
    else:
        device = torch.device("cpu")

    model = models.ModifiedDenseNet121(num_classes=parameters['number_of_classes'])
    model.load_from_path(parameters["initial_parameters"])
    model = model.to(device)
    model.eval()

    # Load exam info
    exam_list = pickling.unpickle_from_file(parameters['data_file'])

    # Create heatmaps
    making_heatmap_with_large_minibatch_potential(parameters, model, exam_list, device)


@app.route('/heatmap')
def main3():
    id_patient = request.args.get('id');
    parser = argparse.ArgumentParser(description='Produce Heatmaps')
    # parser.add_argument('--model-path', required=True)
    #   parser.add_argument('--data-path', required=True)
    #   parser.add_argument('--image-path', required=True)
    # parser.add_argument('--batch-size', default=100, type=int)
    # parser.add_argument('--output-heatmap-path', required=True)
    parser.add_argument('--seed', default=0, type=int)
    # parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
    # parser.add_argument("--gpu-number", type=int, default=0)
    parser.add_argument("--use-hdf5", action="store_true")
    args = parser.parse_args()

    parameters = dict(
        device_type='gpu',
        gpu_number=0,

        patch_size=256,

        stride_fixed=70,
        more_patches=5,
        minibatch_size=100,
        seed=args.seed,
        initial_parameters="C:/Users/ultra/PycharmProjects/breast_cancer_classifier/models/sample_patch_model.p",
        input_channels=3,
        number_of_classes=4,

        data_file="C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/optimal_centers/sample_output/" + id_patient + '.pkl',
        orginal_image_path="C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/" + id_patient,
        save_heatmap_path=[os.path.join(
            'C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_' + id_patient,
            'heatmap_malignant'),
                           os.path.join(
                               'C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_' + id_patient,
                               'heatmap_benign')],

        heatmap_type=[0, 1],  # 0: malignant 1: benign 0: nothing

        use_hdf5=args.use_hdf5
    )

    load_model_and_produce_heatmaps(parameters)
    return jsonify("Done")


############# GENERATE HEATMAPS END   ---3--- #####################


############# RUN CLASSIFIER   ---4--- #####################




def run_model(model, exam_list, parameters):
    """
    Returns predictions of image only model or image+heatmaps model.
    Prediction for each exam is averaged for a given number of epochs.
    """
    if (parameters["device_type"] == "gpu") and torch.has_cudnn:
        device = torch.device("cuda:{}".format(parameters["gpu_number"]))
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    random_number_generator = np.random.RandomState(parameters["seed"])

    image_extension = ".hdf5" if parameters["use_hdf5"] else ".png"

    with torch.no_grad():
        predictions_ls = []
        for datum in tqdm.tqdm(exam_list):
            predictions_for_datum = []
            loaded_image_dict = {view: [] for view in VIEWS.LIST}
            loaded_heatmaps_dict = {view: [] for view in VIEWS.LIST}
            for view in VIEWS.LIST:
                for short_file_path in datum[view]:
                    loaded_image = loading.load_image(
                        image_path=os.path.join(parameters["image_path"], short_file_path + image_extension),
                        view=view,
                        horizontal_flip=datum["horizontal_flip"],
                    )
                    if parameters["use_heatmaps"]:
                        loaded_heatmaps = loading.load_heatmaps(
                            benign_heatmap_path=os.path.join(parameters["heatmaps_path"], "heatmap_benign",
                                                             short_file_path + ".hdf5"),
                            malignant_heatmap_path=os.path.join(parameters["heatmaps_path"], "heatmap_malignant",
                                                                short_file_path + ".hdf5"),
                            view=view,
                            horizontal_flip=datum["horizontal_flip"],
                        )
                    else:
                        loaded_heatmaps = None

                    loaded_image_dict[view].append(loaded_image)
                    loaded_heatmaps_dict[view].append(loaded_heatmaps)
            for data_batch in tools.partition_batch(range(parameters["num_epochs"]), parameters["batch_size"]):
                batch_dict = {view: [] for view in VIEWS.LIST}
                for _ in data_batch:
                    for view in VIEWS.LIST:
                        image_index = 0
                        if parameters["augmentation"]:
                            image_index = random_number_generator.randint(low=0, high=len(datum[view]))
                        cropped_image, cropped_heatmaps = loading.augment_and_normalize_image(
                            image=loaded_image_dict[view][image_index],
                            auxiliary_image=loaded_heatmaps_dict[view][image_index],
                            view=view,
                            best_center=datum["best_center"][view][image_index],
                            random_number_generator=random_number_generator,
                            augmentation=parameters["augmentation"],
                            max_crop_noise=parameters["max_crop_noise"],
                            max_crop_size_noise=parameters["max_crop_size_noise"],
                        )
                        if loaded_heatmaps_dict[view][image_index] is None:
                            batch_dict[view].append(cropped_image[:, :, np.newaxis])
                        else:
                            batch_dict[view].append(np.concatenate([
                                cropped_image[:, :, np.newaxis],
                                cropped_heatmaps,
                            ], axis=2))

                tensor_batch = {
                    view: torch.tensor(np.stack(batch_dict[view])).permute(0, 3, 1, 2).to(device)
                    for view in VIEWS.LIST
                }
                output = model(tensor_batch)
                batch_predictions = compute_batch_predictions(output)
                pred_df = pd.DataFrame({k: v[:, 1] for k, v in batch_predictions.items()})
                pred_df.columns.names = ["label", "view_angle"]
                predictions = pred_df.T.reset_index().groupby("label").mean().T.values
                predictions_for_datum.append(predictions)
            predictions_ls.append(np.mean(np.concatenate(predictions_for_datum, axis=0), axis=0))

    return np.array(predictions_ls)


def compute_batch_predictions(y_hat):
    """
    Format predictions from different heads
    """
    batch_prediction_dict = col.OrderedDict([
        ((label_name, view_angle),
         np.exp(y_hat[view_angle][:, i].cpu().detach().numpy()))
        for i, label_name in enumerate(LABELS.LIST)
        for view_angle in VIEWANGLES.LIST
    ])
    return batch_prediction_dict


def load_run_save(model_path, data_path, output_path, parameters):
    """
    Outputs the predictions as csv file
    """
    input_channels = 3 if parameters["use_heatmaps"] else 1
    import src.modeling.models as am

    model = am.SplitBreastModel(input_channels)
    model.load_state_dict(torch.load(model_path)["model"])
    exam_list = pickling.unpickle_from_file(data_path)
    predictions = run_model(model, exam_list, parameters)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Take the positive prediction
    df = pd.DataFrame(predictions, columns=LABELS.LIST)
    df.to_csv(output_path, index=False, float_format='%.4f')
    return df.to_json(orient='records',index=True)






############# RUN CLASSIFIER END   ---4--- #####################









if __name__ == "__main__":
    import src.utilities.IP as ip
    app.run(host=ip.ip_locale, port=2501, debug=True)
