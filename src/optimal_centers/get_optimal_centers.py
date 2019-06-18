
import argparse
import numpy as np
import os
from itertools import repeat
from multiprocessing import Pool

from src.constants import INPUT_SIZE_DICT
import src.utilities.pickling as pickling
import src.utilities.data_handling as data_handling
import src.utilities.reading_images as reading_images
import src.data_loading.loading as loading
import src.optimal_centers.calc_optimal_centers as calc_optimal_centers
from flask import Flask, request
from flask_restful import Resource
from flask_cors import CORS

from flask import jsonify

app = Flask(__name__, template_folder="templates")
CORS(app)



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

@app.route('/center')
def main():
    id_patient = request.args.get('id');
    cropped_exam_list_path='C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/'+id_patient+'/'+id_patient+'.pkl'
    data_prefix='C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/'+id_patient
    output_exam_list_path='sample_output/'+id_patient+'.pkl'
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
    return jsonify("Done")


if __name__ == "__main__":
    app.run(host='192.168.1.42', port=2502, debug=True)
