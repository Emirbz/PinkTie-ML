
import argparse
import collections as col
import numpy as np
import os
import pandas as pd
import torch
import tqdm

import src.utilities.pickling as pickling
import src.utilities.tools as tools
import src.modeling.models as models
import src.data_loading.loading as loading
from src.constants import VIEWS, VIEWANGLES, LABELS
from flask import Flask, request
from flask_restful import Resource
from flask_cors import CORS

from flask import jsonify
app = Flask(__name__, template_folder="templates")
CORS(app)


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
    model = models.SplitBreastModel(input_channels)
    model.load_state_dict(torch.load(model_path)["model"])
    exam_list = pickling.unpickle_from_file(data_path)
    predictions = run_model(model, exam_list, parameters)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Take the positive prediction
    df = pd.DataFrame(predictions, columns=LABELS.LIST)
    df.to_csv(output_path, index=False, float_format='%.4f')
    return df.to_json(orient='records',index=True)


@app.route('/run')
def main():
    id_patient = request.args.get('id');
    parser = argparse.ArgumentParser(description='Run image-only model or image+heatmap model')
   # parser.add_argument('--model-path', required=True)
    #parser.add_argument('--data-path', required=True)
   # parser.add_argument('--image-path', required=True)
   # parser.add_argument('--output-path', required=True)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--use-heatmaps', action="store_true")
    parser.add_argument('--heatmaps-path')
    parser.add_argument('--use-augmentation', action="store_true")
    parser.add_argument('--use-hdf5', action="store_true")
    #parser.add_argument('--num-epochs', default=1, type=int)
    #parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
    #parser.add_argument("--gpu-number", type=int, default=0)
    args = parser.parse_args()

    parameters = {
        "device_type": 'gpu',
        "gpu_number": 0,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": 'C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/cropping/sample_output/'+id_patient,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "augmentation": args.use_augmentation,
        "num_epochs": 10,
        "use_heatmaps": args.use_heatmaps,
       "heatmaps_path": args.heatmaps_path,
     #   "heatmaps_path": 'C:/Users/ultra/jPycharmProjects/breast_cancer_classifier/src/cropping/sample_output/heatmaps_'+id_patient,
        "use_hdf5": args.use_hdf5
    }
    #load_run_save(
    #    model_path='C:/Users/ultra/PycharmProjects/breast_cancer_classifier/models/sample_image_model.p',
    #    data_path='C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/optimal_centers/sample_output/' + id_patient + '.pkl',
    #   # output_path='C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/optimal_centers/sample_output/image_predictions.csv',
    #   output_path='C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/optimal_centers/sample_output/' + id_patient + '.csv',
    #   parameters=parameters,
    #)
    return load_run_save(
        model_path='C:/Users/ultra/PycharmProjects/breast_cancer_classifier/models/sample_image_model.p',
        data_path='C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/optimal_centers/sample_output/'+id_patient+'.pkl',
       # output_path='C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/optimal_centers/sample_output/image_predictions.csv',
        output_path='C:/Users/ultra/PycharmProjects/breast_cancer_classifier/src/optimal_centers/sample_output/'+id_patient+'.csv',
        parameters=parameters,
    )



if __name__ == "__main__":
    app.run(host='192.168.1.42', port=2504, debug=True)
