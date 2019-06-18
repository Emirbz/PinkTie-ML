# PinkTie:  Deep Neural Networks Improve Radiologists' Performance in Breast Cancer Screening

## Introduction
This is an implementation of the model used for breast cancer classification as described in this paper [Deep Neural Networks Improve Radiologists' Performance in Breast Cancer Screening]. The implementation allows users to get breast cancer predictions by applying one of our models: a model which takes images as input (*image-only*) and a model which takes images and heatmaps as input (*image-and-heatmaps*). 

* Input images: 2 CC view mammography images of size 2677x1942 and 2 MLO view mammography images of size 2974x1748. Each image is saved as 16-bit png file and gets standardized separately before being fed to the models.
* Input heatmaps: output of the patch classifier constructed to be the same size as its corresponding mammogram. Two heatmaps are generated for each mammogram, one for benign and one for malignant category. The value of each pixel in both of them is between 0 and 1.
* Output: 2 predictions for each breast, probability of benign and malignant findings: `left_benign`, `right_benign`, `left_malignant`, and `right_malignant`.

Both models act on screening mammography exams with four standard views (L-CC, R-CC, L-MLO, R-MLO). As a part of this repository, we provide 4 sample exams (in `sample_data/images` directory and exam list stored in `sample_data/exam_list_before_cropping.pkl`). Heatmap generation model and cancer classification models are implemented in PyTorch. 

## Prerequisites

* Python (3.6)
* PyTorch (0.4.0)
* torchvision (0.2.0)
* NumPy (1.14.3)
* SciPy (1.0.0)
* H5py (2.7.1)
* imageio (2.4.1)
* pandas (0.22.0)
* tqdm (4.19.8)
* opencv-python (3.4.2)


## How to run the code

`pinktie.py` will automatically run the entire pipeline and save the prediction results in csv and return rsult via JSON. 

It's recommended running the code with a gpu (set by default). To run the code with cpu only, please change `DEVICE_TYPE`  to 'cpu'.  


This is an example of the outputs . 

Predictions using *image-only* model

| index | left_benign | right_benign | left_malignant | right_malignant |
| ----- | ----------- | ------------ | -------------- | --------------- |
| 0     | 0.0580      | 0.0091       | 0.0754         | 0.0179          |



Predictions using *image-and-heatmaps* model 

| index | left_benign  | right_benign | left_malignant | right_malignant |
| ----- | ------------ | ------------ | -------------- | --------------- |
| 0     | 0.0612       | 0.0099       | 0.0754         | 0.0179          |




## Data

To use one of the pretrained models, the input is required to consist of at least four images, at least one for each view (L-CC, L-MLO, R-CC, R-MLO). 



```python
{
  'horizontal_flip': 'NO',
  'L-CC': ['0_L_CC'],
  'R-CC': ['0_R_CC'],
  'L-MLO': ['0_L_MLO'],
  'R-MLO': ['0_R_MLO'],
}
```

We expect images from `L-CC` and `L-MLO` views to be facing right direction, and images from `R-CC` and `R-MLO` views are facing left direction. `horizontal_flip` indicates whether all images in the exam are flipped horizontally from expected. Values for `L-CC`, `R-CC`, `L-MLO`, and `R-MLO` are list of image filenames without extension and directory name. 




The labels for the included exams are as follows:

| index | left_benign | right_benign | left_malignant | right_malignant |
| ----- | ----------- | ------------ | -------------- | --------------- |
| 0     | 0           | 0            | 0              | 0               |
| 1     | 0           | 0            | 0              | 1               |
| 2     | 1           | 0            | 0              | 0               |
| 3     | 1           | 1            | 1              | 1               |


## Pipeline

The pipeline consists of four stages.

1. Crop mammograms
2. Calculate optimal centers
3. Generate Heatmaps
4. Run classifiers

The following variables defined in `run.sh` can be modified as needed:
* `NUM_PROCESSES`: The number of processes to be used in preprocessing (`src/cropping/crop_mammogram.py` and `src/optimal_centers/get_optimal_centers.py`). Default: 10.
* `DEVICE_TYPE`: Device type to use in heatmap generation and classifiers, either 'cpu' or 'gpu'. Default: 'gpu'
* `NUM_EPOCHS`: The number of epochs to be averaged in the output of the classifiers. Default: 10.
* `HEATMAP_BATCH_SIZE`: The batch size to use in heatmap generation. Default: 100.
* `GPU_NUMBER`: Specify which one of the GPUs to use when multiple GPUs are available. Default: 0. 

* `DATA_FOLDER`: The directory where the mammogram is stored.
* `INITIAL_EXAM_LIST_PATH`: The path where the initial exam list without any metadata is stored.
* `PATCH_MODEL_PATH`: The path where the saved weights for the patch classifier is saved.
* `IMAGE_MODEL_PATH`: The path where the saved weights for the *image-only* model is saved.
* `IMAGEHEATMAPS_MODEL_PATH`: The path where the saved weights for the *image-and-heatmaps* model is saved.

* `CROPPED_IMAGE_PATH`: The directory to save cropped mammograms.
* `CROPPED_EXAM_LIST_PATH`: The path to save the new exam list with cropping metadata.
* `EXAM_LIST_PATH`: The path to save the new exam list with best center metadata.
* `HEATMAPS_PATH`: The directory to save heatmaps.
* `IMAGE_PREDICTIONS_PATH`: The path to save predictions of *image-only* model.
* `IMAGEHEATMAPS_PREDICTIONS_PATH`: The path to save predictions of *image-and-heatmaps* model.



#### Crop mammograms
```bash
python src/cropping/crop_mammogram.py \
    --input-data-folder $DATA_FOLDER \
    --output-data-folder $CROPPED_IMAGE_PATH \
    --exam-list-path $INITIAL_EXAM_LIST_PATH  \
    --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH  \
    --num-processes $NUM_PROCESSES
```
`src/import_data/crop_mammogram.py` crops the mammogram around the breast and discards the background in order to improve image loading time and time to run segmentation algorithm and saves each cropped image to `$PATH_TO_SAVE_CROPPED_IMAGES/short_file_path.png` using h5py. In addition, it adds additional information for each image and creates a new image list to `$CROPPED_IMAGE_LIST_PATH` while discarding images which it fails to crop. Optional --verbose argument prints out information about each image. The additional information includes the following:
- `window_location`: location of cropping window w.r.t. original dicom image so that segmentation map can be cropped in the same way for training.
- `rightmost_points`: rightmost nonzero pixels after correctly being flipped.
- `bottommost_points`: bottommost nonzero pixels after correctly being flipped.
- `distance_from_starting_side`: records if zero-value gap between the edge of the image and the breast is found in the side where the breast starts to appear and thus should have been no gap. Depending on the dataset, this value can be used to determine wrong value of `horizontal_flip`.


#### Calculate optimal centers
```bash
python src/optimal_centers/get_optimal_centers.py \
    --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH \
    --data-prefix $CROPPED_IMAGE_PATH \
    --output-exam-list-path $EXAM_LIST_PATH \
    --num-processes $NUM_PROCESSES
```
`src/optimal_centers/get_optimal_centers.py` outputs new exam list with additional metadata to `$EXAM_LIST_PATH`. The additional information includes the following:
- `best_center`: optimal center point of the window for each image. The augmentation windows drawn with `best_center` as exact center point could go outside the boundary of the image. This usually happens when the cropped image is smaller than the window size. In this case, we pad the image and shift the window to be inside the padded image in augmentation. Refer to [the data report](https://cs.nyu.edu/~kgeras/reports/datav1.0.pdf) for more details.


### Heatmap Generation
```bash
python3 src/heatmaps/run_producer.py \
    --model-path $PATCH_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --batch-size $HEATMAP_BATCH_SIZE \
    --output-heatmap-path $HEATMAPS_PATH \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER
```

`src/heatmaps/run_producer.py` generates heatmaps by combining predictions for patches of images and saves them as hdf5 format in `$HEATMAPS_PATH` using `$DEVICE_TYPE` device. `$DEVICE_TYPE` can either be 'gpu' or 'cpu'. `$HEATMAP_BATCH_SIZE` should be adjusted depending on available memory size.  An optional argument `--gpu-number`  can be used to specify which GPU to use.

### Running the models

`src/modeling/run_model.py` can provide predictions using cropped images either with or without heatmaps. When using heatmaps, please use the`--use-heatmaps` flag and provide appropriate the `--model-path` and `--heatmaps-path` arguments. Depending on the available memory, the optional argument `--batch-size` can be provided. Another optional argument `--gpu-number` can be used to specify which GPU to use.

#### Run image only model
```bash
python3 src/modeling/run_model.py \
    --model-path $IMAGE_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --output-path $IMAGE_PREDICTIONS_PATH \
    --use-augmentation \
    --num-epochs $NUM_EPOCHS \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER
```

This command makes predictions only using images for `$NUM_EPOCHS` epochs with random augmentation and outputs averaged predictions per exam to `$IMAGE_PREDICTIONS_PATH`. 

#### Run image+heatmaps model 
```bash
python3 src/modeling/run_model.py \
    --model-path $IMAGEHEATMAPS_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --output-path $IMAGEHEATMAPS_PREDICTIONS_PATH \
    --use-heatmaps \
    --heatmaps-path $HEATMAPS_PATH \
    --use-augmentation \
    --num-epochs $NUM_EPOCHS \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER
```

This command makes predictions using images and heatmaps for `$NUM_EPOCHS` epochs with random augmentation and outputs averaged predictions per exam to `$IMAGEHEATMAPS_PREDICTIONS_PATH`. 

## Reference


