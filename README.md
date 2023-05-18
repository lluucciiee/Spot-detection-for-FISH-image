# Spot-detection-for-FISH-image
## Introduction
A FISH image analysis workflow composed of cell segmentation, spot labeling, model training, spot detection and a basic quantitative analysis of RNA subcellular localization for both 2d and 3d image.
## Dependency

1. Cell segmentation https://www.kaggle.com/code/linshokaku/faster-hpa-cell-segmentation
2. Spot labeling https://github.com/arjunrajlaboratory/rajlabimagetools
3. requirements.txt

## How to train a model on your dataset
### Configure your dataset

```
mkdir name_project
cd name_project
mkdir raw 

cd Spot-detection-for-FISH-image
python get_meta.py
```
Before running: upload your data in folder raw and modify get_meta.py

raw/ should contain fov folders, each contains a set of images of channel dapi, trans, cy, alexa,... and the images are named in format 'dapi001.tif','alexa001.tif'.

A summary of raw image directory, named 'meta.csv', will be output in name_project/

### Cell segmentation
Install the package mentioned in dependency (reset path for model weights is required)
```
python run_seg.py
```
Segmentation results will be saved in name_project/seg and figures for visualizing in name_project/view/seg

### Spot labeling
Download and configure path for the package mentioned in dependency, then add or substitute the 3 documents of matlab_add/ in the rajlabimagetools/
```
matlab Main();
```
Before running: Modify the last column todo in 'meta.csv' to select fov and modify Main.m to read correctly 'meta.csv'  

Results will be saved in name_project/spot_detect_matlab/

### Train and test with visualization
```
python main.py
```
Labeled cells are assigned to train/valid/test set, then we generate quasi-segmentation mask and save temporary .npz documents for training (including keys: 2d image, synthetic spot segmentation matrix, true coordinates) for each cell respectively in name_project/train_set/,name_project/valid_set/,name_project/test_set/

Model will be saved in name_project/model and figures for visualizing in name_project/view/pred

### Predict and analyse
```
python stat.py
```
Before running: Modify the path of images to predict before running the code

