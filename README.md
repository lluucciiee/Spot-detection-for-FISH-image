# Spot-detection-for-FISH-image
## Introduction
## Dependency
## How to train a model on your dataset
### Configure your dataset

```
mkdir name_project
cd name_project
mkdir raw 

cd Spot-detection-for-FISH-image
python get_meta.py
```
Upload your data in folder raw and modify the path before running get_meta.py

raw/ should contain fov folders, each contains a set of 3d images of channel dapi, trans, cy, alexa,... and the images are named in format 'dapi001.tif','alexa001.tif'

A summary of raw image directory, named 'meta.csv', will be output in name_project/

### Cell segmentation
```
python run_seg.py
```
Segmentation results will be saved in name_project/seg and figures for visualizing in name_project/view/seg

### Spot labeling
```
matlab Main();
```
You can select any number of fov to label spots by modifying the last column todo in 'meta.csv'

Results will be saved in name_project/spot_detect_matlab

### Train and test with visualization
```
python main.py
```
Labeled cells are assigned to train/valid/test set, then we generate quasi-segmentation mask and save temporary .npz documents for training (including keys: 2d image, synthetic spot segmentation matrix, true coordinates) for each cell respectively in name_project/train_set,name_project/valid_set,name_project/test_set

Model will be saved in name_project/model and figures for visualizing in name_project/view/pred

### Predict and analyse
```
python stat.py
```
Modify the path of images to predict before running the code

