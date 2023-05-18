# Spot-detection-for-FISH-image
mkdir name_project
cd name_project
mkdir raw

raw/ should contain fov folders. A folder contains 3d images of all channels(dapi, trans, cy, alexa,...)for one fov and the images are named in format 'dapi001.tif','alexa001.tif'

cd SpotLoc-Master
python get_meta.py
Modify the path before running
a summary of raw image directory, named 'meta.csv', will be found in name_project/

python run_seg.py
segmentation results will be saved in name_project/seg and figures for visualizing in name_project/view/seg
You can select any number of fov to label spots by modifying the last column todo in 'meta.csv'

matlab Main();
automatic spot detection followed by manual thresholding adjusting is launch. Results will be saved in name_project/spot_detect_matlab

python main.py
Ensemble of train and test, including visualization
labeled cells are assigned to train/valid/test set, direct input of network(2d image, synthetic spot segmentation matrix, true coordinates) are combined and saved respectively in name_project/train_set,name_project/valid_set,name_project/test_set

predictions are saved in name_project/spot_pred, figures for visualizing in name_project/view/pred

python stat.py


