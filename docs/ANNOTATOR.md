## Usage
### 2D Keypoints Annotation
Users can annotate 2D Keypoints of images by running `annotate_2d.py`. To use on the default LSD dataset, simply run `python annotate_2d.py`. To run on a custom dataset, run `python annotate_2d.py -d DATASET_PATH` where DATASET_PATH is the path to the dataset folder.

Users can annotate 2D Keypoints in the following order by clicking on the image:
(TODO: add a snapshot of using the annotator for 2D and 3D annotation)
1. Right Ankle
2. Right Knee
3. Right Hip
4. Left Hip
5. Left Knee
6. Left Ankle
7. Right Wrist
8. Right Elbow
9. Right Shoulder
10. Left Shoulder
11. Left Elbow
12. Left Wrist
13. Neck
14. Head top
15. Spine
16. Thorax
17. Nose

Press `c` to erase all of the assigned keypoints from the image and start over. If you are done annotating all the keypoints, press `q` to save the keypoints and move on to the next image.

### SMPLify 3D Keypoints Fitting
Make sure you have the images and annotations folder in the corresponding dataset folder. As it is compatible in Python 2, run the program in the Python 2 Anaconda environment `conda activate py2; python process.py -d DATASET_PATH`. The DATASET_PATH is the path to the dataset folder. The output results is written in DATASET_PATH/processed folder.

### 3D Keypoints Interactive Viewing
Users can run the 3D interactive tool to view the 3D keypoints result and adjust it to match with the image. The program is ran in Python 3, by running the command `python interactive.py -d DATASET_PATH` where DATASET_PATH is the path to the dataset folder. Users can select the limb to be adjusted and arrow keys to change the angle of said limb. (Todo: Update this section to be more comprehensive). Pressing `q` will save the adjusted keypoints in DATASET_PATH/processed.
