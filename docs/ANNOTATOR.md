The annotator is composed of three parts:
1. 2D annotation: interactively annotate 2D key-points for RGB images
2. 3D parameter fitting: obtain coarse 3D skeleton fitting results based on SMPLify.
3. 3D annotation: interactively modify 3D parameters.

## 2D Keypoints Annotation
Users can annotate 2D Keypoints of images by running the script `annotate_2d.py` under ${EvoSkeleton}/tools. 
```bash
python annotate_2d.py -d DATASET_PATH
```
DATASET_PATH is the path to the folder containing images.
<p align="center">
  <img src="https://github.com/Nicholasli1995/EvoSkeleton/blob/master/imgs/annotator_2d.gif"/>
</p>

Users can annotate 2D Keypoints in the following order by clicking on the image:
Right Ankle, Right Knee, Right Hip, Left Hip, Left Knee, Left Ankle, Right Wrist, Right Elbow, Right Shoulder, 
Left Shoulder, Left Elbow, Left Wrist, Neck, Head top, Spine, Thorax, Nose

Other keyborad short-cuts are:
Press Q to exit the tool.
Press N to go to the next image.
Press Z to save the annotation.
Press C to erase all of the assigned keypoints from the image and start over.

## Coarse 3D Keypoints Estimation
Manually annotating 3D skeleton from scratch is time-consuming, thus we use a tool to obtain an initial 3D pose estimation given 2D annotation. Any method that outputs 3D pose inference given 2D key-points can be employed.

Here we use SMPLify to estimate coarse 3D skeleton. You need to set up a Python 2.7 environment where a spec-list can be used as your reference if you use Anaconda. Then you need to install chumpy and opendr using pip:
```bash
pip install chumpy
pip install opendr
```
After setting up the environment, you need to download the SMPL model files here and organize your project files as follows:
  ```
   ${EvoSkeleton}
   ├── libs
      ├── annotator
          ├── smplify
              ├── models
                  ├── basicModel_neutral_lbs_10_207_0_v1.0.0.pkl 
                  ├── gmm_08.pkl 
   ```
Make sure you have the images and annotations folder in the corresponding dataset folder. As it is compatible in Python 2, run the program in the Python 2 Anaconda environment `conda activate py2; python process.py -d DATASET_PATH`. The DATASET_PATH is the path to the dataset folder. The output results is written in DATASET_PATH/processed folder.

### 3D Keypoints Interactive Viewing
Users can run the 3D interactive tool to view the 3D keypoints result and adjust it to match with the image. The program is ran in Python 3, by running the command `python interactive.py -d DATASET_PATH` where DATASET_PATH is the path to the dataset folder. Users can select the limb to be adjusted and arrow keys to change the angle of said limb. (Todo: Update this section to be more comprehensive). Pressing `q` will save the adjusted keypoints in DATASET_PATH/processed.
