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

Here we use SMPLify to estimate coarse 3D skeleton. You need to set up a Python 2.7 environment where a [spec-list](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/libs/annotator/smpl-spec-list.txt) can be used as your reference if you use Anaconda. Then you need to install chumpy and opendr using pip:
```bash
pip install chumpy
pip install opendr
```
After setting up the environment, you need to download the SMPL model files [here](https://drive.google.com/drive/folders/12qJQP-h4E43FkgE74tybQUjeP_pnqAor?usp=sharing) and organize your project files as follows:
  ```
   ${EvoSkeleton}
   ├── libs
      ├── annotator
          ├── smplify
              ├── models
                  ├── basicModel_neutral_lbs_10_207_0_v1.0.0.pkl 
                  ├── gmm_08.pkl 
      ├── fit_3d.py               
                  
   ```
Then one can run fit_3d under ${EvoSkeleton}/libs/annotator/fit_3d.py to fit the SMPL model
```bash
python fit_3d.py -dataset_dir DATASET_PATH -model_dir MODEL_PATH
```
DATASET_PATH is the path to the folder containing the annotated 2D key-point file "annotation.npy".
MODEL_PATH is the path to the used SMPL model (for example, basicModel_neutral_lbs_10_207_0_v1.0.0.pkl). There are other available models depending on the gender of the subject.
The fitting process can be shown during running and the file annotation.npy will be updated with 3D parameters.
<p align="center">
  <img src="https://github.com/Nicholasli1995/EvoSkeleton/blob/master/imgs/fitted.png"/>
</p>

## 3D Skeleton Interactive Annotation 
After you have obtained fitted parameters for your dataset, you can modify the 3D parameters interactively with this tool. If you just want to try this tool without finishing the above steps, you can play with the pre-fitted parameters [here](https://drive.google.com/file/d/1OJokg844KDpRG3YQsZNpXwFXlVF8iOH5/view?usp=sharing) for U3DPW images. To start the tool, go to ${EvoSkeleton}/tools and run
```bash
python annotate_3D.py -dataset_dir DATASET_PATH
```
DATASET_PATH is the path to the folder containing the fitted parameters "fitted.npy". The tool will select one unannotated image, display it and start the interactive process. You can modify the global orientation as well as the limb orientation of the 3D skeleton. A 2D image with projected key-points will be plotted on-line so that you can check if your annotation is reasonable or not. 

Some important keyborad short-cuts are:

Press number keys (2-9) to select which bone vector to rotate.

Press 0 so that the 3D skeleton will rotate as a whole.

Press arrow keys (up and down) to rotate the bone vector.

Press "m" to save an updated annotation file.

Other keyboard inputs are detailed in annotate_3D.py

<p align="center">
  <img src="https://github.com/Nicholasli1995/EvoSkeleton/blob/master/imgs/tool.gif"/>
</p>
