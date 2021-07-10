[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cascaded-deep-monocular-3d-human-pose-1/weakly-supervised-3d-human-pose-estimation-on)](https://paperswithcode.com/sota/weakly-supervised-3d-human-pose-estimation-on?p=cascaded-deep-monocular-3d-human-pose-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cascaded-deep-monocular-3d-human-pose-1/monocular-3d-human-pose-estimation-on-human3)](https://paperswithcode.com/sota/monocular-3d-human-pose-estimation-on-human3?p=cascaded-deep-monocular-3d-human-pose-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cascaded-deep-monocular-3d-human-pose-1/3d-human-pose-estimation-on-human36m)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-human36m?p=cascaded-deep-monocular-3d-human-pose-1)
# EvoSkeleton
This is the project website containing relevant files for the CVPR 2020 paper "Cascaded deep monocular 3D human pose estimation with evolutionary training data". The usage and instructions are organized into several parts serving distinct purposes. Please visit the corresponding sub-page for details. For Q&A, go to [discussions](https://github.com/Nicholasli1995/EvoSkeleton/discussions). If you believe there is a technical problem, submit to [issues](https://github.com/Nicholasli1995/EvoSkeleton/issues). 

News:

(2021-04-08): Release v-1.0. The support for pre-trained models is strengthened. More details have been added to the supplementary material.
  
## Cascaded 2D-to-3D Lifting
[This sub-page](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/TRAINING.md) details how to train a cascaded model to lift 2D key-points to 3D skeletons on H36M.

If you do not want to prepare synthetic data and train the model by yourself, you can access an examplar pre-trained model [here](https://drive.google.com/file/d/158oCTK-9Y8Bl9qxidoHcXfqfeeA7qT93/view?usp=sharing) and follow the instructions in the [document](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/TRAINING.md). This model can be used for in-the-wild inference as well as reproducing the results on MPI-INF-3DHP. The evaluation metric for MPI-INF-3DHP can be accessed [here](https://github.com/chenxuluo/OriNet-demo/tree/master/src/test_util).
<p align="center">
  <img src="https://github.com/Nicholasli1995/EvoSkeleton/blob/master/imgs/architecture.jpg"/>
</p>

Performance on H36M ([Link to pre-trained models](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/Zoo.md))
| Protocol \#1| Avg.|Dir. | Disc| Eat| Greet| Phone| Photo | Pose | Purch.| Sit| SitD.| Smoke| Wait| WalkD.| Walk | WalkT.| 
|-------------------------------------------------------------|------------------|------------------|---------------|------------------|---------------|---------------|------|---------------|------------------|------------------|---------------|---------------|---------------|---------------|---------------|---------------|
| [Martinez](https://github.com/una-dinosauria/3d-pose-baseline) et al. (ICCV'17)   |62.9| 51.8 | 56.2| 58.1| 59.0   | 69.5 | 78.4| 55.2 | 58.1  | 74.0 | 94.6| 62.3 | 59.1  | 65.1 | 49.5 | 52.4  |
| Ours (S15678)                                        |**49.7**|**45.6**|**44.6**|**49.3**|**49.3**|**52.5**|**58.5**|**46.4**|**44.3**|**53.8**|**67.5**|**49.4**|**46.1**|**52.5**|**41.4**|**44.4**|

| Protocol \#2| Avg.|Dir. | Disc| Eat| Greet| Phone| Photo | Pose | Purch.| Sit| SitD.| Smoke| Wait| WalkD.| Walk | WalkT.| 
|-------------------------------------------------------------|------------------|------------------|---------------|------------------|---------------|---------------|------|---------------|------------------|------------------|---------------|---------------|---------------|---------------|---------------|---------------|
| [Martinez](https://github.com/una-dinosauria/3d-pose-baseline) et al. (ICCV'17)   |47.7| 39.5 | 43.2 | 46.4 | 47.0 | 51.0| 56.0 | 41.4 | 40.6 | 56.5 | 69.4 | 49.2 | 45.0  | 49.5 | 38.0  | 43.1  |
| Ours (S15678)                                        |**37.7** |**34.2**|**34.6**|**37.3**|**39.3**|**38.5**|**45.6**|**34.5**|**32.7**|**40.5**|**51.3**|**37.7**|**35.4**|**39.9**|**29.9**|**34.5**|

## Hierarchical Human Representation and Data Synthesis
[This sub-page](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/HHR.md) gives instructions on how to use the 3D skeleton model and how the evolution algorithm can be used to discover novel data.
<p align="center">
  <img src="https://github.com/Nicholasli1995/EvoSkeleton/blob/master/imgs/hierarchical.jpg"  width="394" height="243" />
</p>

## 2D Human Pose Estimation on H3.6M

[This page](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/2DHPE.md) shows how to perform 2D human pose estimation on Human 3.6M dataset with the pre-trained high-resolution heatmap regression model. The highly accurate 2D joint predictions may benefit your 3D human pose estimation project.
<p align="center">
  <img src="https://github.com/Nicholasli1995/EvoSkeleton/blob/master/imgs/h36m2dpose2.png" width="789" height="208"/>
</p>

| Method                    | Parameters| FLOPs|Average Joint Localization Error (pixels) |
| ------------------------- | ---------------| --------------| --------------| 
| CPN (CVPR' 18)            | -|-| 5.4           |
| Ours (HRN + U + S)           |63.6M| 32.9G           | **4.4**        |



## Dataset: Unconstrained 3D Pose in the Wild
[This sub-page](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/DATASET.md) describs the newly collected dataset Unconstrained 3D Human Pose in the Wild (U3DPW) and gives instructions on how to download it.
<p align="center">
  <img src="https://github.com/Nicholasli1995/EvoSkeleton/blob/master/imgs/U3DPW.png"/>
</p>

## Interactive Annotation Tool
[This sub-page](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/ANNOTATOR.md) provides usage of an annotation tool that can be used to label 2D and 3D skeleton for an input image. U3DPW was obtained created with this tool and this tool may help increasing the scale of 3D annotation for in-the-wild images.
<p align="center">
  <img src="https://github.com/Nicholasli1995/EvoSkeleton/blob/master/imgs/tool.gif" width="531" height="291"/>
</p>

## Environment
- Python 3.6
- Numpy 1.16
- PyTorch 1.0.1
- CUDA 9

For a complete list of other python packages, please refer to [spec-list.txt](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/spec-list.txt). The recommended environment manager is Anaconda, which can create an environment using the provided spec-list. Certain tool in this project may need other specified environment, which is detailed in its corresponding page.

## License
A MIT license is used for this repository. However, certain third-party dataset (Human 3.6M) and tool (SMPLify) are subject to their respective licenses and may not grant commercial use.

## Citation
Please star this repository and cite the following paper in your publications if it helps your research:

    @InProceedings{Li_2020_CVPR,
    author = {Li, Shichao and Ke, Lei and Pratama, Kevin and Tai, Yu-Wing and Tang, Chi-Keung and Cheng, Kwang-Ting},
    title = {Cascaded Deep Monocular 3D Human Pose Estimation With Evolutionary Training Data},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
    }
    
Link to the paper:
[Cascaded Deep Monocular 3D Human Pose Estimation With Evolutionary Training Data](https://arxiv.org/abs/2006.07778)

Link to the oral presentation video:
[Youtube](https://www.youtube.com/watch?v=erYymlWw2bo)
