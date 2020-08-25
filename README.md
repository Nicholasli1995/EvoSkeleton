# EvoSkeleton
This is the project website containing relevant files for the CVPR 2020 paper "Cascaded Deep Monocular 3D Human Pose Estimation with Evolutionary Training Data". The usage and instructions are organized into several parts serving distinct purposes. Please visit the corresponding sub-page for details. 

*TODO: More comments and documentation will be added in future updates.* 

## Hierarchical Human Representation and Data Synthesis
[This sub-page](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/HHR.md) gives instructions on how to use the 3D skeleton model and how the evolution algorithm can be used to discover novel data.
<p align="center">
  <img src="https://github.com/Nicholasli1995/EvoSkeleton/blob/master/imgs/hierarchical.jpg"  width="492" height="304" />
</p>
  
## Cascaded 2D-to-3D Lifting 
[This sub-page](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/TRAINING.md) details how to train a cascaded model to lift 2D key-points to 3D skeletons.

If you do not want to prepare synthetic data and train the model by yourself, you can access an examplar pre-trained model [here](https://drive.google.com/file/d/158oCTK-9Y8Bl9qxidoHcXfqfeeA7qT93/view?usp=sharing) and follow the inference section in the [document](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/TRAINING.md).  
<p align="center">
  <img src="https://github.com/Nicholasli1995/EvoSkeleton/blob/master/imgs/architecture.jpg"/>
</p>

## Dataset: Unconstrained 3D Pose in the Wild
[This sub-page](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/DATASET.md) describs the newly collected dataset Unconstrained 3D Human Pose in the Wild (U3DPW) and gives instructions on how to download it.
<p align="center">
  <img src="https://github.com/Nicholasli1995/EvoSkeleton/blob/master/imgs/U3DPW.png"/>
</p>

## Interactive Annotation Tool
[This sub-page](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/ANNOTATOR.md) provides usage of an annotator that can be used to label 2D and 3D skeleton for an input image. U3DPW was obtained created with this tool and this tool may help increasing the scale of 3D annotation for in-the-wild images.
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
