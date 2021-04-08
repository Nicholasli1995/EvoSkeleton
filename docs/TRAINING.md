## Data Preparation
Similar to other repositories ([SimpleBaseline](https://github.com/una-dinosauria/3d-pose-baseline), [TemporalConvolution](https://github.com/facebookresearch/VideoPose3D)) on training 2D-to-3D  networks, we provide pre-processed 2D detections, camera parameters and 3D poses for training. The 2D detections are produced by our modified high-resolution model, while the camera parameters and the 3D poses are taken from [SimpleBaseline](https://github.com/una-dinosauria/3d-pose-baseline).

The training data need to downloaded from [here](https://drive.google.com/drive/folders/1zyW8ryGXLq4bumWnVGUROpDNdubWUExg?usp=sharing) and placed under "${EvoSkeleton}/data" folder. Your directory should look like this:
   ```
   ${EvoSkeleton}
   ├── data
      ├── human3.6M
          ├── cameras.npy (Camera parameters provided by Human 3.6M)
          ├── threeDPose_train.npy (3D skeletons from Human 3.6M training split)
          ├── threeDPose_test.npy (3D skeletons from Human 3.6M test split)
          ├── twoDPose_HRN_train.npy (2D key-point detections obtained from the heatmap regression model for the training split)
          ├── twoDPose_HRN_test.npy (2D key-point detections obtained from the heatmap regression model for the test split)
   ```
   
## Weakly-Supervised Experiments on Human 3.6M Dataset
To compare with other weakly-supervised methods, only a subset of training data (e.g., subject 1 data) is used to simulate an environment where data is scarce. 
To perform training, go to ./tools and run
```bash
python 2Dto3Dnet.py -train True -num_stages 2 -ws True -ws_name "S1"
```
This command performs training on synthetic 2D key-points to remove the influence of the heatmap regression model, whose results correspond to P1* in the performance table. S1 stands for subject 1 data. "num_stages" specify the number of deep learners used in the cascade.
To train on real detections obtained by the high-resolution heatmap regression model, run
```bash
python 2Dto3Dnet.py -train True -num_stages 2 -ws True -ws_name "S1" -twoD_source "HRN"
```
To train on evolved dataset, you need to specify the path to the evolved data as
```bash
python 2Dto3Dnet.py -train True -num_stages 2 -ws True -ws_name "S1" -twoD_source "HRN/synthetic" -evolved_path "YourDataPath"
```
See [this page](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/HHR.md) on how to evolve a dataset.


After data augmentation using the evolved data, we noticed the model generalization improves significantly despite the initial population size is small. Other methods utilize multi-view or temporal consistency instead of data augmentation to supervise deep models when data is scarce. Compared to them, we achieve state-of-the-art performance by synthesizing new data to supervise the deep model. P1 and P2 refers to the two protocols used for calculating average MPJPE over all 15 actions in H36M. 

| Method                    | Avg. MPJPE (P1) |  Avg. MPJPE (P2) |
| ------------------------- | --------------- |  --------------- |
| Rhodin et al. (CVPR' 18)  | -               |  64.6            |
| Kocabas et al. (CVPR' 19) | 65.3            |  57.2            |
| Pavllo et al. (CVPR' 19)  | 64.7            |  -               |
| Li et al. (ICCV' 19)      | 88.8            |  66.5            |
| Ours                      | **60.8**        |  **46.2**        |

## Fully-Supervised Experiments on Human 3.6M Dataset
To train on real detections obtained by the high-resolution heatmap regression model, run
```bash
python 2Dto3Dnet.py -train True -num_stages 2 -twoD_source "HRN"
```
To train on evolved dataset, you need to specify the path to the evolved data as
```bash
python 2Dto3Dnet.py -train True -num_stages 3 -num_blocks 3 -twoD_source "HRN/synthetic" -evolved_path "YourDataPath"
```
Here we increase model capacity with "-num_stages 3 -num_blocks 3" since the training data size is much larger (if you evolve enough generations).
While the improvement using data evolution is less obvious in fully-supervised setting compared with weakly-supervised setting, our cascaded model still achieved competitive performance compared with other 2D-to-3D lifting models.

| Method                     | Avg. MPJPE (P1) |  Avg. MPJPE (P2) |
| -------------------------- | --------------- |  --------------- |
| [Martinez et al.](https://github.com/una-dinosauria/3d-pose-baseline) (ICCV' 17) | 62.9            |  47.7            |
| Yang et al. (CVPR' 18)     | 58.6            |  **37.7**        |
| Zhao et al. (CVPR' 19)     | 57.6            |  -               |
| Sharma et al. (CVPR' 19)   | 58.0            |  40.9            |
| Moon et al. (ICCV' 19)     | 54.4            |  -               |
| Ours                       | **50.9**        |  38.0            |

## Inference Example
If you only want to use a pre-trained model to conduct inference on in-the-wild images (skipping data synthesis and model training), you can download the sample images and a pre-trained checkpoint [here](https://drive.google.com/file/d/158oCTK-9Y8Bl9qxidoHcXfqfeeA7qT93/view?usp=sharing). Un-zip the downloaded file to "${EvoSkeleton}/examples" folder and your directory should look like this:
   ```
   ${EvoSkeleton}
   ├── examples
      ├── imgs (sample images)
      ├── example_annot.npy (2D key-points for the samples)
      ├── example_model.th (pre-trained model)
      ├── stats.npy (model statistics)
      ├── inference.py
   ``` 
Then you can run the following command at "${EvoSkeleton}/examples" to perform inference
```bash
python inference.py
```

<p align="center">
  <img src="https://github.com/Nicholasli1995/EvoSkeleton/blob/master/imgs/example.png"/>
</p>
