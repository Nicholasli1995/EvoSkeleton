This page presents model performance on H36M under various settings. Pre-trained models and instructions for reproduction can also be found.

## Fully-supervised Setting (S15678)

[Download our pre-trained model](https://drive.google.com/drive/folders/1IRKUWrnheD03Dj30LLGlh_LLT1CK6dr5?usp=sharing)

[Download our pre-evolved data](https://drive.google.com/drive/folders/1FKFkmTJQcEdrCvZOSc8cF5OTTjFvyLav?usp=sharing)

Inference command:
```bash
python 2Dto3Dnet.py -evaluate True -twoD_source "HRN" -ckpt_dir "YourMODELPath"
```
Training command ([Docs](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/TRAINING.md)):
```bash
python 2Dto3Dnet.py -train True -num_stages 3 -num_blocks 3 -twoD_source "HRN" -evolved_path "YourDataPath"
```
Data synthesis command ([Docs](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/HHR.md)):
```bash
python evolve.py -SS "S15678" -T 1.5 -SD "YourDataPath" -generate True
```
MPJPE (P1) for each action under fully-supervised setting is shown in the table below.
| Protocol \#1                                      | Dir.             | Disc             | Eat           | Greet            | Phone         | Photo         | Pose | Purch.        | Sit              | SitD.            | Smoke         | Wait          | WalkD.        | Walk          | WalkT.        | Avg.          |
|-------------------------------------------------------------|------------------|------------------|---------------|------------------|---------------|---------------|------|---------------|------------------|------------------|---------------|---------------|---------------|---------------|---------------|---------------|
| [Martinez](https://github.com/una-dinosauria/3d-pose-baseline) et al. (ICCV'17)   | 51.8             | 56.2             | 58.1          | 59.0             | 69.5          | 78.4          | 55.2 | 58.1          | 74.0             | 94.6             | 62.3          | 59.1          | 65.1          | 49.5          | 52.4          | 62.9          |
| [Fang](https://arxiv.org/abs/1710.06513) et al. (AAAI'18)        | 50.1             | 54.3             | 57.0          | 57.1             | 66.6          | 73.3          | 53.4 | 55.7          | 72.8             | 88.6             | 60.3          | 57.7          | 62.7          | 47.5          | 50.6          | 60.4          |
| [Yang](https://arxiv.org/abs/1803.09722) et al. (CVPR'18)               | 51.5             | 58.9             | 50.4          | 57.0             | 62.1          | 65.4          | 49.8 | 52.7          | 69.2             | 85.2             | 57.4          | 58.4          | 43.6          | 60.1          | 47.7          | 58.6          |
| [Pavlakos](https://github.com/geopavlakos/ordinal-pose3d) et al. (CVPR'18)  | 48.5             | 54.4             | 54.4          | 52.0             | 59.4          | 65.3          | 49.9 | 52.9          | 65.8             | 71.1             | 56.6          | 52.9          | 60.9          | 44.7          | 47.8          | 56.2          |
| [Lee](https://openaccess.thecvf.com/content_ECCV_2018/papers/Kyoungoh_Lee_Propagating_LSTM_3D_ECCV_2018_paper.pdf) et al. (ECCV'18)        | 40.2    | 49.2             | 47.8          | 52.6             | 50.1 | 75.0          | 50.2 | 43.0          | 55.8             | 73.9             | 54.1          | 55.6          | 58.2          | 43.3          | 43.3 | 52.8          |
| [Zhao](https://arxiv.org/abs/1904.03345) et al. (CVPR'19)           | 47.3             | 60.7             | 51.4          | 60.5             | 61.1          | 49.9 | 47.3 | 68.1          | 86.2             | 55.0             | 67.8          | 61.0          | 42.1 | 60.6          | 45.3          | 57.6          |
| [Sharma](https://arxiv.org/abs/1904.01324) et al. (ICCV'19)    | 48.6             | 54.5             | 54.2          | 55.7             | 62.6          | 72.0          | 50.5 | 54.3          | 70.0             | 78.3             | 58.1          | 55.4          | 61.4          | 45.2          | 49.7          | 58.0          |
| [Moon](https://github.com/mks0601/3DMPPE_POSENET_RELEASE) et al. (ICCV'19)    | 51.5             | 56.8             | 51.2          | 52.2             | 55.2          | 47.7 | 50.9 | 63.3          | 69.9             | 54.2    | 57.4          | 50.4          | 42.5          | 57.5          | 47.7          | 54.4          |
| [Liu](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550324.pdf) et al. (ECCV'20)      | 46.3             | 52.2             | 47.3 | 50.7             | 55.5          | 67.1          | 49.2 | 46.0          | 60.4             | 71.1             | 51.5          | 50.1          | 54.5          | 40.3 | 43.7          | 52.4          |
| Ours (S15678)                                        |45.6|44.6|49.3|49.3|52.5|58.5|46.4|44.3|53.8|67.5|49.4|46.1|52.5|41.4|44.4| 49.7 |

MPJPE (P2) for each action under fully-supervised setting is shown in the table below.

| Protocol \#2                                       | Dir.             | Disc             | Eat           | Greet            | Phone         | Photo         | Pose | Purch.        | Sit              | SitD.            | Smoke         | Wait          | WalkD.        | Walk          | WalkT.        | Avg.          |
|-------------------------------------------------------------|------------------|------------------|---------------|------------------|---------------|---------------|------|---------------|------------------|------------------|---------------|---------------|---------------|---------------|---------------|---------------|
| [Martinez](https://github.com/una-dinosauria/3d-pose-baseline) et al. (ICCV'17)   | 39.5             | 43.2             | 46.4          | 47.0             | 51.0          | 56.0          | 41.4 | 40.6          | 56.5             | 69.4             | 49.2          | 45.0          | 49.5          | 38.0          | 43.1          | 47.7          |
| [Fang](https://arxiv.org/abs/1710.06513) et al. (AAAI'18)        | 38.2             | 41.7             | 43.7          | 44.9             | 48.5          | 55.3          | 40.2 | 38.2          | 54.5             | 64.4             | 47.2          | 44.3          | 47.3          | 36.7          | 41.7          | 45.7          |
| [Pavlakos](https://github.com/geopavlakos/ordinal-pose3d) et al. (CVPR'18)  | 34.7             | 39.8             | 41.8          | 38.6             | 42.5          | 47.5          | 38.0 | 36.6          | 50.7             | 56.8             | 42.6          | 39.6          | 43.9          | 32.1          | 36.5          | 41.8          |
| [Yang](https://arxiv.org/abs/1803.09722) et al. (CVPR'18)               | 26.9             | 30.9             | 36.3          | 39.9             | 43.9          | 47.4          | 28.8 | 29.4          | 36.9             | 58.4             | 41.5          | 30.5          | 29.5          | 42.5          | 32.2          | 37.7          |
| [Sharma](https://arxiv.org/abs/1904.01324) et al. (ICCV'19)    | 35.3             | 35.9             | 45.8          | 42.0             | 40.9          | 52.6          | 36.9 | 35.8          | 43.5             | 51.9             | 44.3          | 38.8          | 45.5          | 29.4          | 34.3          | 40.9          |
| [Cai](https://openaccess.thecvf.com/content_ICCV_2019/papers/Cai_Exploiting_Spatial-Temporal_Relationships_for_3D_Pose_Estimation_via_Graph_Convolutional_ICCV_2019_paper.pdf) et al. (ICCV'19)     | 35.7             | 37.8             | 36.9          | 40.7             | 39.6          | 45.2          | 37.4 | 34.5          | 46.9             | 50.1 | 40.5          | 36.1          | 41.0          | 29.6          | 33.2          | 39.0          |
| [Liu](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550324.pdf) et al. (ECCV'20)    | 35.9             | 40.0             | 38.0          | 41.5             | 42.5          | 51.4          | 37.8 | 36.0          | 48.6             | 56.6             | 41.8          | 38.3          | 42.7          | 31.7          | 36.2          | 41.2          |
| Ours (S15678)                                        |34.2|34.6|37.3|39.3|38.5|45.6|34.5|32.7|40.5|51.3|37.7|35.4|39.9|29.9|34.5| 37.7 |

## Weakly-supervised Setting (S1)

[Download our pre-trained model](https://drive.google.com/drive/folders/1PZoiizPKeoFTsvnFKIxaRDNbyb0Csx50?usp=sharing)

[Download our pre-evolved data](https://drive.google.com/drive/folders/1nTW2CCCT_sbJ1CejhuiQLTgDDU5sJjZj?usp=sharing)

Inference command:
```bash
python 2Dto3Dnet.py -evaluate True -twoD_source "HRN" -ckpt_dir "YourMODELPath" 
```
Training command ([Docs](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/TRAINING.md)):
```bash
python 2Dto3Dnet.py -train True -num_stages 2 -ws True -ws_name "S1" -twoD_source "HRN" -evolved_path "YourDataPath"
```
Data synthesis command ([Docs](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/HHR.md)):
```bash
python evolve.py -generate True -WS True -SS "S1"
```
MPJPE (P1) for each action under weakly-supervised setting is shown in the table below.
| Protocol \#1                                           | Dir.             | Disc             | Eat           | Greet            | Phone         | Photo            | Pose | Purch.        | Sit              | SitD. | Smoke         | Wait          | WalkD.        | Walk          | WalkT.        | Avg.          |
|--------------------------------------------------------|------------------|------------------|---------------|------------------|---------------|------------------|------|---------------|------------------|-------|---------------|---------------|---------------|---------------|---------------|---------------|
| [Kocabas](https://arxiv.org/abs/1903.02330) et al. (CVPR'19)  | -                | -                | -             | -                | -             | -                | -    | -             | -                | -     | -             | -             | -             | -             | -             | 65.3          |
| [Pavllo](https://arxiv.org/abs/1811.11742) et al. (CVPR'19)  | -                | -                | -             | -                | -             | -                | -    | -             | -                | -     | -             | -             | -             | -             | -             | 64.7          |
| [Li](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_On_Boosting_Single-Frame_3D_Human_Pose_Estimation_via_Monocular_Videos_ICCV_2019_paper.pdf) et al. (ICCV'19)          | 70.4             | 83.6             | 76.6          | 78.0             | 85.4          | 106.1   | 72.2 | 103.0         | 115.8            | 165.0 | 82.4          | 74.3          | 94.6          | 60.1          | 70.6          | 88.8          |
| Ours (S1)                                       | 52.8             | 56.6 | 54.0 | 57.5 | 62.8 | 72.0 | 55.0 | 61.3 | 65.8 | 80.7  | 58.9 | 56.7 | 69.7 | 51.6 | 57.2 | 60.8 |

MPJPE (P2) for each action under fully-supervised setting is shown in the table below.

| Protocol \#2                                           | Dir.             | Disc             | Eat           | Greet            | Phone         | Photo            | Pose | Purch.        | Sit              | SitD. | Smoke         | Wait          | WalkD.        | Walk          | WalkT.        | Avg.          |
|--------------------------------------------------------|------------------|------------------|---------------|------------------|---------------|------------------|------|---------------|------------------|-------|---------------|---------------|---------------|---------------|---------------|---------------|
| [Rhodin](https://arxiv.org/abs/1803.04775) et al. (CVPR'18)  | -                | -                | -             | -                | -             | -                | -    | -             | -                | -     | -             | -             | -             | -             | -             | 64.6          |
| [Kocabas](https://arxiv.org/abs/1903.02330) et al. (CVPR'19)  | -                | -                | -             | -                | -             | -                | -    | -             | -                | -     | -             | -             | -             | -             | -             | 57.2          |
| [Li](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_On_Boosting_Single-Frame_3D_Human_Pose_Estimation_via_Monocular_Videos_ICCV_2019_paper.pdf) et al. (ICCV'19)          | -                | -                | -             | -                | -             | -                | -    | -             | -                | -     | -             | -             | -             | -             | -             | 66.5          |
| Ours (S1)                                       | 40.2             | 43.4 | 41.9| 46.1 | 48.2 | 55.1 | 42.8 | 42.6 | 49.6 | 61.1  | 44.5 | 43.2 | 51.5 | 38.1 | 44.4 | 46.2 |

