## Data Preparation
Please prepare data as instructed in the model training [sub-page](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/TRAINING.md).
The training data need to downloaded from [here](https://drive.google.com/drive/folders/1zyW8ryGXLq4bumWnVGUROpDNdubWUExg?usp=sharing) and placed under "${EvoSkeleton}/data" folder:
   ```
   ${EvoSkeleton}
   ├── data
      ├── human3.6M
          ├── your downloaded files
   ```
## Model Preparation
During data space exploration, a function that evaluates the validity of 3D skeletons is used. This function is parametrized with a model propsosed by Ijaz Akhter in CVPR 2015. You need to download the "constraints" folder from [here](https://drive.google.com/drive/folders/1MUcR9oBNUpTAJ7YUWdyVLKCQW874FszI?usp=sharing) which contains the model parameters and place them under "${EvoSkeleton}/resources" folder:
   ```
   ${EvoSkeleton}
   ├── recources
      ├── constraints
   ```
## Dataset Evolution
To evolve from a population of 3D skeleton (default to Human 3.6M data), go to "${EvoSkeleton}/tools" folder and run
```bash
python evolve.py -generate True
```
### Controling the Initial Population
To reproduce the experiments in different settings, you need to specify the choice of initial population.
For weakly-supervised experiments, you should only start with subject 1 (S1) data (a subset of H36M training data) as follows
```bash
python evolve.py -generate True -WS True -SS "S1"
```
You can even start with extremly scarce data (e.g., 1 percent of S1 data) as follows
```bash
python evolve.py -generate True -WS True -SS "0.01S1"
```

After finished dataset evolution, you can use the saved file for [training](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/TRAINING.md) to see how dataset evolution might help improve model generalization especially when the initial population is scarce.

## Reference

      @inproceedings{akhter2015pose,
        title={Pose-conditioned joint angle limits for 3D human pose reconstruction},
        author={Akhter, Ijaz and Black, Michael J},
        booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
        pages={1446--1455},
        year={2015}
      }
