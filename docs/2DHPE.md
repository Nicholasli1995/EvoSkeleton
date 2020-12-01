The pre-trained model can be downloaded from [here](https://drive.google.com/file/d/1NjQFCz0GdS7oIdYrK5ouxEI07wYYh4r8/view?usp=sharing) and placed under "${EvoSkeleton}/examples/h36m2Dpose" folder. Your directory should look like this:

   ```
   ${EvoSkeleton}
   ├── examples
      ├── h36m2Dpose
          ├── cropped (prepared testing images from Human 3.6M)
          ├── cfgs.yaml (configuration file)
          ├── final_state.pth (pre-trained high-resolution heatmap regression model)
   ```
   
Then run h36m2Dpose.py at ${EvoSkeleton}/examples
```bash
python h36m2Dpose.py
```

You should expect to see results like [this](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/imgs/h36m2dpose2.png).
