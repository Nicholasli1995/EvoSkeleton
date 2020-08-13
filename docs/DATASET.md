## Download
You can access the dataset [here](https://drive.google.com/file/d/1JRJuL69J0drZOAUT8VDK5ywmxt-Gm7s-/view?usp=sharing).

## Folder Structure
- imgs - Contains the collected images
- annotation.npy - Contains the pose annotation

   ```
   ${U3DPW}
   ├── imgs
   ├── annotation.npy
   ```
## Annotation
The annotation file is a Python dictionary that has the following format:
p2d is a numpy array of shape (num_keypoints, 2) that stores the image coordinates of the 2D key-points. Each row in the array stores (x, y) coordinate of the corresponding key-point. These key-points are re-annotated with a style similar to that of Human 3.6M, and can be accessed through key 'h36m'.
lsp is a boolean flag that indicates whether the image is collected from [Leeds Sport Pose dataset](https://sam.johnson.io/research/lsp.html) or not.

```
{
'image_name1':{'p2d':array1, 'lsp':True/False, 'h36m':array2},
'image_name2':{'p2d':array3, 'lsp':True/False, 'h36m':array4},
...
}
```

## Key-point Semantics
The name of the Human 3.6M style key-points are:

| Index | Keypoint |
|---|-------------|
| 0 | Hip Center  | 
| 1 | Right Hip   | 
| 2 | Right Knee  |
| 3 | Right Ankle |
| 4 | Left Hip    |
| 5 | Left Knee   |
| 6 | Left Ankle  | 
| 7 | Spine       |
| 8 | Thorax      |
| 9 | Neck        |  
| 10 | Head Top   |
| 11 | Left SHoulder |
| 12 | Left Elbow |
| 13 | Left Wrist |
| 14 | Right Shoulder |
| 15 | Right Elbow|
| 16 | Right Wrist| 
