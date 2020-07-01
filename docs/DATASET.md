## Download
You can access the dataset [here](https://drive.google.com/file/d/1T19RH73F0EUSsxrs3i595nSfkSQplFx0/view?usp=sharing).

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
p2d is a numpy array of shape (num_keypoints, 2) that stores the image coordinates of the 2D key-points. Each row in the array stores (x, y) coordinate of the corresponding key-point.
lsp is a boolean flag that indicates whether the image is collected from [Leeds Sport Pose dataset](https://sam.johnson.io/research/lsp.html) or not.

```
{
'image_name1':{p2d:array1, lsp:True/False},
'image_name2':{p2d:array2, lsp:True/False},
...
}
```

## Key-point Semantics
The name of the key-points are:

| Index | Keypoint | Index | Keypoint |
|---|-------------|----|----------------|
| 0 | Right Ankle | 7  | Right Elbow    |
| 1 | Right Knee  | 8  | Right Shoulder |
| 2 | Right Hip   | 9  | Left SHoulder  |
| 3 | Left Hip    | 10 | Left Elbow     |
| 4 | Left Knee   | 11 | Left Wrist     |
| 5 | Left Ankle  | 12 | Neck           |
| 6 | Right Wrist | 13 | Head Top       |
