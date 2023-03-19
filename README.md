# pointpainting_re  

## 0319:  
1. 重新整理了Kitti数据集：因为pointpainting需要image_03的数据，把它下载好后又重新使用mmdet3d进行了整理。以下是运行结果：  
```
(base) xilm@xilm-MS-7D17:~/mmlab/mmdetection3d-master$ python tools/create_data.py kitti --root-path /home/xilm/kitti/KITTI/ --out-dir /home/xilm/kitti/KITTI/ --extra-tag kitti
Generate info. this may take several minutes.
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3712/3712, 201.2 task/s, elapsed: 18s, ETA:     0s
Kitti info train file is saved to /home/xilm/kitti/KITTI/kitti_infos_train.pkl
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3769/3769, 195.1 task/s, elapsed: 19s, ETA:     0s
Kitti info val file is saved to /home/xilm/kitti/KITTI/kitti_infos_val.pkl
Kitti info trainval file is saved to /home/xilm/kitti/KITTI/kitti_infos_trainval.pkl
Kitti info test file is saved to /home/xilm/kitti/KITTI/kitti_infos_test.pkl
create reduced point cloud for training set
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3712/3712, 356.3 task/s, elapsed: 10s, ETA:     0s
create reduced point cloud for validation set
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3769/3769, 345.2 task/s, elapsed: 11s, ETA:     0s
create reduced point cloud for testing set
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 7518/7518, 333.5 task/s, elapsed: 23s, ETA:     0s
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3712/3712, 105.8 task/s, elapsed: 35s, ETA:     0s
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3769/3769, 105.0 task/s, elapsed: 36s, ETA:     0s
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 7481/7481, 110.0 task/s, elapsed: 68s, ETA:     0s
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 7518/7518, 140.5 task/s, elapsed: 54s, ETA:     0s
Create GT Database of KittiDataset
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3712/3712, 134.6 task/s, elapsed: 28s, ETA:     0s
load 2207 Pedestrian database infos
load 14357 Car database infos
load 734 Cyclist database infos
load 1297 Van database infos
load 488 Truck database infos
load 224 Tram database infos
load 337 Misc database infos
load 56 Person_sitting database infos
```  
2. 
