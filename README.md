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
2. 运行`python train.py --cfg_file cfgs/kitti_models/pointpillar_painted.yaml`失败  
- 首先修改了一些库的版本；  
- 然后把kitti数据集的地址进行了修改（在`painted_kitti_dataset.yaml`文件中修改）；  
- 发现运行train.py的时候 又报错了：  
```
Traceback (most recent call last):                                                                 | 0/928 [00:00<?, ?it/s]
  File "/home/xilm/fuxian/PointPainting/detector/tools/train.py", line 198, in <module>
    main()
  File "/home/xilm/fuxian/PointPainting/detector/tools/train.py", line 153, in main
    train_model(
  File "/home/xilm/fuxian/PointPainting/detector/tools/train_utils/train_utils.py", line 86, in train_model
    accumulated_iter = train_one_epoch(
  File "/home/xilm/fuxian/PointPainting/detector/tools/train_utils/train_utils.py", line 19, in train_one_epoch
    batch = next(dataloader_iter)
  File "/home/xilm/anaconda3/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 521, in __next__
    data = self._next_data()
  File "/home/xilm/anaconda3/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1203, in _next_data
    return self._process_data(data)
  File "/home/xilm/anaconda3/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1229, in _process_data
    data.reraise()
  File "/home/xilm/anaconda3/lib/python3.9/site-packages/torch/_utils.py", line 434, in reraise
    raise exception
KeyError: Caught KeyError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/home/xilm/anaconda3/lib/python3.9/site-packages/torch/utils/data/_utils/worker.py", line 287, in _worker_loop
    data = fetcher.fetch(index)
  File "/home/xilm/anaconda3/lib/python3.9/site-packages/torch/utils/data/_utils/fetch.py", line 49, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/xilm/anaconda3/lib/python3.9/site-packages/torch/utils/data/_utils/fetch.py", line 49, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/xilm/fuxian/PointPainting/detector/pcdet/datasets/kitti/painted_kitti_dataset.py", line 368, in __getitem__
    sample_idx = info['point_cloud']['lidar_idx']
KeyError: 'lidar_idx'
```  
- 找到了解决方案：是因为我使用mmdet3d的分割方法分割的数据，实际应该使用Openpcdet的分割方法；  
```
The pkl files may be generated by mmdetection3d in which the keys are different.
Delete or move all the pkl files. Re-generate the data infos by running the following command:
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```  
- 使用`python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml`重新分割。  
- 这里修改路径应该在`detector/pcdet/datasets/kitti/kitti_dataset.py`的434行`create_kitti_infos`中修改；  
```
Database Pedestrian: 2207
Database Car: 14357
Database Cyclist: 734
Database Van: 1297
Database Truck: 488
Database Tram: 224
Database Misc: 337
Database Person_sitting: 56
---------------Data preparation Done---------------
```  
- 人傻了，我应该先painted lidar信息的，不然train的时候找不到painted lidar...  
- 在paint.py中修改kitti数据集的路径(第17行)，就开始慢慢烤gpu了；  
- 三个半小时，painted完毕！  
```
(base) xilm@xilm-MS-7D17:~/fuxian/PointPainting/painting$ python painting.py
Using Segmentation Network -- deeplabv3plus
load checkpoint from local path: ./mmseg/checkpoints/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth
  0%|                                                                                             | 0/7481 [00:00<?, ?it/s]/home/xilm/fuxian/PointPainting/painting/painting.py:94: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  output_permute = torch.tensor(result[0]).permute(1,2,0) # H, W, 19
100%|████████████████████████████████████████████████████████████████████████████████| 7481/7481 [3:25:43<00:00,  1.65s/it]
```  
- 接下来开始准备kitti数据集：`python -m pcdet.datasets.kitti.painted_kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/painted_kitti_dataset.yaml`  
```
Database Pedestrian: 2207
Database Car: 14357
Database Cyclist: 734
Database Van: 1297
Database Truck: 488
Database Tram: 224
Database Misc: 337
Database Person_sitting: 56
---------------Data preparation Done---------------
```  
- 准备训练：`python train.py --cfg_file cfgs/kitti_models/pointpillar_painted.yaml`  
## 0320:  
- 训练完毕，获得ckpt。  
- 这里我把我的权重文件贴出来：链接: https://pan.baidu.com/s/1F8L77nrRQmrY4vIDnGHrZw?pwd=grv2 提取码: grv2   
## 0321:  
- 绘制demo：`python demo.py --cfg_file cfgs/kitti_models/pointpillar_painted.yaml --ckpt ../output/kitti_models/pointpillar_painted/default/ckpt/checkpoint_epoch_80.pth --data_path /home/xilm/kitti/KITTI/training/painted_lidar/000000.npy --ext .npy`  
```
2023-03-21 11:04:37,869   INFO  -----------------Quick Demo of OpenPCDet-------------------------
2023-03-21 11:04:37,869   INFO  Total number of samples:        1
/home/xilm/anaconda3/lib/python3.9/site-packages/torch/functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2157.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2023-03-21 11:04:39,778   INFO  ==> Loading parameters from checkpoint ../output/kitti_models/pointpillar_painted/default/ckpt/checkpoint_epoch_80.pth to CPU
2023-03-21 11:04:39,811   INFO  ==> Checkpoint trained from version: pcdet+0.3.0+0000000
2023-03-21 11:04:39,860   INFO  ==> Done (loaded 127/127)
2023-03-21 11:04:39,869   INFO  Visualized sample index:        1
[[ 38  38  38 255]
 [ 38  38  38 255]
 [ 38  38  38 255]
 ...
 [ 39  39  39 255]
 [ 39  39  39 255]
 [ 39  39  39 255]]
2023-03-21 11:05:53,338   INFO  Demo done.
```  
![927a8aa440e022d92a73f356d1b1c63](https://user-images.githubusercontent.com/96283702/226508906-ac1ee7e1-06b6-4534-9eef-b3fe9ae32420.png)  

以上就是完整的pointpainting复现过程。  
## 0330：  
- 阅读了painting.py的代码，写了注释。  
- 准备复现pointpainting使用sequence的方法，用于在RViz上实现。  
- 安装了一下午tensorrt，头秃。。。  
- 复现了demo.py, demo_video.py, 以及visualizer.py，结果如下，可以看到着色的点云；不过需要注意，只有视野范围内的点云被着色了，也就是车前方-45～45度的视角。  
![](https://github.com/XxxuLimei/pointpainting_re/blob/main/picture/Screenshot%20from%202023-03-30%2021-13-44.png)  
## 0418:  
1. 准备做一组消融实验，就是控制语义分割网络不变，然后变点云检测方法，以及换过来。  
2. 首先是`deeplabv3`
## 0419：  
1. painted.py完毕;  
2. 准备检测：首先使用pointpillar_painted配置文件进行检测。  
- 经过对比发现，`pointpillar_painted`仅仅在`DATA_CONFIG->_BASE_CONFIG_->`以及`DATA_CONFIG->DATA_AUGMENTOR->AUG_CONFIG_LIST->NUM_POINT_FEATURES`两处进行了修改。  
- 运行`python -m pcdet.datasets.kitti.painted_kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/painted_kitti_dataset.yaml`  
- 运行`python train.py --cfg_file cfgs/kitti_models/pointpillar_painted.yaml`  
- 无需运行`python test.py --ckpt /home/xilm/fuxian/PointPainting/detector/output/pointpillar_painted/default/ckpt/checkpoint_epoch_80.pth --batch_size 4 --cfg_file ./cfgs/kitti_models/pointpillar_painted.yaml`,因为已经评估好了。  
- 运行`python demo.py --cfg_file cfgs/kitti_models/pointpillar_painted.yaml --ckpt ../output/pointpillar_painted/default/ckpt/checkpoint_epoch_80_deeplab_pointpillar.pth --data_path /home/xilm/kitti/KITTI/training/painted_lidar/000000.npy --ext .npy`,获得可视化结果。  
3. 接下来进行pvrcnn进行检测。  
- 首先获得`pvrcnn_painted.yaml`文件  
## 0420：  
1. 使用pv_rcnn进行检测  
## 0423:  
1. 使用voxel_rcnn进行检测  
## 0425：  
1. 在deeplabv3下的三种点云检测模型都检测完毕；  
2. 使用tensorboard查看实时loss下降结果的方法：`tensorboard --logdir=/home/xilm/fuxian/PointPainting/detector/output/voxel_rcnn_pointed/default/tensorboard --port=18888`  
3. 接下来使用deeplabv3+进行点云绘制。  
## 0430:  
1. 所有方法都已经运行完成，接下来先把deeplabv3plus_pointpillar对应的test性能测试一下。  
`python test.py --cfg_file ./cfgs/kitti_models/pointpillar_painted.yaml --ckpt /home/xilm/Downloads/checkpoint_epoch_80.pth`  
2. 将所有的权重文件重新下载下来，在kitti数据集上分别验证，看看哪些图的说明结果更好。  
- KITTI数据集上的2,17,22,31  
3. 整理一下六个评估结果：  
- Easy mode  

| 方法 | Car-3D Detection | Car-BEV Detection | Pedestrian-3D Detection | Pedestrian-BEV Detection | Cyclists-3D Detection | Cyclists-BEV Detection |  
|:------:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Pointpillar               | 87.7518 | 92.0399 | 57.3015 | 61.5971 | 81.5670 | 85.2585 |
| Deeplabv3_Pointpillar     | 87.0488 | 92.7873 | 59.7824 | 66.7332 | 80.5899 | 88.1952 | 
| Deeplabv3plus_Pointpillar | 88.8145	| 93.0280	| 58.6507	| 64.9471	| 82.1917	| 87.0623 | 
| pvrcnn                    | 92.1047 | 93.0239 | 62.7110 | 65.9365 | 89.1011 | 93.4584 | 
| Deeplabv3_pvrcnn          | 91.7842 | 92.9594 | 69.9658 | 72.7107 | 89.4468 | 92.0936 | 
| Deeplabv3plus_pvrcnn      | 91.8943	| 94.6249	| 66.3709	| 68.2514	| 93.2572	| 94.5304 |
| Voxelrcnn                 | 92.4890 | 95.4822 | 66.8157 | 69.2214 | 91.5068 | 93.2288 | 
| Deeplabv3_voxelrcnn       | 92.5075 | 95.3499 | 64.2595 | 67.4045 | 92.5878 | 93.0751 |
| Deeplabv3plus_voxelrcnn   | 92.3256 | 95.5708 | 67.5197 | 70.3331 | 90.6253 | 91.9494 |


- Moderate mode  

| 方法 | Car-3D Detection | Car-BEV Detection | Pedestrian-3D Detection | Pedestrian-BEV Detection | Cyclists-3D Detection | Cyclists-BEV Detection |  
|:------:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Pointpillar               | 78.3964 | 88.0556 | 51.4145 | 56.0143 | 62.8074 | 66.2439 | 
| Deeplabv3_Pointpillar     | 77.9516 | 87.8021 | 52.7419 | 59.4873 | 61.6325 | 68.4552 | 
| Deeplabv3plus_Pointpillar | 77.9333	| 87.6655	| 53.0701	| 59.7557	| 63.0031	| 67.3700 | 
| pvrcnn                    | 84.3605 | 90.3255 | 54.4902 | 58.5166 | 70.3809 | 74.5322 |
| Deeplabv3_pvrcnn          | 84.5432	| 90.9980	| 61.8915	| 64.9520	| 71.6946	| 74.8322 | 
| Deeplabv3plus_pvrcnn      | 83.0667	| 90.4686	| 58.1198	| 60.9370	| 74.2496	| 77.0280 |
| Voxelrcnn                 | 85.0112 | 91.0943 | 60.2380 | 63.3761 | 72.5279 | 75.6497 | 
| Deeplabv3_voxelrcnn       | 85.1840 | 91.3552 | 56.7385 | 60.8148 | 72.7842 | 74.0956 | 
| Deeplabv3plus_voxelrcnn   | 85.0536 | 91.0679 | 60.2547 | 63.2770 | 72.9780 | 75.6802 |

- Hard mode  

| 方法 | Car-3D Detection | Car-BEV Detection | Pedestrian-3D Detection | Pedestrian-BEV Detection | Cyclists-3D Detection | Cyclists-BEV Detection |  
|:------:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Pointpillar               | 75.1843 | 86.6625 | 46.8715 | 52.0457 | 58.8314 | 62.2173 | 
| Deeplabv3_Pointpillar     | 75.3190 | 86.7787 | 48.2330 | 55.3915 | 58.0191 | 64.2489 | 
| Deeplabv3plus_Pointpillar | 75.1616	| 86.7752	| 48.6049	| 56.1615	| 58.9886	| 63.1139 | 
| pvrcnn                    | 82.4830 | 88.5319 | 49.8798 | 54.1258 | 66.0168 | 70.1025 |
| Deeplabv3_pvrcnn          | 82.6717	| 88.8078	| 56.8798	| 60.2252	| 67.2554	| 70.2091 | 
| Deeplabv3plus_pvrcnn      | 82.4754	| 88.5908	| 53.3453	| 57.0771	| 69.5003	| 72.3781 |
| Voxelrcnn                 | 82.7410 | 88.9366 | 55.8170 | 58.9166 | 68.0872 | 71.3415 | 
| Deeplabv3_voxelrcnn       | 82.9317 | 89.0419 | 52.1230 | 56.5321 | 68.6926 | 70.8189 | 
| Deeplabv3plus_voxelrcnn   | 82.7638 | 88.8630 | 55.2613 | 59.3268 | 68.7283 | 71.2809 |

4. 重新使用纯点云检测方法进行评估：  
- 首先准备KITTI数据集：`python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml`  
- 下载PointPillar, pvrcnn, voxelrcnn训练权重文件；  
- 测试：`python test.py --cfg_file ./cfgs/kitti_models/pointpillar.yaml --ckpt /home/xilm/Downloads/pointpillar_7728.pth`  
- 绘制pic demo：在使用Pointpainting里的demo.py脚本绘制纯点云的图时，需要注释掉`visualize_utils.py`中第84行`rgba[:, :3] = pts[:, 5:8] * 255`，这一行是用来着色点云的，纯velodyne点云没有着色信息，所以需要去掉。`python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt /home/xilm/Downloads/pointpillar_7728.pth --data_path /home/xilm/kitti/KITTI/training/velodyne/000000.bin`    
- 绘制FOV图失败，只能先全景图。。。  
5. 使用pv_rcnn进行评估。  
## 0502:  
整个性能对比实验已经完成了。  

