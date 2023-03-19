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
