import argparse  # 导入Python内置的一个用于命令项选项与参数解析的模块，通过在程序中定义好我们需要的参数，argparse将会从sys.argv中解析出这些参数，并自动生成帮助和使用信息
import glob  # 导入一个文件搜索库，输入路径即可得到该文件
from pathlib import Path  # 导入一个python输入路径的库 

import mayavi.mlab as mlab  # 导入mayavi 3D绘图库
import numpy as np  # 导入python数据处理库
import torch  # 导入pytorch

from pcdet.config import cfg, cfg_from_yaml_file  # 从openpcdet框架中导入cfg(cfg = EasyDict()),以及一个从yaml文件里读取配置信息的函数cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate  # 导入openpcdet中数据集定义模块
from pcdet.models import build_network, load_data_to_gpu  # 导入网络搭建函数（使用配置文件，类别数以及数据集就可以搭建网络），以及将数据加载到gpu上的函数
from pcdet.utils import common_utils  # 导入一些辅助函数，里面大多是关于如何处理点云的
from visual_utils import visualize_utils as V  # 导入一些可视化3D框的辅助函数


# 首先定义数据集的类，继承自openpcdet对数据集的要求
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
