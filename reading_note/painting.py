import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from PIL import Image
from tqdm import tqdm

from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

import pcdet_utils.calibration_kitti as calibration_kitti

# 训练图像存放的路径
TRAINING_PATH = "../detector/data/kitti/training/"
# 是否包含image_3，也就是使用左右两个相机的信息；但是显然这个代码有不足，必须使用两个相机的信息，没有为只有一个相机的情况做考虑
TWO_CAMERAS = True
# 选择分割网络
SEG_NET_OPTIONS = ["deeplabv3", "deeplabv3plus", "hma"]
# TODO choose the segmentation network you want to use, deeplabv3 = 0 deeplabv3plus = 1 hma = 2
SEG_NET = 1 #TODO choose your preferred network
# 现在选择了deeplabv3+

class Painter:
    def __init__(self, seg_net_index):
        # 初始化训练图像存放路径
        self.root_split_path = TRAINING_PATH
        # 初始化绘制后点云的存放路径
        self.save_path = TRAINING_PATH + "painted_lidar/"
        # 创建绘制后点云的存放路径
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        # 初始化分割的网络索引
        self.seg_net_index = seg_net_index
        # 初始化模型，当前为空
        self.model = None
        if seg_net_index == 0:
            print(f'Using Segmentation Network -- {SEG_NET_OPTIONS[seg_net_index]}')
            # 加载模型
            self.model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
            # 评估模型
            self.model.eval()
            # 判断是否有cuda存在
            if torch.cuda.is_available():
                # 把模型放到gpu上运行
                self.model.to('cuda')
        elif seg_net_index == 1:
            print(f'Using Segmentation Network -- {SEG_NET_OPTIONS[seg_net_index]}')
            # 读取配置文件
            config_file = './mmseg/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes.py'
            # 读取权重文件
            checkpoint_file = './mmseg/checkpoints/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth'
            # 加载模型
            self.model = init_segmentor(config_file, checkpoint_file, device='cuda:0') # TODO edit here if you want to use different device

        
    def get_lidar(self, idx):
        # 首先表示出索引为idx的点云文件的路径
        lidar_file = self.root_split_path + 'velodyne/' + ('%s.bin' % idx)
        # 从文本或二进制文件中的数据构造一个数组，并把它转换为N*4的形状
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_score(self, idx, left):
        ''' idx : index string
            left : string indicates left/right camera 
        return:
            a tensor H  * W * 4(deeplab)/5(deeplabv3plus), for each pixel we have 4/5 scorer that sums to 1
        '''
        # 先声明这个变量（后面用来计算softmax）
        output_reassign_softmax = None
        # 判断使用的是哪种语义分割网络
        # 如果使用的是deeplabv3
        if self.seg_net_index == 0:
            # 获取当前点云对应的左目相机采集的图像的路径
            filename = self.root_split_path + left + ('%s.png' % idx)
            # 获取该图像
            input_image = Image.open(filename)
            # 对图像的预处理
            # transforms.Compose将要转换的方法组成列表
            preprocess = transforms.Compose([
                # 将PIL图像或ndarray转换为tensor并相应地缩放值
                transforms.ToTensor(),
                # 归一化
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # 对输入图像进行预处理，获取输入的tensor
            input_tensor = preprocess(input_image)
            # 给tensor多加一个维度（在最外面的括号外再加一个括号）
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                # 把输入放到gpu上
                input_batch = input_batch.to('cuda')

            # 不进行梯度下将
            with torch.no_grad():
                # 
                output = self.model(input_batch)['out'][0]

            # 更换output的维度位置（这里把第一个维度踢到最后了）
            output_permute = output.permute(1,2,0)
            # 输出output第二个维度上的最大值（包括预测标签和预测概率）
            output_probability, output_predictions = output_permute.max(2)

            # 表示经过判断该物体不是背景、自行车、汽车和人，而是其他物体
            other_object_mask = ~((output_predictions == 0) | (output_predictions == 2) | (output_predictions == 7) | (output_predictions == 15))
            # 表示经过判断，它是我们要检测的物体（背景、自行车、汽车或人）
            detect_object_mask = ~other_object_mask
            # 定义一个沿着第二个维度计算softmax的变量
            sf = torch.nn.Softmax(dim=2)

            # bicycle = 2 car = 7 person = 15 background = 0
            # 建立一个纯零向量，大小为（output.size[1], output.size[2], 4）
            output_reassign = torch.zeros(output_permute.size(0),output_permute.size(1),4)
            # 输出是背景或其他物体的概率
            output_reassign[:,:,0] = detect_object_mask * output_permute[:,:,0] + other_object_mask * output_probability # background
            # 输出是自行车的概率
            output_reassign[:,:,1] = output_permute[:,:,2] # bicycle
            # 输出是汽车的概率
            output_reassign[:,:,2] = output_permute[:,:,7] # car
            # 输出是人的概率
            output_reassign[:,:,3] = output_permute[:,:,15] # person
            # 计算输出是这四类中哪个（使用softmax，在物体类别上进行）
            output_reassign_softmax = sf(output_reassign).cpu().numpy()

        # 如果使用的是deeplabv3+网络
        elif self.seg_net_index == 1:
            # 获取当前点云对应的图片的路径
            filename = self.root_split_path + left + ('%s.png' % idx)
            # 使用deeplabv3+获取图像分割结果
            result = inference_segmentor(self.model, filename)
            # person 11, rider 12, vehicle 13/14/15/16, bike 17/18
            output_permute = torch.tensor(result[0]).permute(1,2,0) # H, W, 19
            # 定义一个沿着第二个维度计算softmax的变量
            sf = torch.nn.Softmax(dim=2)

            output_reassign = torch.zeros(output_permute.size(0),output_permute.size(1), 5)
            output_reassign[:,:,0], _ = torch.max(output_permute[:,:,:11], dim=2) # background
            output_reassign[:,:,1], _ = torch.max(output_permute[:,:,[17, 18]], dim=2) # bicycle
            output_reassign[:,:,2], _ = torch.max(output_permute[:,:,[13, 14, 15, 16]], dim=2) # car
            output_reassign[:,:,3] = output_permute[:,:,11] #person
            output_reassign[:,:,4] = output_permute[:,:,12] #rider
            output_reassign_softmax = sf(output_reassign).cpu().numpy()
        
        elif self.seg_net_index == 2:
            filename = self.root_split_path + "score_hma/" + left + ('%s.npy' % idx)
            output_reassign_softmax = np.load(filename)

        return output_reassign_softmax

    def get_calib(self, idx):
        calib_file = self.root_split_path + 'calib/' + ('%s.txt' % idx)
        return calibration_kitti.Calibration(calib_file)
    
    def get_calib_fromfile(self, idx):
        # 获得校准文件存在的路径
        calib_file = self.root_split_path + 'calib/' + ('%s.txt' % idx)
        # 获得校准矩阵，返回的是一个字典
        calib = calibration_kitti.get_calib_from_file(calib_file)
        # 在P2矩阵的最后添加一行[0., 0., 0., 1.]
        calib['P2'] = np.concatenate([calib['P2'], np.array([[0., 0., 0., 1.]])], axis=0)
        # 在P3矩阵的最后添加一行[0., 0., 0., 1.]
        calib['P3'] = np.concatenate([calib['P3'], np.array([[0., 0., 0., 1.]])], axis=0)
        # 接下来是定义旋转矩阵，把3*3变成了4*4的矩阵
        calib['R0_rect'] = np.zeros([4, 4], dtype=calib['R0'].dtype)
        calib['R0_rect'][3, 3] = 1.
        calib['R0_rect'][:3, :3] = calib['R0']
        # 在Tr_velo2cam矩阵的最后添加一行[0., 0., 0., 1.]
        calib['Tr_velo2cam'] = np.concatenate([calib['Tr_velo2cam'], np.array([[0., 0., 0., 1.]])], axis=0)
        return calib
    
    def create_cyclist(self, augmented_lidar):
        if self.seg_net_index == 0:
            bike_idx = np.where(augmented_lidar[:,5]>=0.2)[0] # 0, 1(bike), 2, 3(person)
            bike_points = augmented_lidar[bike_idx]
            cyclist_mask_total = np.zeros(augmented_lidar.shape[0], dtype=bool)
            for i in range(bike_idx.shape[0]):
                cyclist_mask = (np.linalg.norm(augmented_lidar[:,:3]-bike_points[i,:3], axis=1) < 1) & (np.argmax(augmented_lidar[:,-4:],axis=1) == 3)
                if np.sum(cyclist_mask) > 0:
                    cyclist_mask_total |= cyclist_mask
                else:
                    augmented_lidar[bike_idx[i], 4], augmented_lidar[bike_idx[i], 5] = augmented_lidar[bike_idx[i], 5], 0
            augmented_lidar[cyclist_mask_total, 7], augmented_lidar[cyclist_mask_total, 5] = 0, augmented_lidar[cyclist_mask_total, 7]
            return augmented_lidar
        elif self.seg_net_index == 1 or 2:
            rider_idx = np.where(augmented_lidar[:,8]>=0.3)[0] # 0, 1(bike), 2, 3(person), 4(rider)
            rider_points = augmented_lidar[rider_idx]
            bike_mask_total = np.zeros(augmented_lidar.shape[0], dtype=bool)
            bike_total = (np.argmax(augmented_lidar[:,-5:],axis=1) == 1)
            for i in range(rider_idx.shape[0]):
                bike_mask = (np.linalg.norm(augmented_lidar[:,:3]-rider_points[i,:3], axis=1) < 1) & bike_total
                bike_mask_total |= bike_mask
            augmented_lidar[bike_mask_total, 8] = augmented_lidar[bike_mask_total, 5]
            augmented_lidar[bike_total^bike_mask_total, 4] = augmented_lidar[bike_total^bike_mask_total, 5]
            return augmented_lidar[:,[0,1,2,3,4,8,6,7]]

    def cam_to_lidar(self, pointcloud, projection_mats):
        """
        Takes in lidar in velo coords, returns lidar points in camera coords

        :param pointcloud: (n_points, 4) np.array (x,y,z,r) in velodyne coordinates
        :return lidar_cam_coords: (n_points, 4) np.array (x,y,z,r) in camera coordinates
        """
        # 输入在velodyne坐标系下的lidar点的坐标，输出在相机坐标系下lidar点的坐标

        # lidar_velo_coords变量表示：lidar点在velodyne坐标系下的坐标，此处深拷贝了点云坐标
        lidar_velo_coords = copy.deepcopy(pointcloud)
        # 获取点云的反射率
        reflectances = copy.deepcopy(lidar_velo_coords[:, -1]) #copy reflectances column
        # 为了使用齐次矩阵相乘，令在velodyne坐标系下点云的坐标的反射率都为1
        lidar_velo_coords[:, -1] = 1 # for multiplying with homogeneous matrix
        # 把在velodyne坐标系下的点云坐标（此时反射率都为1）与转换坐标相乘，获得在相机坐标系下的点云坐标
        lidar_cam_coords = projection_mats['Tr_velo2cam'].dot(lidar_velo_coords.transpose())
        # 把它转置回去
        lidar_cam_coords = lidar_cam_coords.transpose()
        # 把反射率放回去
        lidar_cam_coords[:, -1] = reflectances
        
        return lidar_cam_coords

    def augment_lidar_class_scores_both(self, class_scores_r, class_scores_l, lidar_raw, projection_mats):
        """
        Projects lidar points onto segmentation map, appends class score each point projects onto.
        """
        #lidar_cam_coords = self.cam_to_lidar(lidar_raw, projection_mats)
        # TODO: Project lidar points onto left and right segmentation maps. How to use projection_mats? 
        ################################
        # 获得点云在相机坐标系下的坐标
        lidar_cam_coords = self.cam_to_lidar(lidar_raw, projection_mats)

        # right
        lidar_cam_coords[:, -1] = 1 #homogenous coords for projection
        # TODO: change projection_mats['P2'] and projection_mats['R0_rect'] to be?
        points_projected_on_mask_r = projection_mats['P3'].dot(projection_mats['R0_rect'].dot(lidar_cam_coords.transpose()))
        points_projected_on_mask_r = points_projected_on_mask_r.transpose()
        points_projected_on_mask_r = points_projected_on_mask_r/(points_projected_on_mask_r[:,2].reshape(-1,1))

        true_where_x_on_img_r = (0 < points_projected_on_mask_r[:, 0]) & (points_projected_on_mask_r[:, 0] < class_scores_r.shape[1]) #x in img coords is cols of img
        true_where_y_on_img_r = (0 < points_projected_on_mask_r[:, 1]) & (points_projected_on_mask_r[:, 1] < class_scores_r.shape[0])
        true_where_point_on_img_r = true_where_x_on_img_r & true_where_y_on_img_r

        points_projected_on_mask_r = points_projected_on_mask_r[true_where_point_on_img_r] # filter out points that don't project to image
        points_projected_on_mask_r = np.floor(points_projected_on_mask_r).astype(int) # using floor so you don't end up indexing num_rows+1th row or col
        points_projected_on_mask_r = points_projected_on_mask_r[:, :2] #drops homogenous coord 1 from every point, giving (N_pts, 2) int array

        # left
        lidar_cam_coords[:, -1] = 1 #homogenous coords for projection
        # TODO: change projection_mats['P2'] and projection_mats['R0_rect'] to be?
        points_projected_on_mask_l = projection_mats['P2'].dot(projection_mats['R0_rect'].dot(lidar_cam_coords.transpose()))
        points_projected_on_mask_l = points_projected_on_mask_l.transpose()
        points_projected_on_mask_l = points_projected_on_mask_l/(points_projected_on_mask_l[:,2].reshape(-1,1))

        # ？
        true_where_x_on_img_l = (0 < points_projected_on_mask_l[:, 0]) & (points_projected_on_mask_l[:, 0] < class_scores_l.shape[1]) #x in img coords is cols of img
        true_where_y_on_img_l = (0 < points_projected_on_mask_l[:, 1]) & (points_projected_on_mask_l[:, 1] < class_scores_l.shape[0])
        true_where_point_on_img_l = true_where_x_on_img_l & true_where_y_on_img_l

        points_projected_on_mask_l = points_projected_on_mask_l[true_where_point_on_img_l] # filter out points that don't project to image
        points_projected_on_mask_l = np.floor(points_projected_on_mask_l).astype(int) # using floor so you don't end up indexing num_rows+1th row or col
        points_projected_on_mask_l = points_projected_on_mask_l[:, :2] #drops homogenous coord 1 from every point, giving (N_pts, 2) int array

        true_where_point_on_both_img = true_where_point_on_img_l & true_where_point_on_img_r
        true_where_point_on_img = true_where_point_on_img_l | true_where_point_on_img_r

        #indexing oreder below is 1 then 0 because points_projected_on_mask is x,y in image coords which is cols, rows while class_score shape is (rows, cols)
        #socre dimesion: point_scores.shape[2] TODO!!!!
        point_scores_r = class_scores_r[points_projected_on_mask_r[:, 1], points_projected_on_mask_r[:, 0]].reshape(-1, class_scores_r.shape[2])
        point_scores_l = class_scores_l[points_projected_on_mask_l[:, 1], points_projected_on_mask_l[:, 0]].reshape(-1, class_scores_l.shape[2])
        #augmented_lidar = np.concatenate((lidar_raw[true_where_point_on_img], point_scores), axis=1)
        augmented_lidar = np.concatenate((lidar_raw, np.zeros((lidar_raw.shape[0], class_scores_r.shape[2]))), axis=1)
        augmented_lidar[true_where_point_on_img_r, -class_scores_r.shape[2]:] += point_scores_r
        augmented_lidar[true_where_point_on_img_l, -class_scores_l.shape[2]:] += point_scores_l
        augmented_lidar[true_where_point_on_both_img, -class_scores_r.shape[2]:] = 0.5 * augmented_lidar[true_where_point_on_both_img, -class_scores_r.shape[2]:]
        augmented_lidar = augmented_lidar[true_where_point_on_img]
        augmented_lidar = self.create_cyclist(augmented_lidar)

        return augmented_lidar

    def run(self):
        # KITTI数据集有7481张训练图片
        num_image = 7481
        # 使迭代能够在一个进度条中显示
        for idx in tqdm(range(num_image)):
            # 把点云索引表示为字符串形式
            sample_idx = "%06d" % idx
            # points: N * 4(x, y, z, r)
            points = self.get_lidar(sample_idx)
            
            # get segmentation score from network
            # 使用指定的分割网络获取图像的分割分数
            # 左目相机
            scores_from_cam = self.get_score(sample_idx, "image_2/")
            # 右目相机
            scores_from_cam_r = self.get_score(sample_idx, "image_3/")
            # scores_from_cam: H * W * 4/5, each pixel have 4/5 scores(0: background, 1: bicycle, 2: car, 3: person, 4: rider)

            # get calibration data
            calib_fromfile = self.get_calib_fromfile(sample_idx)
            
            # paint the point clouds
            # points: N * 8
            points = self.augment_lidar_class_scores_both(scores_from_cam_r, scores_from_cam, points, calib_fromfile)
            
            np.save(self.save_path + ("%06d.npy" % idx), points)

if __name__ == '__main__':
    # 实例化一个Painter,传入分割网络的类别
    painter = Painter(SEG_NET)
    # 运行painter
    painter.run()