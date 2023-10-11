from enum import Flag
import numpy as np
from mmdet.datasets import DATASETS
import torch
# from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
import mmcv
import cv2
import os.path as osp
from tqdm import tqdm
import copy
import random


@DATASETS.register_module()
class InternalDatasetSweep(Custom3DDataset):
    r"""Internal Dataset.
    """
    CLASSES = ('VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN')
    
    cams = ['center_camera_fov120', 'left_front_camera', 'left_rear_camera',\
            'rear_camera', 'right_rear_camera', "right_front_camera"]

    def __init__(self,
                 data_root,
                 ann_file=None,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 test_mode=False,
                 box_type_3d='LiDAR',
                 shuffle=False):

        self.shuffle = shuffle
        if self.shuffle:
            print("Building a shuffle dataset")

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            test_mode=test_mode,
            box_type_3d=box_type_3d)

        self.data_root = data_root
    
    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        if self.shuffle:
            random.seed(0)
            random.shuffle(data_infos)
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.
        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )
        center2lidar = np.matrix(info['center2lidar'])
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info['cams'].items():
                img_timestamp.append(cam_info['timestamp'] / 1e6) # 时间戳
                image_paths.append(cam_info['data_path']) # 图片路径
                intrinsic = np.matrix(np.array(cam_info['cam_intrinsic']).reshape(3,3)) # 相机内参
                extrinsic = np.matrix(np.array(cam_info['extrinsic']).reshape(4,4)) # lidar2cam
                extrinsic = extrinsic @ center2lidar # center2cam
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = np.array((viewpad @ extrinsic)) # center2img

                intrinsics.append(viewpad) # 相机内参4x4
                extrinsics.append(extrinsic.T) # center2cam
                lidar2img_rts.append(lidar2img_rt) # center2img

            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        gt_bboxes_3d = np.array(info['gt_boxes'])
        gt_velocity = np.array([[0, 0]] * gt_bboxes_3d.shape[0])
        gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity] , axis=-1)
        gt_names_3d = info['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)
        
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5,0.5,0.5)).convert_to(self.box_mode_3d)
        
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d
        )
        
        return anns_results

    def bev_to_corners(self, bev):
        n = bev.shape[0]
        bev[:, -1] = -bev[:, -1]
        # origin_corners = torch.stack(
        #     (0.5 * bev[:, 0], 0.5 * bev[:, 1], torch.ones(n),
        #     0.5 * bev[:, 0], -0.5 * bev[:, 1], torch.ones(n),
        #     -0.5 * bev[:, 0], -0.5 * bev[:, 1], torch.ones(n),
        #     -0.5 * bev[:, 0], 0.5 * bev[:, 1], torch.ones(n))
        # ).reshape(3, 4, n)
        # origin_corners = origin_corners.permute(1,2,0)
        # Tr = torch.stack(
        #     (torch.cos(bev[:, -1]), -torch.sin(bev[:, -1]), bev[:, 0],
        #     torch.sin(bev[:, -1]), torch.cos(bev[:, -1]), bev[:, 1],
        #     torch.zeros(n), torch.zeros(n), torch.ones(n))
        # )
        # Tr = Tr.reshape(n, 3, 3)
        # Tr.shape
        corners = torch.stack(
            (0.5 * bev[:, 2] * torch.cos(bev[:, -1]) - 0.5 * bev[:, 3] * torch.sin(bev[:, -1]) + bev[:, 0],
            0.5 * bev[:, 2] * torch.sin(bev[:, -1]) + 0.5 * bev[:, 3] * torch.cos(bev[:, -1]) + bev[:, 1],
            0.5 * bev[:, 2] * torch.cos(bev[:, -1]) + 0.5 * bev[:, 3] * torch.sin(bev[:, -1]) + bev[:, 0],
            0.5 * bev[:, 2] * torch.sin(bev[:, -1]) - 0.5 * bev[:, 3] * torch.cos(bev[:, -1]) + bev[:, 1],
            -0.5 * bev[:, 2] * torch.cos(bev[:, -1]) + 0.5 * bev[:, 3] * torch.sin(bev[:, -1]) + bev[:, 0],
            -0.5 * bev[:, 2] * torch.sin(bev[:, -1]) - 0.5 * bev[:, 3] * torch.cos(bev[:, -1]) + bev[:, 1],
            -0.5 * bev[:, 2] * torch.cos(bev[:, -1]) - 0.5 * bev[:, 3] * torch.sin(bev[:, -1]) + bev[:, 0],
            -0.5 * bev[:, 2] * torch.sin(bev[:, -1]) + 0.5 * bev[:, 3] * torch.cos(bev[:, -1]) + bev[:, 1],)
        )
        corners = corners.reshape(4,2,n).permute(2,0,1)
        return corners

    def draw_bev_result(self, img, pred_bev_corners, gt_bev_corners):
        bev_size = 1600
        scale = 10

        if img is None:
            img = np.zeros((bev_size, bev_size, 3))

        # draw circle
        for i in range(bev_size//(20*scale)):
            cv2.circle(img, (bev_size//2,bev_size//2), (i+1)*10*scale, (125,217,233), 2)
            if i == 4:
                cv2.circle(img, (bev_size//2,bev_size//2), (i+1)*10*scale, (255,255,255), 2)
        # # draw circle
        # for i in range(bev_size//200):
        #     cv2.circle(img, (bev_size//2,bev_size//2), (i+1)*100, (255,255,255), 1)

        if gt_bev_corners is not None:
            gt_bev_buffer = copy.deepcopy(gt_bev_corners)
            gt_bev_corners[:,:,0] = -gt_bev_buffer[:,:,1] * scale + bev_size//2
            gt_bev_corners[:,:,1] = -gt_bev_buffer[:,:,0] * scale + bev_size//2

            gt_color = (61, 102, 255)
            for corners in gt_bev_corners:
                cv2.line(img, (int(corners[0,0]), int(corners[0,1])), (int(corners[1,0]), int(corners[1,1])), gt_color, 4)
                cv2.line(img, (int(corners[1,0]), int(corners[1,1])), (int(corners[2,0]), int(corners[2,1])), gt_color, 4)
                cv2.line(img, (int(corners[2,0]), int(corners[2,1])), (int(corners[3,0]), int(corners[3,1])), gt_color, 4)
                cv2.line(img, (int(corners[3,0]), int(corners[3,1])), (int(corners[0,0]), int(corners[0,1])), gt_color, 4)

        if pred_bev_corners is not None:
            pred_bev_buffer = copy.deepcopy(pred_bev_corners)
            pred_bev_corners[:,:,0] = -pred_bev_buffer[:,:,1] * scale + bev_size//2
            pred_bev_corners[:,:,1] = -pred_bev_buffer[:,:,0] * scale + bev_size//2
            pred_color = (241, 101, 72)
            for corners in pred_bev_corners:
                cv2.line(img, (int(corners[0,0]), int(corners[0,1])), (int(corners[1,0]), int(corners[1,1])), pred_color, 3)
                cv2.line(img, (int(corners[1,0]), int(corners[1,1])), (int(corners[2,0]), int(corners[2,1])), pred_color, 3)
                cv2.line(img, (int(corners[2,0]), int(corners[2,1])), (int(corners[3,0]), int(corners[3,1])), pred_color, 3)
                cv2.line(img, (int(corners[3,0]), int(corners[3,1])), (int(corners[0,0]), int(corners[0,1])), pred_color, 3)

        return img

    def draw_bev_lidar(self, img, points):
        bev_size = 1600
        scale = 10
        if img is None:
            img = np.zeros((bev_size, bev_size, 3))
        # draw circle
        for i in range(bev_size//(20*scale)):
            cv2.circle(img, (bev_size//2,bev_size//2), (i+1)*10*scale, (125,217,233), 2)

        idx = (points[:, 0] < bev_size //(2*scale)) &\
              (points[:, 0] > -bev_size //(2*scale)) &\
              (points[:, 1] < bev_size //(2*scale)) &\
              (points[:, 1] > -bev_size //(2*scale))
        points = points[idx]

        points[:, 0] = points[:, 0] * scale + bev_size//2
        points[:, 1] = -points[:, 1] * scale + bev_size//2
        for i in range(0, len(points), 10):
            point = points[i]
            cv2.circle(img, (int(point[0]), int(point[1])), 1, (255,255,255), 1)

        return img

    def show_bev(self, results, out_dir, pipeline=None):
        pipeline = self._get_pipeline(pipeline)
        for i, result in tqdm(list(enumerate(results))):
            data_info = self.data_infos[i]
            file_name = data_info

            pred_bboxes = result['pts_bbox']['boxes_3d']
            scores = result['pts_bbox']['scores_3d']

            idx = scores > 0.2
            pred_bboxes = pred_bboxes[idx]

            pred_bev_corners = self.bev_to_corners(pred_bboxes.bev)
            

            img = None
            # img = self.draw_bev_lidar(img, points)
            img = self.draw_bev_result(img, pred_bev_corners, None)
            # result_path = osp.join(out_dir, file_name)
            # mmcv.mkdir_or_exist(result_path)
            # save_path_list = file_name.split('__')
            save_path = osp.join(out_dir, file_name + '.png')
            if img is not None:
                mmcv.imwrite(img, save_path)

    def show(self, results, out_dir, show=False, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in tqdm(list(enumerate(results))):
            data_info = self.data_infos[i]
            result_path = osp.join(out_dir, data_info[:-4])
            pred_bboxes = result['pts_bbox']['boxes_3d']
            scores = result['pts_bbox']['scores_3d']
            idx = scores > 0.2
            pred_bboxes = pred_bboxes[idx]

            mmcv.mkdir_or_exist(result_path)
            for num, cam in enumerate(self.cams):
                img_path = osp.join(self.data_root, "calib_images", cam, data_info)
                img = mmcv.imread(img_path)
                file_name = osp.split(img_path)[-1].split('.')[0]
            
                # need to transpose channel to first dim
                # anno_info = self.get_ann_info(i)
                # gt_bboxes = anno_info['gt_bboxes_3d']
                pred_bbox_color=(241, 101, 72)
                img = draw_lidar_bbox3d_on_img(
                    copy.deepcopy(pred_bboxes), img, self.lidar2img_rts[num], None, color=pred_bbox_color,thickness=3)
                mmcv.imwrite(img, osp.join(result_path, f'{cam}_pred.png'))
    
    def plot_rect3d_on_img(self, 
                       img,
                       num_rects,
                       rect_corners,
                       color=(0, 255, 0),
                       thickness=1):
        """Plot the boundary lines of 3D rectangular on 2D images.

        Args:
            img (numpy.array): The numpy array of image.
            num_rects (int): Number of 3D rectangulars.
            rect_corners (numpy.array): Coordinates of the corners of 3D
                rectangulars. Should be in the shape of [num_rect, 8, 2].
            color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
            thickness (int, optional): The thickness of bboxes. Default: 1.
        """
        line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                        (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
        for i in range(num_rects):
            corners = rect_corners[i].astype(np.int)
            for start, end in line_indices:
                cv2.line(img, (corners[start, 0], corners[start, 1]),
                        (corners[end, 0], corners[end, 1]), color, thickness,
                        cv2.LINE_AA)

        return img.astype(np.uint8)

    def draw_lidar_bbox3d_on_img(self,
                             bboxes3d,
                             raw_img,
                             lidar2img_rt,
                             img_metas,
                             color=(0, 255, 0),
                             thickness=1):
        """Project the 3D bbox on 2D plane and draw on input image.

        Args:
            bboxes3d (:obj:`LiDARInstance3DBoxes`):
                3d bbox in lidar coordinate system to visualize.
            raw_img (numpy.array): The numpy array of image.
            lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
                according to the camera intrinsic parameters.
            img_metas (dict): Useless here.
            color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
            thickness (int, optional): The thickness of bboxes. Default: 1.
        """
        img = raw_img.copy()
        corners_3d = bboxes3d.corners
        num_bbox = corners_3d.shape[0]
        pts_4d = np.concatenate(
            [corners_3d.reshape(-1, 3),
            np.ones((num_bbox * 8, 1))], axis=-1)
        lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
        if isinstance(lidar2img_rt, torch.Tensor):
            lidar2img_rt = lidar2img_rt.cpu().numpy()
        pts_2d = pts_4d @ lidar2img_rt.T

        pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        imgfov_pts_3d = pts_2d[..., :3].reshape(num_bbox, 8, 3)

        obj2d_list = list()
        for i, obj in enumerate(imgfov_pts_3d):
            in_front = obj[:, 2] > 0.1
            if all(in_front) is False:
                continue
            obj2d_list.append(obj[:, :2])
       
        num_bbox = len(obj2d_list)

        return self.plot_rect3d_on_img(img, num_bbox, obj2d_list, color, thickness)

    def show_panorama(self, results, out_dir, pipeline=None, show_pred=True, sample_rate=1, video_writer=None):
        pipeline = self._get_pipeline(pipeline)
        file_client_args = dict(
            backend='petrel',
            path_mapping=dict({
                'detr3d/': 'sh1984:s3://sh1984_datasets/detr3d/',
                'cla-datasets/': 'sh1984:s3://sh1984_datasets/cla-datasets/'
            }))
        file_client = mmcv.FileClient(**file_client_args)
        for i, result in tqdm(list(enumerate(results))):
            if i % sample_rate != 0:
                continue
            data_info = self.data_infos[i]
            anno_info = self.get_ann_info(i)
            file_name = str(data_info['timestamp'])
            center2lidar = np.matrix(data_info['center2lidar'])
            gt_bboxes = anno_info['gt_bboxes_3d']
            pred_bboxes = result['pts_bbox']['boxes_3d']

            gt_bboxes.tensor[:, -1] = -gt_bboxes.tensor[:, -1]
            pred_bboxes.tensor[:, 6] = -pred_bboxes.tensor[:, 6]
            
            scores = result['pts_bbox']['scores_3d']
            idx = scores > 0.4
            pred_bboxes = pred_bboxes[idx]

            gt_bev_corners = self.bev_to_corners(gt_bboxes.bev)
            pred_bev_corners = self.bev_to_corners(pred_bboxes.bev)


            bev_img = None
            bev_img = self.draw_bev_result(bev_img, pred_bev_corners, gt_bev_corners)
            cam_imgs = []

            for num, cam in enumerate(self.cams):
                img_path = data_info['cams'][cam]['data_path']
                img_bytes = file_client.get(img_path)
                img = mmcv.imfrombytes(img_bytes)
                cam_info = data_info['cams'][cam]
                intrinsic = np.matrix(cam_info['cam_intrinsic'])
                extrinsic = np.matrix(cam_info['extrinsic'])
                extrinsic = extrinsic @ center2lidar
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = np.array((viewpad @ extrinsic))
                gt_bbox_color=(61, 102, 255) # red
                pred_bbox_color=(241, 101, 72) # blue
                if len(gt_bboxes) != 0:
                    img = self.draw_lidar_bbox3d_on_img(
                        copy.deepcopy(gt_bboxes), img, lidar2img_rt, None, color=gt_bbox_color,thickness=3)
                if len(pred_bboxes) != 0 and show_pred:
                    img = self.draw_lidar_bbox3d_on_img(
                        copy.deepcopy(pred_bboxes), img, lidar2img_rt, None, color=pred_bbox_color,thickness=3)
                cam_imgs.append(img)
            
            img_size = (1600,2400,3)
            pano = np.zeros(img_size, np.uint8)
            bev_img = cv2.resize(bev_img, (800,800))
            pano[400:1200,800:1600]=bev_img

            cam1 = cam_imgs[0]
            cam1 = cv2.resize(cam1, (800,400))
            pano[:400,800:1600]=cam1

            cam2 = cam_imgs[1]
            cam2 = cv2.resize(cam2, (800,400))
            pano[400:800,:800]=cam2

            cam3 = cam_imgs[2]
            cam3 = cv2.resize(cam3, (800,400))
            pano[800:1200,:800]=cam3

            cam4 = cam_imgs[3]
            cam4 = cv2.resize(cam4, (800,400))
            pano[-400:,800:1600]=cam4

            cam5 = cam_imgs[4]
            cam5 = cv2.resize(cam5, (800,400))
            pano[800:1200,-800:]=cam5

            cam6 = cam_imgs[5]
            cam6 = cv2.resize(cam6, (800,400))
            pano[400:800,-800:]=cam6
            save_path = osp.join(out_dir, file_name + '.png')
            if img is not None:
                if video_writer is not None:
                    image = cv2.resize(pano, (1200, 800),
                                    interpolation=cv2.INTER_NEAREST)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    video_writer.writeFrame(image)
                else:
                    mmcv.imwrite(pano, save_path)
        return video_writer


