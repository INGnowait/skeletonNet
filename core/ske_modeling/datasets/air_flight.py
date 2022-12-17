import hashlib
import json
import logging
import os, random
import os.path as osp
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
import scipy.io as scio
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)
import time, copy
from collections import OrderedDict
from numpy import *
import mmcv
import numpy as np
from tqdm import tqdm
from mmcv import Config
from transforms3d.quaternions import mat2quat, quat2mat
import ref
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from lib.pysixd import inout, misc
from lib.utils.mask_utils import binary_mask_to_rle, cocosegm2mask
from lib.utils.utils import dprint, iprint, lazy_property


logger = logging.getLogger(__name__)
DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets")) #"/media/j/data/dataset/LINEMOD/" #osp.normpath(osp.join(PROJ_ROOT, "datasets"))


class AIR_Dataset(object):
    """air_flight splits."""

    def __init__(self, data_cfg):
        """
        Set with_depth and with_masks default to True,
        and decide whether to load them into dataloader/network later
        with_masks:
        """
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg

        self.objs = data_cfg["objs"]  # selected objects

        self.ann_files = data_cfg["ann_files"]  # idx files with image ids
        self.image_prefixes = data_cfg["image_prefixes"]
        self.skeleton_prefixes = data_cfg["skeleton_prefixes"]
        # self.xyz_prefixes = data_cfg["xyz_prefixes"]

        self.dataset_root =data_cfg["dataset_root"]  # BOP_DATASETS/air_flight/
        assert osp.exists(self.dataset_root), self.dataset_root
        self.models_root = data_cfg["models_root"]  # BOP_DATASETS/air_flight/models
        self.scale_to_meter = data_cfg["scale_to_meter"]  # 0.001

        self.with_masks = data_cfg["with_masks"]  # True (load masks but may not use it)
        self.with_skeleton = data_cfg["with_skeleton"]  # True (load depth path here, but may not use it)

        self.height = data_cfg["height"]  # 480
        self.width = data_cfg["width"]  # 640

        self.cache_dir = data_cfg.get("cache_dir", osp.join(PROJ_ROOT, ".cache"))  # .cache
        self.use_cache = data_cfg.get("use_cache", True)
        self.num_to_load = data_cfg["num_to_load"]  # -1
        self.filter_invalid = data_cfg["filter_invalid"]
        self.filter_scene = data_cfg.get("filter_scene", False)
        self.debug_im_id = data_cfg.get("debug_im_id", None)
        ##################################################

        # NOTE: careful! Only the selected objects
        self.cat_ids = [cat_id for cat_id, obj_name in ref.air_flight_full.id2obj.items() if obj_name in self.objs]
        # map selected objs to [0, num_objs-1]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, obj_id) for obj_id, obj in enumerate(self.objs))
        ##########################################################

    def __call__(self):  # AIR_Dataset
        """Load light-weight instance annotations of all images into a list of
        dicts in Detectron2 format.

        Do not load heavy data into memory in this file, since we will
        load the annotations of all images into memory.
        """
        # cache the dataset_dicts to avoid loading masks from files
        hashed_file_name = hashlib.md5(
            (
                "".join([str(fn) for fn in self.objs])
                + "dataset_dicts_{}_{}_{}_{}_{}".format(
                    self.name, self.dataset_root, self.with_masks, self.with_skeleton, __name__ # self.name, self.dataset_root, self.with_masks, self.with_depth, self.n_per_obj, __name__
                )
            ).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(self.cache_dir, "dataset_dicts_{}_{}.pkl".format(self.name, hashed_file_name))

        if osp.exists(cache_path) and self.use_cache:
            logger.info("load cached dataset dicts from {}".format(cache_path))
            return mmcv.load(cache_path)

        t_start = time.perf_counter()

        cfg = Config.fromfile(PROJ_ROOT+'/configs/sken/air_flight/sken_base.py')
        logger.info("loading dataset dicts: {}".format(self.name))
        norm_scale = cfg.NORM_SCALE
        self.num_instances_without_valid_segmentation = 0
        self.num_instances_without_valid_box = 0
        dataset_dicts = []  # ######################################################
        assert len(self.ann_files) == len(self.image_prefixes), f"{len(self.ann_files)} != {len(self.image_prefixes)}"
        # assert len(self.ann_files) == len(self.xyz_prefixes), f"{len(self.ann_files)} != {len(self.xyz_prefixes)}"
        # for ann_file, scene_root, xyz_root in zip(tqdm(self.ann_files), self.image_prefixes, self.xyz_prefixes):
        for ann_file, scene_root, skeleton_root in zip(tqdm(self.ann_files), self.image_prefixes, self.skeleton_prefixes):
            # linemod each scene is an object
            with open(ann_file, "r") as f_ann:
                indices = [line.strip("\r\n") for line in f_ann.readlines()]  # string ids
            cam_dict = mmcv.load(osp.join(scene_root, "GT", "scene_camera.json"))
            for im_id in tqdm(indices):
                gt_dict = mmcv.load(osp.join(scene_root, "GT", im_id.split(" ")[0].split("/")[-2], "scene_gt.json"))
                gt_info_dict = mmcv.load(osp.join(scene_root, "GT", im_id.split(" ")[0].split("/")[-2], "scene_gt_info.json"))  # bbox_obj, bbox_visib

                obj_name = im_id.split(" ")[0].split("/")[0]
                objet_id = int(ref.air_flight_full.obj2id[obj_name])
                int_im_id = int(im_id.split(" ")[0].split("_")[-1])   # 图片的数字id
                image_name = im_id.split(" ")[0].split("/")[-1]          # 图片的全称
                int_scene_id = int(im_id.split(" ")[0].split("/")[-2]) # 第几个视角，即第几个摄像机，一共有11个摄像机视角
                str_scene_id = str(int_scene_id)
                str_im_id = str(int_im_id)
                rgb_path = osp.join(osp.join(scene_root, "images/{:04d}/{}.jpg").format(int_scene_id, image_name))
                assert osp.exists(rgb_path), rgb_path

                # scene_id = int(rgb_path.split("/")[-3])
                obj_scene_im_id = f"{objet_id}/{int_scene_id}/{int_im_id}" # 目标id+场景id+图片id

                if self.debug_im_id is not None:
                    if self.debug_im_id != obj_scene_im_id:
                        continue

                K = np.array(cam_dict["cam_K"], dtype=np.float32).reshape(3, 3)
                depth_factor = 100000.0 / cam_dict["depth_scale"]
                # if self.filter_scene:
                #     if scene_id not in self.cat_ids:
                #         continue
                record = {
                    "dataset_name": self.name,
                    "file_name": osp.relpath(rgb_path, PROJ_ROOT),
                    "height": self.height,
                    "width": self.width,
                    "image_id": int_im_id,
                    "image_name": image_name,
                    "obj_scene_im_id": obj_scene_im_id,  # for evaluation
                    "cam": K,
                    "depth_factor": depth_factor,
                    "img_type": "air",
                }
                insts = []
                gt_dict_list=[]
                gt_dict_list.append(gt_dict[int(im_id.split(" ")[-1])-1]) #将字典转化为列表，这样enumerate才能使用
                for anno_i, anno in enumerate(gt_dict_list):

                    if gt_info_dict[int(im_id.split(" ")[-1])-1]["keypoints_num"]<10: # TODO:将少于10个关键点的点删除
                        continue
                    obj_id = int(anno["obj_id"])
                    if obj_id not in self.cat_ids:
                        continue
                    cur_label = self.cat2label[obj_id]  # 0-based label
                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = np.array(anno["cam_t_m2c"], dtype="float32") / 100.0 / norm_scale  # 计算的是3dbbox的中心点，而不是UE4中机体坐标系的中心点,UE4中的单位为cm，此处除以100为转化为米
                    pose = np.hstack([R, t.reshape(3, 1)])
                    quat = mat2quat(R).astype("float32")

                    proj = (record["cam"] @ t.T).T
                    proj = proj[:2] / proj[2]

                    air_keypoints_2d = np.array(gt_info_dict[int(im_id.split(" ")[-1])-1]["keypoints_2d"])
                    air_keypoints_3d_c = np.array(gt_info_dict[int(im_id.split(" ")[-1]) - 1]["keypoints_3d_c"]) / 100.0 / norm_scale
                    air_keypoints_3d_2d = np.array(gt_info_dict[int(im_id.split(" ")[-1]) - 1]["keypoints_3d_2d"])
                    model = np.array(gt_info_dict[int(im_id.split(" ")[-1]) - 1]["model"])
                    aircraft_size = gt_info_dict[int(im_id.split(" ")[-1]) - 1]["aircraft_size"]
                    img_H = gt_info_dict[int(im_id.split(" ")[-1]) - 1]["img_H"]
                    img_W = gt_info_dict[int(im_id.split(" ")[-1]) - 1]["img_W"]
                    cam_K = np.array(gt_info_dict[int(im_id.split(" ")[-1]) - 1]["cam_K"])
                    eul = gt_info_dict[int(im_id.split(" ")[-1]) - 1]["eul"]
                    position = gt_info_dict[int(im_id.split(" ")[-1]) - 1]["position"]
                    bbox_visib = gt_info_dict[int(im_id.split(" ")[-1])-1]["bbox_visib"]
                    bbox_obj = gt_info_dict[int(im_id.split(" ")[-1])-1]["bbox_obj"]
                    bbox_detect = gt_info_dict[int(im_id.split(" ")[-1]) - 1]["bbox_detect"]
                    x1, y1, w, h = bbox_visib
                    if self.filter_invalid:
                        if h <= 1 or w <= 1:
                            self.num_instances_without_valid_box += 1
                            continue

                    mask_file = osp.join(scene_root, "mask/{:04d}/{}.png".format(int_scene_id, image_name))
                    mask_visib_file = osp.join(scene_root, "mask/{:04d}/{}.png".format(int_scene_id, image_name))
                    assert osp.exists(mask_file), mask_file
                    assert osp.exists(mask_visib_file), mask_visib_file

                    skeleton_file = osp.join(scene_root, "skeleton/{:04d}/{}.png".format(int_scene_id, image_name))
                    assert osp.exists(skeleton_file), skeleton_file

                    # load mask visib  TODO: load both mask_visib and mask_full
                    mask_single = mmcv.imread(mask_visib_file, "unchanged")
                    skeleton_single = mmcv.imread(skeleton_file, "unchanged")
                    area = mask_single.sum()
                    if area < 3:  # filter out too small or nearly invisible instances
                        self.num_instances_without_valid_segmentation += 1
                        continue
                    mask_rle = binary_mask_to_rle(mask_single, compressed=True)
                    skeleton_rle = binary_mask_to_rle(skeleton_single, compressed=True)

                    skeleton_3D = np.array([[0,0,0]])
                    skl_3d_bbox_size = int(cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES)  #64*64*64
                    skl_3d_key_num=cfg.SKELETON_3D_KEY_NUM  # 用多少个点生成骨架
                    key=model.reshape(3, -1).T[:10,:]
                    key_lines = np.array([0,1,1,5,5,9,2,3,2,4,8,6,8,7])
                    for key_i in range(0,key_lines.shape[0],2):
                        skeleton_3D = lines_ca(key[int(key_lines[key_i+1]),:], key[int(key_lines[key_i]),:],skeleton_3D)
                    skeleton_3D_c = np.dot(skeleton_3D, np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3).T) \
                                    + np.array(anno["cam_t_m2c"], dtype="float32")
                    skeleton_3D_2d = project3d_2d(skeleton_3D,cam_K.reshape(3, -1)[:,:3], np.insert(np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3), 3, values=np.array(anno["cam_t_m2c"], dtype="float32"), axis=1))
                    # 将1920*1000的图片映射到skl_3d_bbox_size*skl_3d_bbox_size的图片上
                    # skeleton_3D_2d_on_outsize =skeleton_3D_2d/np.array([img_W,img_H])*np.array([skl_3d_bbox_size,skl_3d_bbox_size])
                    #点的顺序是0,1,2,3，...,取机头2和机身2，因为这两个点的连线与机身平行
                    key_1=air_keypoints_3d_c.reshape(3, -1).T[1,:]*100*norm_scale
                    key_5=air_keypoints_3d_c.reshape(3, -1).T[5,:]*100*norm_scale
                    key_1_proj = air_keypoints_3d_2d.reshape(2, -1).T[1, :]
                    key_5_proj = air_keypoints_3d_2d.reshape(2, -1).T[5, :]
                    long1_1_5_proj = pow((key_1[0]-key_5[0])**2+(key_1[1]-key_5[1])**2,0.5)
                    long2_1_5_proj = pow((key_1_proj[0]-key_5_proj[0])**2+(key_1_proj[1]-key_5_proj[1])**2,0.5)
                    line_scale=long1_1_5_proj/long2_1_5_proj
                    skeleton_3D_small=skeleton_3D/line_scale
                    skeleton_3D_small_R=np.dot(skeleton_3D_small, np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3).T)
                    skl_xmax = skeleton_3D_small_R[:, 0].max()
                    skl_ymax = skeleton_3D_small_R[:, 1].max()
                    skl_zmax = skeleton_3D_small_R[:, 2].max()
                    skl_xmin = skeleton_3D_small_R[:, 0].min()
                    skl_ymin = skeleton_3D_small_R[:, 1].min()
                    skl_zmin = skeleton_3D_small_R[:, 2].min()
                    size_bb=max(max(skl_xmax-skl_xmin,skl_ymax-skl_ymin),skl_zmax-skl_zmin) + 2  # +2为留点余量
                    # 将旋转过后的3D骨架放在一个skl_3d_bbox_size*skl_3d_bbox_size*skl_3d_bbox_size（64*64*64）的正方形内，机体中的(0,0,0)点在正方形中心,并选100个用于生成骨架
                    # 将z的取值范围改为[0,64]
                    skeleton_3D_small_R_out_size=skeleton_3D_small_R/np.array([1,1,max(skl_zmax,abs(skl_zmin))+2])*np.array([1,1,skl_3d_bbox_size/2])\
                                                 + np.array([[0,0,(skl_3d_bbox_size)/2]])
                    all_key_num = skeleton_3D_small_R_out_size.shape[0] # 初始化的时候用了第一个，所以要少一个
                    skl_3D_select= skeleton_3D_small_R_out_size[0,:].reshape(1, 3)
                    skl_3D_c_select = skeleton_3D_c[0, :].reshape(1, 3)
                    skl_3D_proj_select = skeleton_3D_2d[0,:].reshape(1, 2)  # 机体坐标系中的(0,0,0)点
                    # trans_centor = skeleton_3D_2d[0,:].reshape(1, 2)-skeleton_3D_small_R_out_size[0,:2].reshape(1, 2)
                    for key_id in linspace(1, all_key_num-1, skl_3d_key_num):
                        skl_3D_select=np.append(skl_3D_select,skeleton_3D_small_R_out_size[round(key_id),:].reshape(1,3),axis=0)
                        skl_3D_c_select = np.append(skl_3D_c_select, skeleton_3D_c[round(key_id), :].reshape(1, 3), axis=0)
                        skl_3D_proj_select = np.append(skl_3D_proj_select, skeleton_3D_2d[round(key_id), :].reshape(1, 2),axis=0)
                    skl_3D_select[:,:2] =  skl_3D_proj_select  #  将3Dskl的xy与2D projection中的xy对齐

                    if False:
                        skl_3d_c_2d_position = np.zeros((int(img_H), int(img_W), skl_3d_bbox_size * 3), dtype=np.float32)
                        skl_3d_c_2d_probility = np.zeros((int(img_H), int(img_W), skl_3d_bbox_size), dtype=np.float32)
                        key_radius = cfg.MODEL.CDPN.ROT_HEAD.KEY_RADIUS
                        for channel in range(skl_3d_key_num):
                            kp_gt_h = round(skl_3D_select.T[1][channel] - 1)  # 图像坐标系中的y坐标，转到矩阵中为行索引
                            kp_gt_w = round(skl_3D_select.T[0][channel] - 1)  # 图像坐标系中的x坐标，转到矩阵中为列索引
                            kp_gt_d = round(skl_3D_select.T[2][channel] - 1)  # 图像坐标系中的z坐标，转到矩阵中为深度索引

                            kp_gt_3d_x = (skl_3D_c_select.T/100/norm_scale)[0][channel]  # 相机坐标系中的x坐标
                            kp_gt_3d_y = (skl_3D_c_select.T/100/norm_scale)[1][channel]  # 相机坐标系中的y坐标
                            kp_gt_3d_z = (skl_3D_c_select.T/100/norm_scale)[2][channel]  # 相机坐标系中的z坐标

                            key_h = np.arange(kp_gt_h - 4 * key_radius, kp_gt_h + 4 * key_radius + 1)
                            key_w = np.arange(kp_gt_w - 4 * key_radius, kp_gt_w + 4 * key_radius + 1)
                            key_h = np.array([int(h_id) for h_id in key_h if h_id > 0 and h_id < int(img_H)])
                            key_w = np.array([int(w_id) for w_id in key_w if w_id > 0 and w_id < int(img_W)])
                            key_hh, key_ww = np.meshgrid(key_h, key_w)
                            key_hh = key_hh.reshape(1, -1)
                            key_ww = key_ww.reshape(1, -1)
                            key_range = (np.exp((-1) * ((key_hh - kp_gt_h) ** 2 + (key_ww - kp_gt_w) ** 2) / (key_radius ** 2))).reshape((len(key_h), len(key_w)))
                            key_range[key_range < np.exp((-1) * (16))] = 0.0
                            skl_3d_c_2d_probility[min(key_h):max(key_h) + 1, min(key_w):max(key_w) + 1, kp_gt_d] = key_range
                            if cfg.MODEL.CDPN.ROT_HEAD.KP_PRESENT == 0:  # TODO: xxxxxyyyyyzzzzz
                                map = deepcopy(key_range)
                                # map = (key_range)
                                map[map > 0] = 1.0
                                skl_3d_c_2d_position[min(key_h):max(key_h) + 1, min(key_w):max(key_w) + 1, kp_gt_d] = map * kp_gt_3d_x
                                skl_3d_c_2d_position[min(key_h):max(key_h) + 1, min(key_w):max(key_w) + 1, kp_gt_d + skl_3d_bbox_size] = map * kp_gt_3d_y
                                skl_3d_c_2d_position[min(key_h):max(key_h) + 1, min(key_w):max(key_w) + 1,  kp_gt_d + 2 * skl_3d_bbox_size] = map * kp_gt_3d_z

                        skl_3d_c_2d_probility=skl_3d_c_2d_probility[int(max(0,bbox_visib[1]-4*key_radius)): int(min(int(img_H), bbox_visib[1]+bbox_visib[3]+4*key_radius)),
                                              int(max(0, bbox_visib[0] - 4 * key_radius)): int(min(int(img_W), bbox_visib[0] + bbox_visib[ 2] + 4 * key_radius)), :]
                        skl_3d_c_2d_position = skl_3d_c_2d_position[int(max(0, bbox_visib[1] - 4 * key_radius)): int(min(int(img_H), bbox_visib[1] + bbox_visib[3] + 4 * key_radius)),
                                                int(max(0, bbox_visib[0] - 4 * key_radius)): int( min(int(img_W), bbox_visib[0] + bbox_visib[2] + 4 * key_radius)), :]
                        # json.dump(skl3D, '/media/j/data/GDR/datasets/BOP_DATASETS/air_flight/3Dskeleton/'+rgb_path.split('/')[-1].split('.')[0]+'.pkl', protocol=4)
                        # json.dump(json.dumps(skl3D), '/media/j/data/GDR/datasets/BOP_DATASETS/air_flight/3Dskeleton/'+rgb_path.split('/')[-1].split('.')[0]+'.json')
                        # mmcv.mkdir_or_exist('/media/j/data/GDR/datasets/BOP_DATASETS/air_flight/3Dskeleton/'+rgb_path.split('/')[-1].split('.')[0]+'.json')
                        # with open('/media/j/data/GDR/datasets/BOP_DATASETS/air_flight/3Dskeleton/A320-sichuan.json','w') as f:
                        #     f.write(json.dumps(skl3D))
                        # skl_3d_c_2d_position = None

                    else:
                        skl_3d_c_2d_position = None
                        skl_3d_c_2d_probility = None



                    bbox3d = get_model_corners(model.reshape(3, -1).T)
                    if False:  # 看看生成的skeleton_3D对不对
                        # fig = plt.figure(figsize=(12, 7))
                        # plt.xlim(key[:,0].min(), key[:,0].max())
                        # plt.ylim(key[:,1].min(), key[:,1].max())
                        ax1 = plt.axes(projection='3d')
                        ax1.scatter(skl_3D_select[:, 0], skl_3D_select[:, 1], skl_3D_select[:, 2])
                        plt.show()


                    inst = {
                        "category_id": cur_label,  # 0-based label
                        "bbox": bbox_visib,  # TODO: load both bbox_obj and bbox_visib
                        "bbox_detect": bbox_detect,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "bbox_3d":bbox3d,  # 大的3dbbox， 单位cm
                        "pose": pose,
                        "quat": quat,
                        "trans": t,
                        "centroid_2d": proj,  # model的坐标原点在图像坐标系中的位置，是目标3d bbox中坐标的原点，不是质心，是对称中心，T也是计算的原点的偏移量。absolute (cx, cy)
                        "segmentation": mask_rle,
                        "mask_full_file": mask_file,  # TODO: load as mask_full, rle
                        "skeleton": skeleton_rle,
                        # "skeleton_3d":skeleton_3D.T,  #  单位cm，3D骨架每个点在相机坐标系中的位置
                        "skeleton_3d_c":skl_3D_c_select.T/100/norm_scale,  #
                        "skeleton_3d_2d":skl_3D_proj_select.T,  # 3D骨架在图像中的投影
                        "skeleton_3D_small_R":skl_3D_select.T,  # 3d骨架缩小后，在图像中的垂直投影和图像中一样大，并且是经过R之后的骨架，但是没有平移
                        "skl_3d_c_2d_position":skl_3d_c_2d_position,
                        "skl_3d_c_2d_probility":skl_3d_c_2d_probility,
                        "air_keypoints_2d": air_keypoints_2d.reshape(2, -1),
                        "air_keypoints_3d_c": air_keypoints_3d_c.reshape(3, -1), # 这里的尺度单位还是cm
                        "air_keypoints_3d_2d": air_keypoints_3d_2d.reshape(2, -1),
                        "model": model.reshape(3, -1).T,  # 这里的尺度单位还是cm 前十个点为常规店，11th为前10个点的中心点，12th为3dbbox的中心，13th and 14th用于生成飞机3dbbox的
                        "img_H": img_H,
                        "img_W": img_W,
                        "cam_K": cam_K.reshape(3, -1),
                        "eul": eul,
                        "position": position,
                        "aircraft_size": aircraft_size
                    }
                    # if "test" not in self.name:
                    #     xyz_path = osp.join(xyz_root, f"{int_im_id:06d}_{anno_i:06d}.pkl")
                    #     assert osp.exists(xyz_path), xyz_path
                    #     inst["xyz_path"] = xyz_path

                    # model_info = self.models_info[str(obj_id)]
                    # inst["model_info"] = model_info
                    # # TODO: using full mask and full xyz
                    # for key in ["bbox3d_and_center"]:
                    #     inst[key] = self.models[cur_label][key]
                    insts.append(inst)
                if len(insts) == 0:  # filter im without anno
                    continue
                record["annotations"] = insts
                dataset_dicts.append(record)

        if self.num_instances_without_valid_segmentation > 0:
            logger.warning(
                "Filtered out {} instances without valid segmentation. "
                "There might be issues in your dataset generation process.".format(
                    self.num_instances_without_valid_segmentation
                )
            )
        if self.num_instances_without_valid_box > 0:
            logger.warning(
                "Filtered out {} instances without valid box. "
                "There might be issues in your dataset generation process.".format(self.num_instances_without_valid_box)
            )
        ##########################################################################
        if self.num_to_load > 0:
            self.num_to_load = min(int(self.num_to_load), len(dataset_dicts))
            dataset_dicts = dataset_dicts[: self.num_to_load]
        logger.info("loaded {} dataset dicts, using {}s".format(len(dataset_dicts), time.perf_counter() - t_start))

        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(dataset_dicts, cache_path, protocol=4)
        logger.info("Dumped dataset_dicts to {}".format(cache_path))
        return dataset_dicts

    @lazy_property
    def models_info(self):
        models_info_path = osp.join(self.models_root, "models_info.json")
        assert osp.exists(models_info_path), models_info_path
        models_info = mmcv.load(models_info_path)  # key is str(obj_id)
        return models_info

    @lazy_property
    def models(self):
        """Load models into a list."""
        cache_path = osp.join(self.cache_dir, "models_{}.pkl".format("_".join(self.objs)))
        if osp.exists(cache_path) and self.use_cache:
            # dprint("{}: load cached object models from {}".format(self.name, cache_path))
            return mmcv.load(cache_path)

        models = []
        for obj_name in self.objs:
            model = inout.load_ply(
                osp.join(self.models_root, f"obj_{ref.air_flight_full.obj2id[obj_name]:06d}.ply"),
                vertex_scale=self.scale_to_meter,
            )
            # NOTE: the bbox3d_and_center is not obtained from centered vertices
            # for BOP models, not a big problem since they had been centered
            model["bbox3d_and_center"] = misc.get_bbox3d_and_center(model["pts"])

            models.append(model)
        logger.info("cache models to {}".format(cache_path))
        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(models, cache_path, protocol=4)
        return models

    def image_aspect_ratio(self):
        return self.width / self.height  # 4/3


########### register datasets ############################################################


def get_air_flight_metadata(obj_names, ref_key):
    """task specific metadata."""

    data_ref = ref.__dict__[ref_key]

    cur_sym_infos = {}  # label based key
    loaded_models_info = data_ref.get_models_info()

    for i, obj_name in enumerate(obj_names):
        obj_id = data_ref.obj2id[obj_name]
        model_info = loaded_models_info[str(obj_id)]
        if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
            sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
            sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)
        else:
            sym_info = None
        cur_sym_infos[i] = sym_info

    meta = {"thing_classes": obj_names, "sym_infos": cur_sym_infos}
    return meta


AIR_OBJECTS = [
    "A320-sichuan",
    "A320-dragon",
    "A350-dongfang",
]  # no bowl, cup
AIR_OCC_OBJECTS = []
################################################################################

SPLITS_AIR = dict(
    air_flight_train=dict(
        name="air_flight_train",
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/"),
        objs=AIR_OBJECTS,  # selected objects
        ann_files=[
            osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/image_set/{}_{}.txt".format(_obj, "train"))
            for _obj in AIR_OBJECTS
        ],
        image_prefixes=[
            osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/{}".format(_obj))
            for _obj in AIR_OBJECTS
        ],
        skeleton_prefixes=[
            osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/{}".format(_obj))
            for _obj in AIR_OBJECTS
        ],
        # xyz_prefixes=[
        #     osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/test/xyz_crop/{:06d}".format(ref.air_flight_full.obj2id[_obj]))
        #     for _obj in AIR_OBJECTS
        # ],
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_skeleton=True,  # (load skeleton path here, but may not use it)
        # with_depth=True,  # (load depth path here, but may not use it)
        height=1000,
        width=1920,
        cache_dir=osp.join(PROJ_ROOT, ".cache"),
        use_cache=True,
        num_to_load=-1,
        filter_scene=True,
        filter_invalid=True,
        ref_key="air_flight_full",
    ),
    air_flight_test=dict(
        name="air_flight_test",
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/"),
        objs=AIR_OBJECTS,
        ann_files=[
            osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/image_set/{}_{}.txt".format(_obj, "test"))
            for _obj in AIR_OBJECTS
        ],
        # NOTE: scene root
        image_prefixes=[
            osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/{}".format(_obj))
            for _obj in AIR_OBJECTS
        ],
        skeleton_prefixes=[
            osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/{}".format(_obj))
            for _obj in AIR_OBJECTS
        ],
        # xyz_prefixes=[
        #     osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/test/xyz_crop/{:06d}".format(ref.air_flight_full.obj2id[_obj]))
        #     for _obj in AIR_OBJECTS
        # ],
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_skeleton=True,  # (load skeleton path here, but may not use it)
        # with_depth=True,  # (load depth path here, but may not use it)
        height=1000,
        width=1920,
        cache_dir=osp.join(PROJ_ROOT, ".cache"),
        use_cache=True,
        num_to_load=-1,
        filter_scene=True,
        filter_invalid=False,
        ref_key="air_flight_full",
    ),
    # air_flighto_train=dict(
    #     name="air_flighto_train",
    #     # use air_flight real all (8 objects) to train for air_flighto
    #     dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/"),
    #     models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/models"),
    #     objs=AIR_OCC_OBJECTS,  # selected objects
    #     ann_files=[
    #         osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/image_set/{}_{}.txt".format(_obj, "all"))
    #         for _obj in AIR_OCC_OBJECTS
    #     ],
    #     image_prefixes=[
    #         osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/test/{:06d}".format(ref.air_flight_full.obj2id[_obj]))
    #         for _obj in AIR_OCC_OBJECTS
    #     ],
    #     # xyz_prefixes=[
    #     #     osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/test/xyz_crop/{:06d}".format(ref.air_flight_full.obj2id[_obj]))
    #     #     for _obj in AIR_OCC_OBJECTS
    #     # ],
    #     scale_to_meter=0.001,
    #     with_masks=True,  # (load masks but may not use it)
    #     # with_depth=True,  # (load depth path here, but may not use it)
    #     height=480,
    #     width=640,
    #     cache_dir=osp.join(PROJ_ROOT, ".cache"),
    #     use_cache=True,
    #     num_to_load=-1,
    #     filter_scene=True,
    #     filter_invalid=True,
    #     ref_key="air_flight_full",
    # ),
    # air_flighto_test=dict(
    #     name="air_flighto_test",
    #     dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flighto/"),
    #     models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flighto/models"),
    #     objs=AIR_OCC_OBJECTS,
    #     ann_files=[osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flighto/image_set/air_flighto_test.txt")],
    #     # NOTE: scene root
    #     image_prefixes=[osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flighto/test/{:06d}").format(2)],
    #     # xyz_prefixes=[None],
    #     scale_to_meter=0.001,
    #     with_masks=True,  # (load masks but may not use it)
    #     # with_depth=True,  # (load depth path here, but may not use it)
    #     height=480,
    #     width=640,
    #     cache_dir=osp.join(PROJ_ROOT, ".cache"),
    #     use_cache=True,
    #     num_to_load=-1,
    #     filter_scene=False,
    #     filter_invalid=False,
    #     ref_key="air_flight_full",
    # ),
    # air_flighto_bop_test=dict(
    #     name="air_flighto_bop_test",
    #     dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flighto/"),
    #     models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flighto/models"),
    #     objs=AIR_OCC_OBJECTS,
    #     ann_files=[osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flighto/image_set/air_flighto_bop_test.txt")],
    #     # NOTE: scene root
    #     image_prefixes=[osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flighto/test/{:06d}").format(2)],
    #     # xyz_prefixes=[None],
    #     scale_to_meter=0.001,
    #     with_masks=True,  # (load masks but may not use it)
    #     # with_depth=True,  # (load depth path here, but may not use it)
    #     height=480,
    #     width=640,
    #     cache_dir=osp.join(PROJ_ROOT, ".cache"),
    #     use_cache=True,
    #     num_to_load=-1,
    #     filter_scene=False,
    #     filter_invalid=False,
    #     ref_key="air_flight_full",
    # ),
)

# single obj splits for air_flight real
for obj in ref.air_flight_full.objects:
    for split in ["train", "test", "all"]:
        name = "air_flight_real_{}_{}".format(obj, split)
        ann_files = [osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/image_set/{}_{}.txt".format(obj, split))]
        if split in ["train", "all"]:  # all is used to train air_flighto
            filter_invalid = True
        elif split in ["test"]:
            filter_invalid = False
        else:
            raise ValueError("{}".format(split))
        if name not in SPLITS_AIR:
            SPLITS_AIR[name] = dict(
                name=name,
                dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/"),
                models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/"),
                objs=[obj],  # only this obj
                ann_files=ann_files,
                image_prefixes=[osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/{}").format(obj)],
                skeleton_prefixes=[osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/{}").format(obj)],
                # xyz_prefixes=[
                #     osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/test/xyz_crop/{:06d}".format(ref.air_flight_full.obj2id[obj]))
                # ],
                scale_to_meter=0.001,
                with_masks=True,  # (load masks but may not use it)
                with_skeleton=True,  # (load skeleton path here, but may not use it)
                # with_depth=True,  # (load depth path here, but may not use it)
                height=1000,
                width=1920,
                cache_dir=osp.join(PROJ_ROOT, ".cache"),
                use_cache=True,
                num_to_load=-1,
                filter_invalid=filter_invalid,
                filter_scene=True,
                ref_key="air_flight_full",
            )

# # single obj splits for air_flighto_test
# for obj in ref.air_flight_full.objects:
#     for split in ["test"]:
#         name = "air_flighto_{}_{}".format(obj, split)
#         if split in ["train", "all"]:  # all is used to train air_flighto
#             filter_invalid = True
#         elif split in ["test"]:
#             filter_invalid = False
#         else:
#             raise ValueError("{}".format(split))
#         if name not in SPLITS_AIR:
#             SPLITS_AIR[name] = dict(
#                 name=name,
#                 dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flighto/"),
#                 models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flighto/models"),
#                 objs=[obj],
#                 ann_files=[osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flighto/image_set/air_flighto_test.txt")],
#                 # NOTE: scene root
#                 image_prefixes=[osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flighto/test/{:06d}").format(2)],
#                 # xyz_prefixes=[None],
#                 scale_to_meter=0.001,
#                 with_masks=True,  # (load masks but may not use it)
#                 # with_depth=True,  # (load depth path here, but may not use it)
#                 height=480,
#                 width=640,
#                 cache_dir=osp.join(PROJ_ROOT, ".cache"),
#                 use_cache=True,
#                 num_to_load=-1,
#                 filter_scene=False,
#                 filter_invalid=False,
#                 ref_key="air_flight_full",
#             )
#
# # single obj splits for air_flighto_bop_test
# for obj in ref.air_flight_full.objects:
#     for split in ["test"]:
#         name = "air_flighto_{}_bop_{}".format(obj, split)
#         if split in ["train", "all"]:  # all is used to train air_flighto
#             filter_invalid = True
#         elif split in ["test"]:
#             filter_invalid = False
#         else:
#             raise ValueError("{}".format(split))
#         if name not in SPLITS_AIR:
#             SPLITS_AIR[name] = dict(
#                 name=name,
#                 dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flighto/"),
#                 models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flighto/models"),
#                 objs=[obj],
#                 ann_files=[osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flighto/image_set/air_flighto_bop_test.txt")],
#                 # NOTE: scene root
#                 image_prefixes=[osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flighto/test/{:06d}").format(2)],
#                 # xyz_prefixes=[None],
#                 scale_to_meter=0.001,
#                 with_masks=True,  # (load masks but may not use it)
#                 # with_depth=True,  # (load depth path here, but may not use it)
#                 height=480,
#                 width=640,
#                 cache_dir=osp.join(PROJ_ROOT, ".cache"),
#                 use_cache=True,
#                 num_to_load=-1,
#                 filter_scene=False,
#                 filter_invalid=False,
#                 ref_key="air_flight_full",
#             )


# ================ add single 视角 dataset for debug =======================================
debug_im_ids = {"train": {obj: [] for obj in ref.air_flight_full.objects}, "test": {obj: [] for obj in ref.air_flight_full.objects}}
for obj in ref.air_flight_full.objects:
    for split in ["train", "test"]:
        cur_ann_file = osp.join(DATASETS_ROOT, f"BOP_DATASETS/air_flight/image_set/{obj}_{split}.txt")
        ann_files = [cur_ann_file]

        im_ids = []
        with open(cur_ann_file, "r") as f:
            for line in f:
                # scene_id(obj_id)/im_id
                im_ids.append("{}/{}".format(ref.air_flight_full.obj2id[obj], int(line.strip("\r\n").split()[0].split("/")[-2])))  # 某个目标下的某个视角

        debug_im_ids[split][obj] = im_ids
        for debug_im_id in debug_im_ids[split][obj]:
            name = "air_flight_single_{}_{}_{}".format(obj, debug_im_id.split("/")[-1], split)  # object_veiw_imgid_train or test
            if name not in SPLITS_AIR:
                SPLITS_AIR[name] = dict(
                    name=name,
                    dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/"),
                    models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/"),
                    objs=[obj],  # only this obj
                    ann_files=ann_files,
                    image_prefixes=[
                        osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/{}").format(obj)
                    ],
                    skeleton_prefixes=[osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/{}").format(obj)],
                    # xyz_prefixes=[
                    #     osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/test/xyz_crop/{:06d}".format(ref.air_flight_full.obj2id[obj]))
                    # ],
                    scale_to_meter=0.001,
                    with_masks=True,  # (load masks but may not use it)
                    with_skeleton=True,  # (load skeleton path here, but may not use it)
                    # with_depth=True,  # (load depth path here, but may not use it)
                    height=1000,
                    width=1920,
                    cache_dir=osp.join(PROJ_ROOT, ".cache"),
                    use_cache=True,
                    num_to_load=-1,
                    filter_invalid=False,
                    filter_scene=True,
                    ref_key="air_flight_full",
                    debug_im_id=debug_im_id,  # NOTE: debug im id
                )


# ================ add single image dataset for debug =======================================
debug_im_ids = {"train": {obj: [] for obj in ref.air_flight_full.objects}, "test": {obj: [] for obj in ref.air_flight_full.objects}}
for obj in ref.air_flight_full.objects:
    for split in ["train", "test"]:
        cur_ann_file = osp.join(DATASETS_ROOT, f"BOP_DATASETS/air_flight/image_set/{obj}_{split}.txt")
        ann_files = [cur_ann_file]

        im_ids = []
        with open(cur_ann_file, "r") as f:
            for line in f:
                # scene_id(obj_id)/im_id
                im_ids.append("{}/{}/{}".format(ref.air_flight_full.obj2id[obj], int(line.strip("\r\n").split(" ")[0].split("/")[-2]), int(line.strip("\r\n").split(" ")[0].split("_")[-1]))) # 某个目标下的某个视角下的某张图片

        debug_im_ids[split][obj] = im_ids
        for debug_im_id in debug_im_ids[split][obj]:
            name = "air_flight_single_{}_{}_{}_{}".format(obj, debug_im_id.split("/")[-2], debug_im_id.split("/")[-1], split) # object_veiw_imgid_train or test
            if name not in SPLITS_AIR:
                SPLITS_AIR[name] = dict(
                    name=name,
                    dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/"),
                    models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/"),
                    objs=[obj],  # only this obj
                    ann_files=ann_files,
                    image_prefixes=[
                        osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/{}/{}").format(obj, debug_im_id.split("/")[-2])
                    ],
                    skeleton_prefixes=[osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/{}/skeleton").format(obj)],
                    # xyz_prefixes=[
                    #     osp.join(DATASETS_ROOT, "BOP_DATASETS/air_flight/test/xyz_crop/{:06d}".format(ref.air_flight_full.obj2id[obj]))
                    # ],
                    scale_to_meter=0.001,
                    with_masks=True,  # (load masks but may not use it)
                    with_skeleton=True,  # (load skeleton path here, but may not use it)
                    # with_depth=True,  # (load depth path here, but may not use it)
                    height=1000,
                    width=1920,
                    cache_dir=osp.join(PROJ_ROOT, ".cache"),
                    use_cache=True,
                    num_to_load=-1,
                    filter_invalid=False,
                    filter_scene=True,
                    ref_key="air_flight_full",
                    debug_im_id=debug_im_id,  # NOTE: debug im id
                )


def register_with_name_cfg(name, data_cfg=None):
    """Assume pre-defined datasets live in `./datasets`.

    Args:
        name: datasnet_name,
        data_cfg: if name is in existing SPLITS, use pre-defined data_cfg
            otherwise requires data_cfg
            data_cfg can be set in cfg.DATA_CFG.name
    """
    dprint("register dataset: {}".format(name))
    if name in SPLITS_AIR:
        used_cfg = SPLITS_AIR[name]
    else:
        assert data_cfg is not None, f"dataset name {name} is not registered"
        used_cfg = data_cfg
    DatasetCatalog.register(name, AIR_Dataset(used_cfg))
    # something like eval_types
    MetadataCatalog.get(name).set(
        id="linemod",  # NOTE: for pvnet to determine module
        ref_key=used_cfg["ref_key"],
        objs=used_cfg["objs"],
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",
    )


def get_available_datasets():
    return list(SPLITS_AIR.keys())


def lines_ca(x1, x2, skeleton_3D):
    v = x2-x1  # 直线方向
    v = v / abs(v[0])
    for t in np.arange(0, x2[0]-x1[0]+1,(x2[0]-x1[0])/abs(x2[0]-x1[0])):
        x = x1[0] + v[0] * t
        y = x1[1] + v[1] * t
        z = x1[2] + v[2] * t
        skeleton_3D=np.append(skeleton_3D,np.array([[x,y,z]]),axis=0)
    return skeleton_3D

def get_model_corners(model):
    """
    model: Nx3
    corners_3d:8x3
    """
    min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
    min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
    min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def project3d_2d(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz=np.insert(xyz, 3, values=np.ones(xyz.shape[0]), axis=1)
    RT = np.insert(RT, 3, values=np.array([0,0,0,1]), axis=0)
    K = np.insert(K, 3, values=np.zeros([1,3]), axis=1)
    xyz_c = (RT).dot(xyz.T)
    xy_im = K.dot(xyz_c)
    xy = (xy_im[:2,:] / xy_im[2,:]).T
    # xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    # xyz = np.dot(xyz, K.T)
    # xy = xyz[:, :2] / xyz[:, 2:]
    return xy

#### tests ###############################################
def test_vis():
    dset_name = sys.argv[1]
    assert dset_name in DatasetCatalog.list()

    meta = MetadataCatalog.get(dset_name)
    dprint("MetadataCatalog: ", meta)
    objs = meta.objs

    t_start = time.perf_counter()
    dicts = DatasetCatalog.get(dset_name)
    logger.info("Done loading {} samples with {:.3f}s.".format(len(dicts), time.perf_counter() - t_start))

    dirname = "output/{}-data-vis".format(dset_name)
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = read_image_cv2(d["file_name"], format="BGR")
        skeleton = mmcv.imread(d["skeleton_file"], "unchanged")

        imH, imW = img.shape[:2]
        annos = d["annotations"]
        masks = [cocosegm2mask(anno["segmentation"], imH, imW) for anno in annos]
        bboxes = [anno["bbox"] for anno in annos]
        bbox_modes = [anno["bbox_mode"] for anno in annos]
        bboxes_xyxy = np.array(
            [BoxMode.convert(box, box_mode, BoxMode.XYXY_ABS) for box, box_mode in zip(bboxes, bbox_modes)]
        )
        kpts_3d_list = [anno["bbox3d_and_center"] for anno in annos]
        quats = [anno["quat"] for anno in annos]
        transes = [anno["trans"] for anno in annos]
        Rs = [quat2mat(quat) for quat in quats]
        # 0-based label
        cat_ids = [anno["category_id"] for anno in annos]
        K = d["cam"]
        kpts_2d = [misc.project_pts(kpt3d, K, R, t) for kpt3d, R, t in zip(kpts_3d_list, Rs, transes)]
        # # TODO: visualize pose and keypoints
        labels = [objs[cat_id] for cat_id in cat_ids]
        for _i in range(len(annos)):
            img_vis = vis_image_mask_bbox_cv2(
                img, masks[_i : _i + 1], bboxes=bboxes_xyxy[_i : _i + 1], labels=labels[_i : _i + 1]
            )
            img_vis_kpts2d = misc.draw_projected_box3d(img_vis.copy(), kpts_2d[_i])
            if "test" not in dset_name:
                # xyz_path = annos[_i]["xyz_path"]
                # xyz_info = mmcv.load(xyz_path)
                # x1, y1, x2, y2 = xyz_info["xyxy"]
                # xyz_crop = xyz_info["xyz_crop"].astype(np.float32)
                # xyz = np.zeros((imH, imW, 3), dtype=np.float32)
                # xyz[y1 : y2 + 1, x1 : x2 + 1, :] = xyz_crop
                # xyz_show = get_emb_show(xyz)
                # xyz_crop_show = get_emb_show(xyz_crop)
                # img_xyz = img.copy() / 255.0
                # mask_xyz = ((xyz[:, :, 0] != 0) | (xyz[:, :, 1] != 0) | (xyz[:, :, 2] != 0)).astype("uint8")
                # fg_idx = np.where(mask_xyz != 0)
                # img_xyz[fg_idx[0], fg_idx[1], :] = xyz_show[fg_idx[0], fg_idx[1], :3]
                # img_xyz_crop = img_xyz[y1 : y2 + 1, x1 : x2 + 1, :]
                #  img_vis_crop = img_vis[y1 : y2 + 1, x1 : x2 + 1, :]
                # # diff mask
                # diff_mask_xyz = np.abs(masks[_i] - mask_xyz)[y1 : y2 + 1, x1 : x2 + 1]

                grid_show(
                    [
                        img[:, :, [2, 1, 0]],
                        img_vis[:, :, [2, 1, 0]],
                        img_vis_kpts2d[:, :, [2, 1, 0]],
                        skeleton
                        # depth,
                        # xyz_show,
                        # diff_mask_xyz,
                        # xyz_crop_show,
                        # img_xyz[:, :, [2, 1, 0]],
                        #  img_xyz_crop[:, :, [2, 1, 0]],
                        # img_vis_crop,
                    ],
                    [
                        "img",
                        "vis_img",
                        "img_vis_kpts2d",
                        skeleton
                        # "depth",
                        # "diff_mask_xyz",
                        # "xyz_crop_show",
                        # "img_xyz",
                        # "img_xyz_crop",
                        # "img_vis_crop",
                    ],
                    row=3,
                    col=3,
                )
            else:
                grid_show(
                    [img[:, :, [2, 1, 0]], img_vis[:, :, [2, 1, 0]], img_vis_kpts2d[:, :, [2, 1, 0]], skeleton],
                    ["img", "vis_img", "img_vis_kpts2d", "skeleton"],
                    [img[:, :, [2, 1, 0]], img_vis[:, :, [2, 1, 0]], img_vis_kpts2d[:, :, [2, 1, 0]]],
                    ["img", "vis_img", "img_vis_kpts2d"],
                    row=2,
                    col=2,
                )


if __name__ == "__main__":
    """Test the  dataset loader.

    python this_file.py dataset_name
    """
    from lib.vis_utils.image import grid_show
    from lib.utils.setup_logger import setup_my_logger

    import detectron2.data.datasets  # noqa # add pre-defined metadata
    from lib.vis_utils.image import vis_image_mask_bbox_cv2
    from core.utils.utils import get_emb_show
    from core.utils.data_utils import read_image_cv2

    print("sys.argv:", sys.argv)
    logger = setup_my_logger(name="core")
    register_with_name_cfg(sys.argv[1])
    print("dataset catalog: ", DatasetCatalog.list())

    test_vis()
