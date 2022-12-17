# -*- coding: utf-8 -*-
import copy
import hashlib
import logging
import os
import os.path as osp
import pickle
import math
import random
import time
from tqdm import tqdm
import scipy.io as scio
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片

import cv2
import mmcv
import numpy as np
import ref
import torch
import torch.multiprocessing as mp
from core.base_data_loader import Base_DatasetFromList
from core.utils.augment import AugmentRGB
from core.utils.data_utils import crop_resize_by_warp_affine, get_2d_coord_np, read_image_cv2, xyz_to_region
from core.utils.dataset_utils import (
    filter_empty_dets,
    filter_invalid_in_dataset_dicts,
    flat_dataset_dicts,
    load_detections_into_dataset,
    my_build_batch_data_loader,
    trivial_batch_collator,
)
from core.utils.my_distributed_sampler import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from core.utils.rot_reps import mat_to_ortho6d_np
from core.utils.ssd_color_transform import ColorAugSSDTransform
from core.utils.utils import egocentric_to_allocentric
from core.utils import quaternion_lf, lie_algebra
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import transforms as T
from detectron2.structures import Boxes, BoxMode, Instances, Keypoints, PolygonMasks
from detectron2.utils.logger import log_first_n
from fvcore.common.file_io import PathManager
from lib.pysixd import inout, misc
from lib.utils.mask_utils import cocosegm2mask, get_edge
from lib.utils.utils import dprint, lazy_property
from transforms3d.quaternions import mat2quat

from .dataset_factory import register_datasets

logger = logging.getLogger(__name__)


def transform_instance_annotations(annotation, transforms, image_size, *, keypoint_hflip_indices=None):
    """
    NOTE: Adapted from detection_utils.
    Apply transforms to box, segmentation, keypoints, etc. of annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    im_H, im_W = image_size
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # Note that bbox is 1d (per-instance bounding box)
    annotation["bbox"] = np.array(transforms.apply_box([bbox])[0])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # NOTE: here we transform segms to binary masks (interp is nearest by default)
        mask = transforms.apply_segmentation(cocosegm2mask(annotation["segmentation"], h=im_H, w=im_W))
        annotation["segmentation"] = mask

    if "keypoints" in annotation:
        keypoints = utils.transform_keypoint_annotations(
            annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
        )
        annotation["keypoints"] = keypoints

    if "centroid_2d" in annotation:
        annotation["centroid_2d"] = transforms.apply_coords(np.array(annotation["centroid_2d"]).reshape(1, 2)).flatten()

    return annotation


def build_ske_augmentation(cfg, is_train):
    """Create a list of :class:`Augmentation` from config. when training 6d
    pose, cannot flip.

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    augmentation = []
    augmentation.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        # augmentation.append(T.RandomFlip())
        logger.info("Augmentations used in training: " + str(augmentation))
    return augmentation


class ske_DatasetFromList(Base_DatasetFromList):
    """NOTE: we can also use the default DatasetFromList and
    implement a similar custom DataMapper,
    but it is harder to implement some features relying on other dataset dicts
    # https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/common.py
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(self, cfg, split, lst: list, copy: bool = True, serialize: bool = True, flatten=True):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool): whether to hold memory using serialized objects, when
                enabled, data loader workers can use shared RAM from master
                process instead of making a copy.
        """
        #样本进行增广训练，图片会被缩放到一定的范围用于训练，详情请查看detectron ResizeShortestEdge的介绍
        self.augmentation = build_ske_augmentation(cfg, is_train=(split == "train"))
        if cfg.INPUT.COLOR_AUG_PROB > 0 and cfg.INPUT.COLOR_AUG_TYPE.lower() == "ssd":
            self.augmentation.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            logging.getLogger(__name__).info("Color augmnetation used in training: " + str(self.augmentation[-1]))
        # fmt: off
        self.img_format = cfg.INPUT.FORMAT  # default BGR
        self.with_depth = cfg.INPUT.WITH_DEPTH
        self.with_skeleton = True
        self.aug_depth = cfg.INPUT.AUG_DEPTH
        # NOTE: color augmentation config
        self.color_aug_prob = cfg.INPUT.COLOR_AUG_PROB
        self.color_aug_type = cfg.INPUT.COLOR_AUG_TYPE
        self.color_aug_code = cfg.INPUT.COLOR_AUG_CODE
        # fmt: on
        self.cfg = cfg
        self.split = split  # train | val | test
        if split == "train" and self.color_aug_prob > 0:
            self.color_augmentor = self._get_color_augmentor(aug_type=self.color_aug_type, aug_code=self.color_aug_code)
        else:
            self.color_augmentor = None
        # ------------------------
        # common model infos
        self.fps_points = {}
        self.model_points = {}
        self.extents = {}
        self.sym_infos = {}
        # ----------------------------------------------------
        self.flatten = flatten
        self._lst = flat_dataset_dicts(lst) if flatten else lst
        # ----------------------------------------------------
        self._copy = copy
        self._serialize = serialize

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            logger.info("Serializing {} elements to byte tensors and concatenating them all ...".format(len(self._lst)))
            self._lst = [_serialize(x) for x in self._lst]
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)
            logger.info("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024 ** 2))

    def __len__(self):
        if self._serialize:
            return len(self._addr)
        else:
            return len(self._lst)


    def _get_fps_points(self, dataset_name, with_center=False):
        """convert to label based keys.

        # TODO: get models info similarly
        """# 在3d model中采用farthest points sampling获取其中的 cfg.MODEL.CDPN.ROT_HEAD.NUM_REGIONS个点以及他们的均值
        if dataset_name in self.fps_points:
            return self.fps_points[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)  # 数据注册时evaluation中的信息
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]  # lm_full.py
        objs = dset_meta.objs
        cfg = self.cfg
        num_fps_points = cfg.MODEL.CDPN.ROT_HEAD.NUM_REGIONS
        cur_fps_points = {}
        loaded_fps_points = data_ref.get_fps_points()
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            if with_center:
                cur_fps_points[i] = loaded_fps_points[str(obj_id)][f"fps{num_fps_points}_and_center"]
            else:
                cur_fps_points[i] = loaded_fps_points[str(obj_id)][f"fps{num_fps_points}_and_center"][:-1]
        self.fps_points[dataset_name] = cur_fps_points
        return self.fps_points[dataset_name]

    def _get_model_points(self, dataset_name):
        """convert to label based keys."""
        if dataset_name in self.model_points:
            return self.model_points[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg

        cur_model_points = {}
        num = np.inf
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_path = osp.join(data_ref.model_dir, f"obj_{obj_id:06d}.ply")
            model = inout.load_ply(model_path, vertex_scale=data_ref.vertex_scale)
            cur_model_points[i] = pts = model["pts"]
            if pts.shape[0] < num:
                num = pts.shape[0]

        num = min(num, cfg.MODEL.CDPN.PNP_NET.NUM_PM_POINTS)
        for i in range(len(cur_model_points)):
            keep_idx = np.arange(num)
            np.random.shuffle(keep_idx)  # random sampling
            cur_model_points[i] = cur_model_points[i][keep_idx, :]

        self.model_points[dataset_name] = cur_model_points
        return self.model_points[dataset_name]

    def _get_extents(self, dataset_name):
        """label based keys."""
        if dataset_name in self.extents:
            return self.extents[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        try:
            ref_key = dset_meta.ref_key
        except:
            # FIXME: for some reason, in distributed training, this need to be re-registered
            register_datasets([dataset_name])
            dset_meta = MetadataCatalog.get(dataset_name)
            ref_key = dset_meta.ref_key

        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg

        cur_extents = {}
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            if "lm" in dataset_name:
                model_path = osp.join(data_ref.model_dir, f"obj_{obj_id:06d}.ply")
                model = inout.load_ply(model_path, vertex_scale=data_ref.vertex_scale)
                pts = model["pts"]
            elif "air" in dataset_name:
                 model = mmcv.load(osp.join(data_ref.model_dir, "{}/models/{}.json").format(obj_name, obj_name))
                 pts = np.array(model["pts"]).reshape(3,-1).T

            xmin, xmax = np.amin(pts[:, 0]), np.amax(pts[:, 0])
            ymin, ymax = np.amin(pts[:, 1]), np.amax(pts[:, 1])
            zmin, zmax = np.amin(pts[:, 2]), np.amax(pts[:, 2])
            size_x = xmax - xmin
            size_y = ymax - ymin
            size_z = zmax - zmin
            cur_extents[i] = np.array([size_x, size_y, size_z], dtype="float32")

        self.extents[dataset_name] = cur_extents
        return self.extents[dataset_name]

    def _get_sym_infos(self, dataset_name):
        """label based keys."""
        if dataset_name in self.sym_infos:
            return self.sym_infos[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg

        cur_sym_infos = {}
        loaded_models_info = data_ref.get_models_info()
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_info = loaded_models_info[str(obj_id)]
            if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
                sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
                sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)
            else:
                sym_info = None
            cur_sym_infos[i] = sym_info

        self.sym_infos[dataset_name] = cur_sym_infos
        return self.sym_infos[dataset_name]

    def read_data(self, dataset_dict):
        """load image and annos random shift & scale bbox; crop, rescale."""
        """ cfg是sken_base.py，dataset_dict是SPLITS_AIR中数据集整理后的信息"""
        t1=time.time()
        cfg = self.cfg
        r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
        pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET
        norm_scale = cfg.NORM_SCALE


        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        dataset_name = dataset_dict["dataset_name"]
        #runtime = 0.011s
        image = read_image_cv2(dataset_dict["file_name"], format=self.img_format)
        # should be consistent with the size in dataset_dict
        utils.check_image_size(dataset_dict, image)
        im_H_ori, im_W_ori = image.shape[:2]

        # currently only replace bg for train ###############################
        if self.split == "train": # TODO：公开数据几syn和real需要进行背景更换，但是飞机姿态估计不需要
            # some synthetic data already has bg, img_type should be real or something else but not syn
            img_type = dataset_dict.get("img_type", "real")
            if img_type == "syn":
                # log_first_n(logging.WARNING, "replace bg", n=10)
                assert "segmentation" in dataset_dict["inst_infos"]
                mask = cocosegm2mask(dataset_dict["inst_infos"]["segmentation"], im_H_ori, im_W_ori)
                image, mask_trunc = self.replace_bg(image.copy(), mask, return_mask=True)   #  image的背景换成了VOC图片的，mask_trunc中值换为了bool型
            elif img_type == "real":  # real image
                if np.random.rand() < cfg.INPUT.CHANGE_BG_PROB:
                    # log_first_n(logging.WARNING, "replace bg for real", n=10)
                    assert "segmentation" in dataset_dict["inst_infos"]
                    mask = cocosegm2mask(dataset_dict["inst_infos"]["segmentation"], im_H_ori, im_W_ori)
                    image, mask_trunc = self.replace_bg(image.copy(), mask, return_mask=True)
                else:
                    mask_trunc = None

        # NOTE: maybe add or change color augment here  但是这里没有做color augment处理===================================
        if self.split == "train" and self.color_aug_prob > 0 and self.color_augmentor is not None:
            if np.random.rand() < self.color_aug_prob:
                if cfg.INPUT.COLOR_AUG_SYN_ONLY and img_type not in ["real"]:
                    image = self._color_aug(image, self.color_aug_type)
                else:
                    image = self._color_aug(image, self.color_aug_type)

        # other transforms (mainly geometric ones);
        # for 6d pose task, flip is now allowed in general except for some 2d keypoints methods

        image, transforms = T.apply_augmentations(self.augmentation, image)
        im_H, im_W = image_shape = image.shape[:2]  # h, w
        # NOTE: scale camera intrinsic if necessary 计算上一步中apply_augmentation图片缩放比例================================
        scale_x = im_W / im_W_ori
        scale_y = im_H / im_H_ori  # NOTE: generally scale_x should be equal to scale_y
        if "cam" in dataset_dict:
            if im_W != im_W_ori or im_H != im_H_ori:
                dataset_dict["cam"][0] *= scale_x
                dataset_dict["cam"][1] *= scale_y
            K = dataset_dict["cam"].astype("float32")
            dataset_dict["cam"] = torch.as_tensor(K)

        input_res = cfg.MODEL.CDPN.BACKBONE.INPUT_RES
        out_res = cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES

        # TODO：CHW -> HWC将坐标由CHW形式转为HWC形式，懒得管有啥用
        coord_2d = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)
        t2=time.time()
        #################################################################################
        if self.split != "train":
            # don't load annotations at test time
            if "air" in dataset_dict["dataset_name"]:
                test_bbox_type = cfg.TEST.TEST_BBOX_TYPE
                norm_scale = cfg.NORM_SCALE
                if test_bbox_type == "gt":
                    bbox_key = "bbox"
                else:
                    bbox_key = f"bbox_{test_bbox_type}"
                assert not self.flatten, "Do not use flattened dicts for test!"
                # here get batched rois
                roi_infos = {}
                # yapf: disable
                roi_keys = ["obj_scene_im_id", "file_name", "cam", "im_H", "im_W",
                            "roi_img", "inst_id", "roi_coord_2d", "roi_cls", "score", "roi_extent",
                            bbox_key, "bbox_mode", "bbox_center", "roi_wh",
                            "scale", "resize_ratio", "model_info",
                            ]
                for _key in roi_keys:
                    roi_infos[_key] = []
                # yapf: enable
                # TODO: how to handle image without detections
                #   filter those when load annotations or detections, implement a function for this
                # "annotations" means detections
                for inst_i, inst_infos in enumerate(dataset_dict["annotations"]):
                    # inherent image-level infos
                    roi_infos["obj_scene_im_id"].append(dataset_dict["obj_scene_im_id"])
                    roi_infos["file_name"].append(dataset_dict["file_name"])
                    roi_infos["im_H"].append(im_H)
                    roi_infos["im_W"].append(im_W)
                    roi_infos["cam"].append(dataset_dict["cam"].cpu().numpy())

                    # roi-level infos
                    roi_infos["inst_id"].append(inst_i)
                    roi_infos["model_info"].append(inst_infos["model_info"])

                    roi_cls = inst_infos["category_id"]
                    roi_infos["roi_cls"].append(roi_cls)
                    roi_infos["score"].append(inst_infos["score"])

                    # extent
                    roi_extent = self._get_extents(dataset_name)[roi_cls]
                    roi_infos["roi_extent"].append(roi_extent)

                    bbox = BoxMode.convert(inst_infos[bbox_key], inst_infos["bbox_mode"], BoxMode.XYXY_ABS)
                    bbox = np.array(transforms.apply_box([bbox])[0])
                    roi_infos[bbox_key].append(bbox)
                    roi_infos["bbox_mode"].append(BoxMode.XYXY_ABS)
                    x1, y1, x2, y2 = bbox
                    bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
                    bw = max(x2 - x1, 1)
                    bh = max(y2 - y1, 1)
                    scale = max(bh, bw) * cfg.INPUT.DZI_PAD_SCALE
                    scale = min(scale, max(im_H, im_W)) * 1.0

                    roi_infos["bbox_center"].append(bbox_center.astype("float32"))
                    roi_infos["scale"].append(scale)
                    roi_infos["roi_wh"].append(np.array([bw, bh], dtype=np.float32))
                    roi_infos["resize_ratio"].append(out_res / scale)

                    # CHW, float32 tensor
                    # roi_image
                    roi_img = crop_resize_by_warp_affine(
                        image, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
                    ).transpose(2, 0, 1)

                    roi_img = self.normalize_image(cfg, roi_img)
                    roi_infos["roi_img"].append(roi_img.astype("float32"))

                    # roi_coord_2d
                    roi_coord_2d = crop_resize_by_warp_affine(
                        coord_2d, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
                    ).transpose(
                        2, 0, 1
                    )  # HWC -> CHW
                    roi_infos["roi_coord_2d"].append(roi_coord_2d.astype("float32"))

                for _key in roi_keys:
                    if _key in ["roi_img", "roi_coord_2d"]:
                        dataset_dict[_key] = torch.as_tensor(roi_infos[_key]).contiguous()
                    elif _key in ["model_info", "obj_scene_im_id", "file_name"]:
                        # can not convert to tensor
                        dataset_dict[_key] = roi_infos[_key]
                    else:
                        dataset_dict[_key] = torch.tensor(roi_infos[_key])

                return dataset_dict
            else:
                test_bbox_type = cfg.TEST.TEST_BBOX_TYPE
                if test_bbox_type == "gt":
                    bbox_key = "bbox"
                else:
                    bbox_key = f"bbox_{test_bbox_type}"
                assert not self.flatten, "Do not use flattened dicts for test!"
                # here get batched rois
                roi_infos = {}
                # yapf: disable
                roi_keys = ["scene_im_id", "file_name", "cam", "im_H", "im_W",
                            "roi_img", "inst_id", "roi_coord_2d", "roi_cls", "score", "roi_extent",
                            bbox_key, "bbox_mode", "bbox_center", "roi_wh",
                            "scale", "resize_ratio", "model_info",
                            ]
                for _key in roi_keys:
                    roi_infos[_key] = []
                # yapf: enable
                # TODO: how to handle image without detections
                #   filter those when load annotations or detections, implement a function for this
                # "annotations" means detections
                for inst_i, inst_infos in enumerate(dataset_dict["annotations"]):
                    # inherent image-level infos
                    roi_infos["scene_im_id"].append(dataset_dict["scene_im_id"])
                    roi_infos["file_name"].append(dataset_dict["file_name"])
                    roi_infos["im_H"].append(im_H)
                    roi_infos["im_W"].append(im_W)
                    roi_infos["cam"].append(dataset_dict["cam"].cpu().numpy())

                    # roi-level infos
                    roi_infos["inst_id"].append(inst_i)
                    roi_infos["model_info"].append(inst_infos["model_info"])

                    roi_cls = inst_infos["category_id"]
                    roi_infos["roi_cls"].append(roi_cls)
                    roi_infos["score"].append(inst_infos["score"])

                    # extent
                    roi_extent = self._get_extents(dataset_name)[roi_cls]
                    roi_infos["roi_extent"].append(roi_extent)

                    bbox = BoxMode.convert(inst_infos[bbox_key], inst_infos["bbox_mode"], BoxMode.XYXY_ABS)
                    bbox = np.array(transforms.apply_box([bbox])[0])
                    roi_infos[bbox_key].append(bbox)
                    roi_infos["bbox_mode"].append(BoxMode.XYXY_ABS)
                    x1, y1, x2, y2 = bbox
                    bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
                    bw = max(x2 - x1, 1)
                    bh = max(y2 - y1, 1)
                    scale = max(bh, bw) * cfg.INPUT.DZI_PAD_SCALE
                    scale = min(scale, max(im_H, im_W)) * 1.0

                    roi_infos["bbox_center"].append(bbox_center.astype("float32"))
                    roi_infos["scale"].append(scale)
                    roi_infos["roi_wh"].append(np.array([bw, bh], dtype=np.float32))
                    roi_infos["resize_ratio"].append(out_res / scale)

                    # CHW, float32 tensor
                    # roi_image
                    roi_img = crop_resize_by_warp_affine(
                        image, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
                    ).transpose(2, 0, 1)

                    roi_img = self.normalize_image(cfg, roi_img)
                    roi_infos["roi_img"].append(roi_img.astype("float32"))

                    # roi_coord_2d
                    roi_coord_2d = crop_resize_by_warp_affine(
                        coord_2d, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
                    ).transpose(
                        2, 0, 1
                    )  # HWC -> CHW
                    roi_infos["roi_coord_2d"].append(roi_coord_2d.astype("float32"))

                for _key in roi_keys:
                    if _key in ["roi_img", "roi_coord_2d"]:
                        dataset_dict[_key] = torch.as_tensor(roi_infos[_key]).contiguous()
                    elif _key in ["model_info", "scene_im_id", "file_name"]:
                        # can not convert to tensor
                        dataset_dict[_key] = roi_infos[_key]
                    else:
                        dataset_dict[_key] = torch.tensor(roi_infos[_key])

                return dataset_dict
        #######################################################################################
        # NOTE: currently assume flattened dicts for train
        assert self.flatten, "Only support flattened dicts for train now"
        inst_infos = dataset_dict.pop("inst_infos")
        dataset_dict["roi_cls"] = roi_cls = inst_infos["category_id"]  #TODO:对应的是air_flight_full中的cat2label,就是3种飞机的种类

        # extent
        if "air" in dataset_dict["dataset_name"]:
            roi_extent = np.array(inst_infos["aircraft_size"], dtype="float32")
        else:
            roi_extent = self._get_extents(dataset_name)[roi_cls]  #  返回的是在3d model中目标的尺寸大小size_x,y,z
        dataset_dict["roi_extent"] = torch.tensor(roi_extent, dtype=torch.float32) #用于摄自point match的loss时的权重

        # load xyz =======================================================
        t3 = time.time()
        if "air" not in dataset_dict["dataset_name"]:
            xyz_info = mmcv.load(inst_infos["xyz_path"])
            x1, y1, x2, y2 = xyz_info["xyxy"]
            # float16 does not affect performance (classification/regresion)
            xyz_crop = xyz_info["xyz_crop"]
            xyz = np.zeros((im_H, im_W, 3), dtype=np.float32)
            xyz[y1 : y2 + 1, x1 : x2 + 1, :] = xyz_crop
            # NOTE: full mask
            mask_obj = ((xyz[:, :, 0] != 0) | (xyz[:, :, 1] != 0) | (xyz[:, :, 2] != 0)).astype(np.bool).astype(np.float32)
            if cfg.INPUT.SMOOTH_XYZ:
                xyz = self.smooth_xyz(xyz)

            if cfg.TRAIN.VIS:
                xyz = self.smooth_xyz(xyz)

            # override bbox info using xyz_infos
            inst_infos["bbox"] = [x1, y1, x2, y2]
            inst_infos["bbox_mode"] = BoxMode.XYXY_ABS

        # USER: Implement additional transformations if you have other types of data
        # inst_infos.pop("segmentation")  # NOTE: use mask from xyz
        anno = transform_instance_annotations(inst_infos, transforms, image_shape, keypoint_hflip_indices=None)

        # augment bbox ===================================================
        bbox_xyxy = anno["bbox"]
        bbox_center, scale = self.aug_bbox(cfg, bbox_xyxy, im_H, im_W)  # bbox_center是原来的bbox的位置在一定范围内平移了一下，scale则是bbox的大小放大的最长边的大小
        bw = max(bbox_xyxy[2] - bbox_xyxy[0], 1)
        bh = max(bbox_xyxy[3] - bbox_xyxy[1], 1)
        tt=time.time()
        # CHW, float32 tensor
        """将图片仿射变换到backbone的输入大小尺寸256*256"""
        ## roi_image ------------------------------------
        roi_img = crop_resize_by_warp_affine(
            image, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)

        roi_img = self.normalize_image(cfg, roi_img)

        # roi_coord_2d ----------------------------------------------------
        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)
        t4 = time.time()
        ## roi_mask ---------------------------------------
        # (mask_trunc < mask_visib < mask_obj)三者去区别不大，只是mask_trunc是maks_visib的bool版本
        if "air" not in dataset_dict["dataset_name"]:
            mask_visib = anno["segmentation"].astype("float32") * mask_obj
        else:
            skeleton = cocosegm2mask(inst_infos["skeleton"], im_H_ori, im_W_ori)
            mask_visib = anno["segmentation"].astype("float32")
            mask_trunc = None  # 之前跳过了mask_trunc这一步，所以要重新申明一下
            mask_obj = mask_visib

        if mask_trunc is None:
            mask_trunc = mask_visib
        else:
            mask_trunc = mask_visib * mask_trunc.astype("float32")

        if cfg.TRAIN.VIS:  #  False
            mask_xyz_interp = cv2.INTER_LINEAR  # 采用双线性插值
        else:
            mask_xyz_interp = cv2.INTER_NEAREST  # 采用最邻近插值

        # maybe truncated mask (true mask for rgb)
        roi_mask_trunc = crop_resize_by_warp_affine(
            mask_trunc[:, :, None], bbox_center, scale, out_res, interpolation=mask_xyz_interp
        )

        # use original visible mask to calculate xyz loss (try full obj mask?)
        roi_mask_visib = crop_resize_by_warp_affine(
            mask_visib[:, :, None], bbox_center, scale, out_res, interpolation=mask_xyz_interp
        )

        roi_mask_obj = crop_resize_by_warp_affine(
            mask_obj[:, :, None], bbox_center, scale, out_res, interpolation=mask_xyz_interp
        )

        if "air" in dataset_dict["dataset_name"]:
            # no 2D skeleton
            if pnp_net_cfg.SKELETON_2D_DETECTION:
                ske_out_res_scale = cfg.MODEL.CDPN.ROT_HEAD.SKE_OUTPUT_RES_SCALE
                dks = cfg.SKE_DKS
                element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dks, dks))
                skeleton_dilate = cv2.dilate(skeleton, element)
                skeleton_crop = crop_resize_by_warp_affine(
                    skeleton_dilate[:, :, None], bbox_center, scale, out_res * ske_out_res_scale,
                    interpolation=cv2.INTER_LINEAR
                )
                roi_skeleton, roi_skeleton_mask, skeleton_orig = ref.air_flight_full.sklarge2flux(self, skeleton_crop)
            else:
                roi_skeleton = np.array([2,2])
                roi_skeleton_mask = np.array([2,2])
                skeleton_orig = np.array([2,2])

        if "air" not in dataset_dict["dataset_name"]:
            ## roi_xyz ----------------------------------------------------
            # T_begin = time.perf_counter()
            roi_xyz = crop_resize_by_warp_affine(xyz, bbox_center, scale, out_res, interpolation=mask_xyz_interp)

            # region label
            if r_head_cfg.NUM_REGIONS > 1:
                fps_points = self._get_fps_points(dataset_name)[roi_cls] # 在3d model中采用farthest points sampling获取其中的 cfg.MODEL.CDPN.ROT_HEAD.NUM_REGIONS个点以及他们的均值
                roi_region = xyz_to_region(roi_xyz, fps_points)  # HW
                dataset_dict["roi_region"] = torch.as_tensor(roi_region.astype(np.int32)).contiguous()

            roi_xyz = roi_xyz.transpose(2, 0, 1)  # HWC-->CHW
            # normalize xyz to [0, 1] using extent
            roi_xyz[0] = roi_xyz[0] / roi_extent[0] + 0.5
            roi_xyz[1] = roi_xyz[1] / roi_extent[1] + 0.5
            roi_xyz[2] = roi_xyz[2] / roi_extent[2] + 0.5

            if ("CE" in r_head_cfg.XYZ_LOSS_TYPE) or ("cls" in cfg.MODEL.CDPN.NAME):  # convert target to int for cls
                # assume roi_xyz has been normalized in [0, 1]
                roi_xyz_bin = np.zeros_like(roi_xyz)
                roi_x_norm = roi_xyz[0]
                roi_x_norm[roi_x_norm < 0] = 0  # clip
                roi_x_norm[roi_x_norm > 0.999999] = 0.999999
                # [0, BIN-1]
                roi_xyz_bin[0] = np.asarray(roi_x_norm * r_head_cfg.XYZ_BIN, dtype=np.uint8)

                roi_y_norm = roi_xyz[1]
                roi_y_norm[roi_y_norm < 0] = 0
                roi_y_norm[roi_y_norm > 0.999999] = 0.999999
                roi_xyz_bin[1] = np.asarray(roi_y_norm * r_head_cfg.XYZ_BIN, dtype=np.uint8)

                roi_z_norm = roi_xyz[2]
                roi_z_norm[roi_z_norm < 0] = 0
                roi_z_norm[roi_z_norm > 0.999999] = 0.999999
                roi_xyz_bin[2] = np.asarray(roi_z_norm * r_head_cfg.XYZ_BIN, dtype=np.uint8)

                # the last bin is for bg
                roi_masks = {"trunc": roi_mask_trunc, "visib": roi_mask_visib, "obj": roi_mask_obj}
                roi_mask_xyz = roi_masks[r_head_cfg.XYZ_LOSS_MASK_GT]
                roi_xyz_bin[0][roi_mask_xyz == 0] = r_head_cfg.XYZ_BIN
                roi_xyz_bin[1][roi_mask_xyz == 0] = r_head_cfg.XYZ_BIN
                roi_xyz_bin[2][roi_mask_xyz == 0] = r_head_cfg.XYZ_BIN

                if "CE" in r_head_cfg.XYZ_LOSS_TYPE:
                    dataset_dict["roi_xyz_bin"] = torch.as_tensor(roi_xyz_bin.astype("uint8")).contiguous()
                if "/" in r_head_cfg.XYZ_LOSS_TYPE and len(r_head_cfg.XYZ_LOSS_TYPE.split("/")[1]) > 0:
                    dataset_dict["roi_xyz"] = torch.as_tensor(roi_xyz.astype("float32")).contiguous()
            else:
                dataset_dict["roi_xyz"] = torch.as_tensor(roi_xyz.astype("float32")).contiguous()
            # T_end = time.perf_counter() - T_begin
            # print("total time11111: ", T_end)

        elif "air" in dataset_dict["dataset_name"]:
            T_begin = time.perf_counter()
            key_radius = cfg.MODEL.CDPN.ROT_HEAD.KEY_RADIUS
            num_key =r_head_cfg.NUM_KP_3D if cfg.SKELETON=='2D' else cfg.SKELETON_3D_KEY_NUM # np.size(inst_infos["air_keypoints_2d"], 1)  # 11 key-points and the last one the centriod point
            key_probility = np.zeros((image_shape[0], image_shape[1], num_key), dtype=np.float32)
            keypoints_3d_c_2d_position = np.zeros((image_shape[0], image_shape[1], num_key * 3), dtype=np.float32)
            keypoints_3d_c_2d = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.float32)
            skl_3d_c_2d_position = np.zeros((image_shape[0], image_shape[1], out_res * 3), dtype=np.float32)
            skl_3d_c_2d_probility = np.zeros((image_shape[0], image_shape[1], out_res), dtype=np.float32)

            if cfg.SKELETON=='2D':
                for channel in range(0, num_key):
                    # GT中的都是图像坐标系中的坐标，计算的时候换为矩阵坐标
                    # if '320-dragon_0847' in inst_infos['mask_full_file']:
                    #     a=1
                    kp_gt_h = inst_infos["air_keypoints_3d_2d"][1][channel] - 1  # 图像坐标系中的y坐标，转到矩阵中为行索引
                    kp_gt_w = inst_infos["air_keypoints_3d_2d"][0][channel] - 1  # 图像坐标系中的x坐标，转到矩阵中为列索引
                    kp_gt_3d_x = inst_infos["air_keypoints_3d_c"][0][channel]  # 相机坐标系中的x坐标
                    kp_gt_3d_y = inst_infos["air_keypoints_3d_c"][1][channel]  # 相机坐标系中的y坐标
                    kp_gt_3d_z = inst_infos["air_keypoints_3d_c"][2][channel]  # 相机坐标系中的z坐标
                    key_h = np.arange(kp_gt_h - 4 * key_radius, kp_gt_h + 4 * key_radius + 1)
                    key_w = np.arange(kp_gt_w - 4 * key_radius, kp_gt_w + 4 * key_radius + 1)
                    key_h = np.array([int(h_id) for h_id in key_h if h_id > 0 and h_id < image_shape[0]])
                    key_w = np.array([int(w_id) for w_id in key_w if w_id > 0 and w_id < image_shape[1]])
                    key_hh, key_ww = np.meshgrid(key_h, key_w)
                    key_hh = key_hh.reshape(1, -1)
                    key_ww = key_ww.reshape(1, -1)
                    key_range = (
                        np.exp((-1) * ((key_hh - kp_gt_h) ** 2 + (key_ww - kp_gt_w) ** 2) / (key_radius ** 2))).reshape(
                        (len(key_h), len(key_w)))
                    key_range[key_range < np.exp((-1) * (16))] = 0.0
                    key_probility[min(key_h):max(key_h) + 1, min(key_w):max(key_w) + 1, channel] = key_range
                    if cfg.MODEL.CDPN.ROT_HEAD.KP_PRESENT == 0:  # TODO: xxxxxyyyyyzzzzz
                        map = copy.deepcopy(key_range)
                        map[map > 0] = 1.0
                        keypoints_3d_c_2d_position[min(key_h):max(key_h) + 1, min(key_w):max(key_w) + 1,
                        channel] = map * kp_gt_3d_x
                        keypoints_3d_c_2d_position[min(key_h):max(key_h) + 1, min(key_w):max(key_w) + 1,
                        channel + num_key] = map * kp_gt_3d_y
                        keypoints_3d_c_2d_position[min(key_h):max(key_h) + 1, min(key_w):max(key_w) + 1,
                        channel + 2 * num_key] = map * kp_gt_3d_z
                        # keypoints_3d_c_2d_position[min(key_h):max(key_h)+1, min(key_w):max(key_w)+1, channel] = np.ones((len(key_h), len(key_w))) * kp_gt_3d_x
                        # keypoints_3d_c_2d_position[min(key_h):max(key_h)+1, min(key_w):max(key_w)+1, channel+num_key] = np.ones((len(key_h), len(key_w))) * kp_gt_3d_y
                        # keypoints_3d_c_2d_position[min(key_h):max(key_h)+1, min(key_w):max(key_w)+1, channel+2*num_key] = np.ones((len(key_h), len(key_w))) * kp_gt_3d_z

                    elif cfg.MODEL.CDPN.ROT_HEAD.KP_PRESENT == 1:  # TODO: xyzxyzyxzyxzxyz
                        keypoints_3d_c_2d_position[min(key_h):max(key_h) + 1, min(key_w):max(key_w) + 1,
                        channel * 3] = np.ones((len(key_h), len(key_w))) * kp_gt_3d_x
                        keypoints_3d_c_2d_position[min(key_h):max(key_h) + 1, min(key_w):max(key_w) + 1,
                        channel * 3 + 1] = np.ones((len(key_h), len(key_w))) * kp_gt_3d_y
                        keypoints_3d_c_2d_position[min(key_h):max(key_h) + 1, min(key_w):max(key_w) + 1,
                        channel * 3 + 2] = np.ones((len(key_h), len(key_w))) * kp_gt_3d_z

                    elif cfg.MODEL.CDPN.ROT_HEAD.KP_PRESENT == 2:  # TODO: xyz
                        keypoints_3d_c_2d[min(key_h):max(key_h) + 1, min(key_w):max(key_w) + 1, 0] = np.ones(
                            (len(key_h), len(key_w))) * inst_infos["air_keypoints_3d_c"][0][channel]
                        keypoints_3d_c_2d[min(key_h):max(key_h) + 1, min(key_w):max(key_w) + 1, 1] = np.ones(
                            (len(key_h), len(key_w))) * inst_infos["air_keypoints_3d_c"][1][channel]
                        keypoints_3d_c_2d[min(key_h):max(key_h) + 1, min(key_w):max(key_w) + 1, 2] = np.ones(
                            (len(key_h), len(key_w))) * inst_infos["air_keypoints_3d_c"][2][channel]
                        keypoints_3d_c_2d_position = keypoints_3d_c_2d
                    else:
                        raise ValueError(
                            f"Unknown cfg.MODEL.CDPN.ROT_HEAD.KP_PRESENT type: {cfg.MODEL.CDPN.ROT_HEAD.KP_PRESENT}")
            elif cfg.SKELETON=='3D' and True:
                for channel in range(0, num_key):
                    kp_gt_h = round(inst_infos["skeleton_3D_small_R"][1][channel] - 1)  # 图像坐标系中的y坐标，转到矩阵中为行索引
                    kp_gt_w = round(inst_infos["skeleton_3D_small_R"][0][channel] - 1)  # 图像坐标系中的x坐标，转到矩阵中为列索引
                    kp_gt_d = round(inst_infos["skeleton_3D_small_R"][2][channel] - 1)  # 图像坐标系中的z坐标，转到矩阵中为深度索引

                    kp_gt_3d_x = inst_infos["skeleton_3d_c"][0][channel]  # 相机坐标系中的x坐标
                    kp_gt_3d_y = inst_infos["skeleton_3d_c"][1][channel]  # 相机坐标系中的y坐标
                    kp_gt_3d_z = inst_infos["skeleton_3d_c"][2][channel]  # 相机坐标系中的z坐标

                    key_h = np.arange(kp_gt_h - 4 * key_radius, kp_gt_h + 4 * key_radius + 1)
                    key_w = np.arange(kp_gt_w - 4 * key_radius, kp_gt_w + 4 * key_radius + 1)
                    key_h = np.array([int(h_id) for h_id in key_h if h_id > 0 and h_id < image_shape[0]])
                    key_w = np.array([int(w_id) for w_id in key_w if w_id > 0 and w_id < image_shape[1]])
                    key_hh, key_ww = np.meshgrid(key_h, key_w)
                    key_hh = key_hh.reshape(1, -1)
                    key_ww = key_ww.reshape(1, -1)
                    key_range = (np.exp((-1) * ((key_hh - kp_gt_h) ** 2 + (key_ww - kp_gt_w) ** 2) / (key_radius ** 2))).reshape((len(key_h), len(key_w)))
                    key_range[key_range < np.exp((-1) * (16))] = 0.0
                    skl_3d_c_2d_probility[min(key_h):max(key_h) + 1, min(key_w):max(key_w) + 1, kp_gt_d] = key_range
                    if cfg.MODEL.CDPN.ROT_HEAD.KP_PRESENT == 0:  # TODO: xxxxxyyyyyzzzzz
                        map = copy.deepcopy(key_range)
                        map[map > 0] = 1.0
                        skl_3d_c_2d_position[min(key_h):max(key_h) + 1, min(key_w):max(key_w) + 1, kp_gt_d] = map * kp_gt_3d_x
                        skl_3d_c_2d_position[min(key_h):max(key_h) + 1, min(key_w):max(key_w) + 1, kp_gt_d + out_res] = map * kp_gt_3d_y
                        skl_3d_c_2d_position[min(key_h):max(key_h) + 1, min(key_w):max(key_w) + 1, kp_gt_d + 2 * out_res] = map * kp_gt_3d_z


                # plt.imshow(roi_img[0,:,:])
                # plt.scatter(skl_3d_kp[:, 0],-skl_3d_kp[ :,1])
                # plt.show()
                key_probility = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.float32)
                keypoints_3d_c_2d_position = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.float32)
                keypoints_3d_c_2d = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.float32)
            else:

                bbox_visib = inst_infos['bbox']
                skl_3d_c_2d_position[int(max(0,bbox_visib[1]-4*key_radius)): int(min(int(image.shape[0]), bbox_visib[1]+bbox_visib[3]+4*key_radius)),
                                              int(max(0, bbox_visib[0] - 4 * key_radius)): int(min(int(image.shape[1]), bbox_visib[0] + bbox_visib[ 2] + 4 * key_radius)), :]\
                    = inst_infos['skl_3d_c_2d_position']
                skl_3d_c_2d_probility[int(max(0,bbox_visib[1]-4*key_radius)): int(min(int(image.shape[0]), bbox_visib[1]+bbox_visib[3]+4*key_radius)),
                                              int(max(0, bbox_visib[0] - 4 * key_radius)): int(min(int(image.shape[1]), bbox_visib[0] + bbox_visib[ 2] + 4 * key_radius)), :]\
                     = inst_infos['skl_3d_c_2d_probility']

                key_probility = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.float32)
                keypoints_3d_c_2d_position = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.float32)
                keypoints_3d_c_2d = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.float32)
            T_end = time.perf_counter() - T_begin
            # print("total time: ", T_end)


            key_probility = crop_resize_by_warp_affine(
                key_probility, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)  # HWC-->CHW 在图片中顺序一般是h w c, 在cnn中一般顺序是b c h w
            keypoints_3d_c_2d_position = crop_resize_by_warp_affine(
                keypoints_3d_c_2d_position, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)  # HWC-->CHW 在图片中顺序一般是h w c, 在cnn中一般顺序是b c h w
            skl_3d_c_2d_probility = crop_resize_by_warp_affine(
                skl_3d_c_2d_probility, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)  # HWC-->CHW 在图片中顺序一般是h w c, 在cnn中一般顺序是b c h w
            skl_3d_c_2d_position = crop_resize_by_warp_affine(
                skl_3d_c_2d_position, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)  # HWC-->CHW 在图片中顺序一般是h w c, 在cnn中一般顺序是b c h w

            dataset_dict["key_probility"] = torch.as_tensor(key_probility.astype("float32")).contiguous()
            dataset_dict["skl_probility"] = torch.as_tensor(skl_3d_c_2d_probility.astype("float32")).contiguous()
            dataset_dict["keypoints_3d_c_2d_position"] = torch.as_tensor(keypoints_3d_c_2d_position.astype("float32")).contiguous()
            dataset_dict["skl_3d_c_2d_position"] = torch.as_tensor(skl_3d_c_2d_position.astype("float32")).contiguous()

        t5=time.time()
        # pose targets ----------------------------------------------------------------------
        pose = inst_infos["pose"]
        allo_pose = egocentric_to_allocentric(pose)
        quat = inst_infos["quat"]
        allo_quat = mat2quat(allo_pose[:3, :3])

        # ====== actually not needed ==========
        if pnp_net_cfg.ROT_TYPE == "allo_quat":
            dataset_dict["allo_quat"] = torch.as_tensor(allo_quat.astype("float32"))
        elif pnp_net_cfg.ROT_TYPE == "ego_quat":
            dataset_dict["ego_quat"] = torch.as_tensor(quat.astype("float32"))
        # rot6d
        elif pnp_net_cfg.ROT_TYPE == "ego_rot6d":
            dataset_dict["ego_rot6d"] = torch.as_tensor(mat_to_ortho6d_np(pose[:3, :3].astype("float32")))
        elif pnp_net_cfg.ROT_TYPE == "allo_rot6d":
            dataset_dict["allo_rot6d"] = torch.as_tensor(mat_to_ortho6d_np(allo_pose[:3, :3].astype("float32")))
        # log quat
        elif pnp_net_cfg.ROT_TYPE == "ego_log_quat":
            dataset_dict["ego_log_quat"] = quaternion_lf.qlog(torch.as_tensor(quat.astype("float32"))[None])[0]
        elif pnp_net_cfg.ROT_TYPE == "allo_log_quat":
            dataset_dict["allo_log_quat"] = quaternion_lf.qlog(torch.as_tensor(allo_quat.astype("float32"))[None])[0]
        # lie vec
        elif pnp_net_cfg.ROT_TYPE == "ego_lie_vec":
            dataset_dict["ego_lie_vec"] = lie_algebra.rot_to_lie_vec(
                torch.as_tensor(pose[:3, :3].astype("float32")[None])
            )[0]
        elif pnp_net_cfg.ROT_TYPE == "allo_lie_vec":
            dataset_dict["allo_lie_vec"] = lie_algebra.rot_to_lie_vec(
                torch.as_tensor(allo_pose[:3, :3].astype("float32"))[None]
            )[0]
        else:
            raise ValueError(f"Unknown rot type: {pnp_net_cfg.ROT_TYPE}")
        if "air" in dataset_dict["dataset_name"]:
            dataset_dict["ego_rot"] = torch.as_tensor(pose[:3, :3].astype("float32"))
            dataset_dict["trans"] = torch.as_tensor(inst_infos["trans"].astype("float32"))

            dataset_dict["roi_points"] = torch.as_tensor(
                inst_infos["model"].astype("float32")) / 100.0 # 转化为米m为单位,
            # dataset_dict["sym_info"] = self._get_sym_infos(dataset_name)[roi_cls]

            dataset_dict["roi_img"] = torch.as_tensor(roi_img.astype("float32")).contiguous()
            dataset_dict["roi_coord_2d"] = torch.as_tensor(roi_coord_2d.astype("float32")).contiguous()

            dataset_dict["roi_mask_trunc"] = torch.as_tensor(roi_mask_trunc.astype("float32")).contiguous()
            dataset_dict["roi_mask_visib"] = torch.as_tensor(roi_mask_visib.astype("float32")).contiguous()
            dataset_dict["roi_mask_obj"] = torch.as_tensor(roi_mask_obj.astype("float32")).contiguous()

            dataset_dict["skeleton"] = torch.as_tensor(roi_skeleton.astype("float32")).contiguous()
            dataset_dict["skeleton_mask"] = torch.as_tensor(roi_skeleton_mask.astype("float32")).contiguous()
            dataset_dict["skeleton_orig"] = torch.as_tensor(skeleton_orig.astype("float32")).contiguous()

            dataset_dict["bbox_center"] = torch.as_tensor(bbox_center, dtype=torch.float32)
            dataset_dict["scale"] = scale
            dataset_dict["bbox"] = anno["bbox"]  # NOTE: original bbox
            dataset_dict["roi_wh"] = torch.as_tensor(np.array([bw, bh], dtype=np.float32))
            dataset_dict["resize_ratio"] = resize_ratio = out_res / scale
            z_ratio = inst_infos["trans"][2] / resize_ratio
            obj_center = anno["centroid_2d"]
            delta_c = obj_center - bbox_center
            dataset_dict["trans_ratio"] = torch.as_tensor([delta_c[0] / bw, delta_c[1] / bh, z_ratio]).to(torch.float32)

        else:
            dataset_dict["ego_rot"] = torch.as_tensor(pose[:3, :3].astype("float32"))
            dataset_dict["trans"] = torch.as_tensor(inst_infos["trans"].astype("float32"))

            # 在model上随机选取NUM_PM_POINTS个点的法向量nx, ny, nz，然后用于在计算point matching中，用于计算Rx+T的损失值.
            dataset_dict["roi_points"] = torch.as_tensor(
                self._get_model_points(dataset_name)[roi_cls].astype("float32"))
            dataset_dict["sym_info"] = self._get_sym_infos(dataset_name)[roi_cls]

            dataset_dict["roi_img"] = torch.as_tensor(roi_img.astype("float32")).contiguous()
            dataset_dict["roi_coord_2d"] = torch.as_tensor(roi_coord_2d.astype("float32")).contiguous()

            dataset_dict["roi_mask_trunc"] = torch.as_tensor(roi_mask_trunc.astype("float32")).contiguous()
            dataset_dict["roi_mask_visib"] = torch.as_tensor(roi_mask_visib.astype("float32")).contiguous()
            dataset_dict["roi_mask_obj"] = torch.as_tensor(roi_mask_obj.astype("float32")).contiguous()

            dataset_dict["bbox_center"] = torch.as_tensor(bbox_center, dtype=torch.float32)
            dataset_dict["scale"] = scale
            dataset_dict["bbox"] = anno["bbox"]  # NOTE: original bbox
            dataset_dict["roi_wh"] = torch.as_tensor(np.array([bw, bh], dtype=np.float32))
            dataset_dict["resize_ratio"] = resize_ratio = out_res / scale
            z_ratio = inst_infos["trans"][2] / resize_ratio
            obj_center = anno["centroid_2d"]
            delta_c = obj_center - bbox_center
            dataset_dict["trans_ratio"] = torch.as_tensor([delta_c[0] / bw, delta_c[1] / bh, z_ratio]).to(torch.float32)
        # print('runtime: %f  %f   %f   %f   %f   %f'%(t2-t1,t3-t2,tt-t3,t5-t4,time.time()-t5, 24*(time.time()-t1) ))
        return dataset_dict

    def smooth_xyz(self, xyz):
        """smooth the edge areas to reduce noise."""
        xyz = np.asarray(xyz, np.float32)
        xyz_blur = cv2.medianBlur(xyz, 3)
        edges = get_edge(xyz)
        xyz[edges != 0] = xyz_blur[edges != 0]
        return xyz

    def __getitem__(self, idx):
        if self.split != "train":
            dataset_dict = self._get_sample_dict(idx)
            return self.read_data(dataset_dict)

        while True:  # return valid data for train
            dataset_dict = self._get_sample_dict(idx)
            processed_data = self.read_data(dataset_dict)
            if processed_data is None:
                idx = self._rand_another(idx)
                continue
            return processed_data


def build_ske_train_loader(cfg, dataset_names):
    """A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Coordinate a random shuffle order shared among all processes (all GPUs)
    3. Each process spawn another few workers to process the dicts. Each worker will:
       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will yield.

    Args:
        cfg (CfgNode): the config

    Returns:
        an infinite iterator of training data
    """
    #TODO：生成训练集索引，包含了训练所需要的所有信息
    dataset_dicts = get_detection_dataset_dicts(
        dataset_names,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE if cfg.MODEL.KEYPOINT_ON else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )

    # TODO：删除visib_fract小于阈值的训练样本，但是所有的标注都不包含visib_fract值，所以都没有删除
    dataset_dicts = filter_invalid_in_dataset_dicts(dataset_dicts, visib_thr=cfg.DATALOADER.FILTER_VISIB_THR)

    for dict_num in range(len(dataset_dicts)): #TODO：Relative path error, change to absolute path
        if "air_flight_train" in dataset_names or "air_flight_test" in dataset_names: #TODO：针对机场
            filename = dataset_dicts[dict_num]['file_name']
            dataset_dicts[dict_num]['file_name'] = osp.join(osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), "../../")), filename)
        else: #TODO：针对公开数据集
            filename = dataset_dicts[dict_num]['file_name']
            depthfile = dataset_dicts[dict_num]['depth_file']
            dataset_dicts[dict_num]['file_name'] = osp.join(osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), "../../")), filename)
            dataset_dicts[dict_num]['depth_file'] = osp.join(osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), "../../")), depthfile)
    #TODO：进行了样本尺寸的增广训练，颜色的不增广，然后将15375个元素序列化为字节张量，并将它们全部连接起来
    dataset = ske_DatasetFromList(cfg, split="train", lst=dataset_dicts, copy=False)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    # TODO avoid if-else?
    if sampler_name == "TrainingSampler":
        #TODO：似乎是随机打乱，shuffle的意思，然后根据数据长度获取随机seed
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    #TODO：以batch的方式加载数据，就是常见的dataloader，只是这里弄了一下horovod和其它变量
    return my_build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_ske_test_loader(cfg, dataset_name, train_objs=None):
    """Similar to `build_detection_train_loader`. But this function uses the
    given `dataset_name` argument (instead of the names in cfg), and uses batch
    size 1.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    # load test detection results
    if cfg.MODEL.LOAD_DETS_TEST:
        det_files = cfg.DATASETS.DET_FILES_TEST
        assert len(cfg.DATASETS.TEST) == len(det_files)
        load_detections_into_dataset(
            dataset_name,
            dataset_dicts,
            det_file=det_files[cfg.DATASETS.TEST.index(dataset_name)],
            top_k_per_obj=cfg.DATASETS.DET_TOPK_PER_OBJ,
            score_thr=cfg.DATASETS.DET_THR,
            train_objs=train_objs,
        )
        if cfg.DATALOADER.FILTER_EMPTY_DETS:
            dataset_dicts = filter_empty_dets(dataset_dicts)
    for dict_num in range(len(dataset_dicts)): #Relative path error, change to absolute path
        if "air_flight_train" in dataset_name or "air_flight_test" in dataset_name: #TODO：针对机场
            filename = dataset_dicts[dict_num]['file_name']
            dataset_dicts[dict_num]['file_name'] = osp.join(osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), "../../")), filename)
        else: #TODO：针对公开数据集
            filename = dataset_dicts[dict_num]['file_name']
            depthfile = dataset_dicts[dict_num]['depth_file']
            dataset_dicts[dict_num]['file_name'] = osp.join(osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), "../../")), filename)
            dataset_dicts[dict_num]['depth_file'] = osp.join(osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), "../../")), depthfile)

    dataset = ske_DatasetFromList(cfg, split="test", lst=dataset_dicts, flatten=False)

    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    # Horovod: limit # of CPU threads to be used per worker.
    # if num_workers > 0:
    #     torch.set_num_threads(num_workers)

    kwargs = {"num_workers": num_workers}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    # https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py
    # if (num_workers > 0 and hasattr(mp, '_supports_context') and
    #         mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
    #     kwargs['multiprocessing_context'] = 'forkserver'
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler, collate_fn=trivial_batch_collator, **kwargs
    )
    return data_loader