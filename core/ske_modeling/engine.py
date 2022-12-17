# coding : utf-8
import logging
import os
import os.path as osp
import torch
from torch.cuda.amp import autocast, GradScaler
import mmcv
import time
import cv2
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片

from detectron2.utils.events import EventStorage
from detectron2.checkpoint import PeriodicCheckpointer
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    print_csv_format,
)

from detectron2.data.common import AspectRatioGroupedDataset
from detectron2.data import MetadataCatalog

from lib.utils.utils import dprint, iprint, get_time_str

from core.utils import solver_utils
import core.utils.my_comm as comm
from core.utils.my_checkpoint import MyCheckpointer
from core.utils.my_writer import MyCommonMetricPrinter, MyJSONWriter, MyPeriodicWriter, MyTensorboardXWriter
from core.utils.utils import get_emb_show
from core.utils.data_utils import denormalize_image
from .data_loader import build_ske_train_loader, build_ske_test_loader
from .engine_utils import batch_data, get_out_coor, get_out_mask
from .ske_evaluator import ske_inference_on_dataset, ske_Evaluator, save_result_of_dataset
from .ske_custom_evaluator import ske_EvaluatorCustom
import ref

# try:
#     import horovod.torch as hvd
# except ImportError:
#     print("You requested to import horovod which is missing or not supported for your OS.")


logger = logging.getLogger(__name__)


def get_evaluator(cfg, dataset_name, output_folder=None):
    """Create evaluator(s) for a given dataset.

    This uses the special metadata "evaluator_type" associated with each
    builtin dataset. For your own dataset, you can simply create an
    evaluator manually in your script and do not have to worry about the
    hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = osp.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)

    _distributed = comm.get_world_size() > 1
    dataset_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    train_obj_names = dataset_meta.objs
    if evaluator_type == "bop":
        if cfg.VAL.get("USE_BOP", False):
            return ske_Evaluator(
                cfg, dataset_name, distributed=_distributed, output_dir=output_folder, train_objs=train_obj_names
            )
        else:
            return ske_EvaluatorCustom(
                cfg, dataset_name, distributed=_distributed, output_dir=output_folder, train_objs=train_obj_names
            )

    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model, epoch=None, iteration=None):
    results = OrderedDict()
    model_name = osp.basename(cfg.MODEL.WEIGHTS).split(".")[0]
    for dataset_name in cfg.DATASETS.TEST:
        if epoch is not None and iteration is not None:
            evaluator = get_evaluator(
                cfg, dataset_name, osp.join(cfg.OUTPUT_DIR, f"inference_epoch_{epoch}_iter_{iteration}", dataset_name)
            )
        else:
            evaluator = get_evaluator(
                cfg, dataset_name, osp.join(cfg.OUTPUT_DIR, f"inference_{model_name}", dataset_name)
            )  # init
        data_loader = build_ske_test_loader(cfg, dataset_name, train_objs=evaluator.train_objs)
        results_i = ske_inference_on_dataset(cfg, model, data_loader, evaluator, amp_test=cfg.TEST.AMP_TEST)
        results[dataset_name] = results_i
        # if comm.is_main_process():
        #     logger.info("Evaluation results for {} in csv format:".format(dataset_name))
        #     print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def get_tbx_event_writer(out_dir, backup=False):
    tb_logdir = osp.join(out_dir, "tb")
    mmcv.mkdir_or_exist(tb_logdir)
    if backup:
        old_tb_logdir = osp.join(out_dir, "tb_old")
        mmcv.mkdir_or_exist(old_tb_logdir)
        os.system("mv -v {} {}".format(osp.join(tb_logdir, "events.*"), old_tb_logdir))

    tbx_event_writer = MyTensorboardXWriter(tb_logdir, backend="tensorboardX")
    return tbx_event_writer


def do_train(cfg, args, model, optimizer, resume=False):
    # use BatchNormalizetion() and Dropout()
    model.train()

    # some basic settings =========================
    dataset_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) # 加载的就是注册的数据集中的evaluation信息
    data_ref = ref.__dict__[dataset_meta.ref_key] # lm_full.py
    obj_names = dataset_meta.objs # 13个目标


# load data ===================================
    train_dset_names = cfg.DATASETS.TRAIN # 训练的数据集
    data_loader = build_ske_train_loader(cfg, train_dset_names)
    data_loader_iter = iter(data_loader)
    # load 2nd train dataloader if needed
    train_2_dset_names = cfg.DATASETS.get("TRAIN2", ())
    train_2_ratio = cfg.DATASETS.get("TRAIN2_RATIO", 0.0) #  train_2_ratio == 0
    if train_2_ratio > 0.0 and len(train_2_dset_names) > 0:
        data_loader_2 = build_ske_train_loader(cfg, train_2_dset_names)
        data_loader_2_iter = iter(data_loader_2)
    else:
        data_loader_2 = None
        data_loader_2_iter = None

    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    if isinstance(data_loader, AspectRatioGroupedDataset): # 不运行这里if，运行else
        dataset_len = len(data_loader.dataset.dataset)
        if data_loader_2 is not None:
            dataset_len += len(data_loader_2.dataset.dataset)
        iters_per_epoch = dataset_len // images_per_batch
    else:
        dataset_len = len(data_loader.dataset)
        if data_loader_2 is not None:
            dataset_len += len(data_loader_2.dataset)
        iters_per_epoch = dataset_len // images_per_batch
    max_iter = cfg.SOLVER.TOTAL_EPOCHS * iters_per_epoch
    dprint("images_per_batch: ", images_per_batch)
    dprint("dataset length: ", dataset_len)
    dprint("iters per epoch: ", iters_per_epoch)
    dprint("total iters: ", max_iter)
    scheduler = solver_utils.build_lr_scheduler(cfg, optimizer, total_iters=max_iter) #设置为余弦退火学习率调节方式，慢-快-慢
    """自动混合精度使用，就是利用单精度（FP32）和半精度（FP16）混合使用，一个精度高，但是速度慢，另外一个相反，所以混合使用。"""
    AMP_ON = cfg.SOLVER.AMP.ENABLED
    logger.info(f"AMP enabled: {AMP_ON}")
    """使用了AMP还是可能出现无法收敛的情况，原因是激活梯度太小，造成溢出，所以使用GradScaler（）来放大loss防止梯度的underflow，
    在backward时，损失变化变大，激活函数梯度不会溢出，backward后，权重梯度再缩小获取。"""
    grad_scaler = GradScaler()

    # resume or load model ===================================
    checkpointer = MyCheckpointer(
        model,
        cfg.OUTPUT_DIR,
        optimizer=optimizer,
        scheduler=scheduler,
        gradscaler=grad_scaler,
        save_to_disk=comm.is_main_process(),
    )
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    if comm._USE_HVD:  # hvd may be not available, so do not use the one in args
        # not needed
        # start_iter = hvd.broadcast(torch.tensor(start_iter), root_rank=0, name="start_iter").item()

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            op=hvd.Adasum if args.use_adasum else hvd.Average,
            compression=compression,
        )  # device_dense='/cpu:0'

    if cfg.SOLVER.CHECKPOINT_BY_EPOCH:
        ckpt_period = cfg.SOLVER.CHECKPOINT_PERIOD * iters_per_epoch
    else:
        ckpt_period = cfg.SOLVER.CHECKPOINT_PERIOD
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, ckpt_period, max_iter=max_iter, max_to_keep=cfg.SOLVER.MAX_TO_KEEP
    )

    # build writers ==============================================
    tbx_event_writer = get_tbx_event_writer(cfg.OUTPUT_DIR, backup=not cfg.get("RESUME", False))
    tbx_writer = tbx_event_writer._writer  # NOTE: we want to write some non-scalar data
    writers = (
        [MyCommonMetricPrinter(max_iter), MyJSONWriter(osp.join(cfg.OUTPUT_DIR, "metrics.json")), tbx_event_writer]
        if comm.is_main_process()
        else []
    )
    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    logger.info("Starting training from iteration {}".format(start_iter))
    iter_time = None
    with EventStorage(start_iter) as storage:
        # for data, iteration in zip(data_loader, range(start_iter, max_iter)):
        for iteration in range(start_iter, max_iter):

            storage.iter = iteration
            epoch = iteration // dataset_len + 1
            t1 = time.time()
            if np.random.rand() < train_2_ratio:
                data = next(data_loader_2_iter)
            else:
                data = next(data_loader_iter)

            t2 = time.time()
            if iter_time is not None:
                storage.put_scalar("time", time.perf_counter() - iter_time)
            iter_time = time.perf_counter()
            # forward ============================================================

            batch = batch_data(cfg, data)
            with autocast(enabled=AMP_ON):
                t3 = time.time()
                if "sken" in cfg.filename:
                    out_dict, loss_dict = model(
                        batch["roi_img"],
                        # gt_xyz=batch.get("roi_xyz", None),
                        # gt_xyz_bin=batch.get("roi_xyz_bin", None),
                        gt_skeleton=batch["skeleton"],
                        gt_skeleton_mask=batch["skeleton_mask"],
                        skeleton_orig=batch["skeleton_orig"],
                        gt_kp_3d=batch["keypoints_3d_c_2d_position"],
                        gt_kp_pro=batch["key_probility"],
                        gt_skl_3d=batch["skl_3d_c_2d_position"],
                        gt_skl_pro=batch["skl_probility"],
                        gt_mask_trunc=batch["roi_mask_trunc"],
                        gt_mask_visib=batch["roi_mask_visib"],
                        gt_mask_obj=batch["roi_mask_obj"],
                        # gt_region=batch.get("roi_region", None),
                        # gt_allo_quat=batch.get("allo_quat", None),  # 没使用
                        # gt_ego_quat=batch.get("ego_quat", None),  # 没使用
                        gt_allo_rot6d=batch.get("allo_rot6d", None),  # 没使用
                        gt_ego_rot6d=batch.get("ego_rot6d", None),  # 没使用
                        gt_ego_rot=batch.get("ego_rot", None),
                        gt_trans=batch.get("trans", None),
                        gt_trans_ratio=batch["roi_trans_ratio"],
                        gt_points=batch.get("roi_points", None),
                        sym_infos=batch.get("sym_info", None),
                        roi_classes=batch["roi_cls"],
                        roi_cams=batch["roi_cam"],  # 以下的在计算pose_tran_pred_centroid_2_abs (pre_ego_rot+pred_trans)
                        roi_whs=batch["roi_wh"],
                        roi_centers=batch["roi_center"],
                        resize_ratios=batch["resize_ratio"],
                        roi_coord_2d=batch.get("roi_coord_2d", None),
                        roi_extents=batch.get("roi_extent", None),
                        do_loss=True,
                    )
                else:
                    out_dict, loss_dict = model(
                        batch["roi_img"],
                        gt_xyz=batch.get("roi_xyz", None),
                        gt_xyz_bin=batch.get("roi_xyz_bin", None),  # 没使用
                        gt_mask_trunc=batch["roi_mask_trunc"],
                        gt_mask_visib=batch["roi_mask_visib"],
                        gt_mask_obj=batch["roi_mask_obj"],
                        gt_region=batch.get("roi_region", None),
                        gt_allo_quat=batch.get("allo_quat", None),
                        gt_ego_quat=batch.get("ego_quat", None),
                        gt_allo_rot6d=batch.get("allo_rot6d", None),
                        gt_ego_rot6d=batch.get("ego_rot6d", None),
                        gt_ego_rot=batch.get("ego_rot", None),
                        gt_trans=batch.get("trans", None),
                        gt_trans_ratio=batch["roi_trans_ratio"],
                        gt_points=batch.get("roi_points", None),
                        sym_infos=batch.get("sym_info", None),
                        roi_classes=batch["roi_cls"],
                        roi_cams=batch["roi_cam"],  # 以下的在计算pose_tran_pred_centroid_2_abs (pre_ego_rot+pred_trans)
                        roi_whs=batch["roi_wh"],
                        roi_centers=batch["roi_center"],
                        resize_ratios=batch["resize_ratio"],
                        roi_coord_2d=batch.get("roi_coord_2d", None),
                        roi_extents=batch.get("roi_extent", None),
                        do_loss=True,
                    )

                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

            # print('pre_process: %f,  dataloader: %f,   forward pass: %f'%(t2-t1, t3-t2, time.time()-t3))
            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            if AMP_ON:
                grad_scaler.scale(losses).backward()

                # # Unscales the gradients of optimizer's assigned params in-place
                # grad_scaler.unscale_(optimizer)
                # # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                if comm._USE_HVD:
                    optimizer.synchronize()
                    with optimizer.skip_synchronize():
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                else:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
            else:
                losses.backward()
                optimizer.step()

            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0 and iteration != max_iter - 1:
                do_test(cfg, model, epoch=epoch, iteration=iteration)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % cfg.TRAIN.PRINT_FREQ == 0 or iteration == max_iter - 1 or iteration < 100
            ):
                for writer in writers:
                    writer.write()
                # visualize some images ========================================
                if cfg.TRAIN.VIS_IMG == True:
                    with torch.no_grad():
                        if "sken" in cfg.filename:
                            vis_i = 0
                            roi_img_vis = batch["roi_img"][vis_i].cpu().numpy()
                            #  roi_img_vis = denormalize_image(roi_img_vis, cfg).transpose(1, 2, 0).astype("uint8")  #后面有一步是将CHW转为HWC，所以这里就不继续转了
                            tbx_writer.add_image("input_image", roi_img_vis, iteration)

                            if cfg.MODEL.CDPN.PNP_NET.SKELETON_ATTENTION != "none" and cfg.MODEL.CDPN.PNP_NET.SKELETON_2D_DETECTION:
                                out_ske = out_dict["skeleton"].detach()
                                out_ske = get_out_mask(cfg, out_ske)
                                out_ske_vis = np.expand_dims(out_ske[vis_i, 0].cpu().numpy(), axis=0)
                                tbx_writer.add_image("out_ske", out_ske_vis, iteration)
                                gt_ske_vis = np.expand_dims(batch["skeleton_orig"][vis_i].detach().cpu().numpy(),axis=0)
                                tbx_writer.add_image("gt_ske", gt_ske_vis, iteration)
                                gt_skeleton_mask = np.expand_dims(batch["skeleton_mask"][vis_i,0].detach().cpu().numpy(),axis=0)
                                tbx_writer.add_image("gt_ske_mask", gt_skeleton_mask, iteration)



                            out_mask = out_dict["mask"].detach()
                            out_mask = get_out_mask(cfg, out_mask)
                            out_mask_vis = np.expand_dims(out_mask[vis_i, 0].cpu().numpy(), axis=0)
                            tbx_writer.add_image("out_mask", out_mask_vis, iteration)
                            gt_mask_vis = np.expand_dims(batch["roi_mask_trunc"][vis_i].detach().cpu().numpy(), axis=0)
                            tbx_writer.add_image("gt_mask", gt_mask_vis, iteration)

                            # out_kp_pro = out_dict["pre_ky_pr"].detach()
                            # out_kp_pro = get_out_mask(cfg, out_kp_pro)
                            # out_kp_pro_vis = np.expand_dims(out_kp_pro[vis_i, 0].cpu().numpy(), axis=0)
                            # tbx_writer.add_image("out_kp_pro", out_kp_pro_vis, iteration)
                            # gt_kp_pro_vis = np.expand_dims(batch["key_probility"][vis_i].detach().cpu().numpy(), axis=0)
                            # tbx_writer.add_image("gt_kp_pro", gt_kp_pro_vis, iteration)
                            #
                            # out_kp_3d = out_dict["pre_ky_3d"].detach()
                            # out_kp_3d = get_out_mask(cfg, out_kp_3d)
                            # out_kp_3d_vis = np.expand_dims(out_kp_3d[vis_i, 0].cpu().numpy(), axis=0)
                            # tbx_writer.add_image("out_kp_pro", out_kp_3d_vis, iteration)
                            # gt_kp_3d_vis = np.expand_dims(batch["keypoints_3d_c_2d_position"][vis_i].detach().cpu().numpy(), axis=0)
                            # tbx_writer.add_image("gt_kp_3d", gt_kp_3d_vis, iteration)

                            # out_coor_x = out_dict["coor_x"].detach()
                            # out_coor_y = out_dict["coor_y"].detach()
                            # out_coor_z = out_dict["coor_z"].detach()
                            # out_xyz = get_out_coor(cfg, out_coor_x, out_coor_y, out_coor_z)
                            #
                            # out_xyz_vis = out_xyz[vis_i].cpu().numpy().transpose(1, 2, 0)
                            # out_xyz_vis = get_emb_show(out_xyz_vis)
                            # tbx_writer.add_image("out_xyz", out_xyz_vis, iteration)
                            #
                            # gt_xyz_vis = batch["roi_xyz"][vis_i].cpu().numpy().transpose(1, 2, 0)
                            # gt_xyz_vis = get_emb_show(gt_xyz_vis)
                            # tbx_writer.add_image("gt_xyz", gt_xyz_vis, iteration)

                        periodic_checkpointer.step(iteration, epoch=epoch)
                        if "sken" not in cfg.filename and False:
                            vis_i = 0
                            roi_img_vis = batch["roi_img"][vis_i].cpu().numpy()
                            #  roi_img_vis = denormalize_image(roi_img_vis, cfg).transpose(1, 2, 0).astype("uint8")  #后面有一步是将CHW转为HWC，所以这里就不继续转了
                            tbx_writer.add_image("input_image", roi_img_vis, iteration)

                            out_coor_x = out_dict["coor_x"].detach()
                            out_coor_y = out_dict["coor_y"].detach()
                            out_coor_z = out_dict["coor_z"].detach()
                            out_xyz = get_out_coor(cfg, out_coor_x, out_coor_y, out_coor_z)

                            out_xyz_vis = out_xyz[vis_i].cpu().numpy().transpose(1, 2, 0)
                            out_xyz_vis = get_emb_show(out_xyz_vis)
                            tbx_writer.add_image("out_xyz", out_xyz_vis, iteration)

                            gt_xyz_vis = batch["roi_xyz"][vis_i].cpu().numpy().transpose(1, 2, 0)
                            gt_xyz_vis = get_emb_show(gt_xyz_vis)
                            tbx_writer.add_image("gt_xyz", gt_xyz_vis, iteration)

                            out_mask = out_dict["mask"].detach()
                            out_mask = get_out_mask(cfg, out_mask)
                            out_mask_vis = out_mask[vis_i, 0].cpu().numpy()
                            tbx_writer.add_image("out_mask", out_mask_vis, iteration)

                            gt_mask_vis = batch["roi_mask"][vis_i].detach().cpu().numpy()
                            tbx_writer.add_image("gt_mask", gt_mask_vis, iteration)
                        periodic_checkpointer.step(iteration, epoch=epoch)


