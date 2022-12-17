import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ref
from mmcv.runner import load_checkpoint
from detectron2.utils.events import get_event_storage
from core.utils.pose_utils import quat2mat_torch
from core.utils.rot_reps import ortho6d_to_mat_batch
from core.utils import quaternion_lf, lie_algebra
from core.utils.solver_utils import build_optimizer_with_params

from ..losses.coor_cross_entropy import CrossEntropyHeatmapLoss
from ..losses.l2_loss import L2Loss
from ..losses.pm_loss import PyPMLoss
from ..losses.rot_loss import angular_distance, rot_l2_loss
from .cdpn_rot_head_sken import RotWithRegionHead
from .cdpn_trans_head import TransHeadNet

# pnp net variants
from .conv_pnp_net_sken import ConvPnPNet
from .model_utils import compute_mean_re_te, get_mask_prob
from .point_pnp_net import PointPnPNet, SimplePointPnPNet
from .pose_from_pred import pose_from_pred
from .pose_from_pred_centroid_z import pose_from_pred_centroid_z
from .pose_from_pred_centroid_z_abs import pose_from_pred_centroid_z_abs
from .resnet_backbone import ResNetBackboneNet, resnet_spec

logger = logging.getLogger(__name__)


# backbone(ResNet34)-rot_head_net-pnp_net

class ske(nn.Module):
    def __init__(self, cfg, backbone, rot_head_net, trans_head_net=None, pnp_net=None):
        super().__init__()
        assert cfg.MODEL.CDPN.NAME == "SKEN", cfg.MODEL.CDPN.NAME
        self.backbone = backbone

        self.rot_head_net = rot_head_net
        self.pnp_net = pnp_net

        self.trans_head_net = trans_head_net

        self.cfg = cfg
        self.output_res = cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES
        self.concat = cfg.MODEL.CDPN.ROT_HEAD.ROT_CONCAT
        self.skeleton_out_dim, self.mask_out_dim, self.kp_pro_out_dim, self.kp_3d_out_dim = get_skeleton_mask_pro_3d_out_dim(cfg)

        # uncertainty multi-task loss weighting
        # https://github.com/Hui-Li/multi-task-learning-example-PyTorch/blob/master/multi-task-learning-example-PyTorch.ipynb
        # a = log(sigma^2)
        # L*exp(-a) + a  or  L*exp(-a) + log(1+exp(a))
        # self.log_vars = nn.Parameter(torch.tensor([0, 0], requires_grad=True, dtype=torch.float32).cuda())
        if cfg.MODEL.CDPN.USE_MTL:
            self.loss_names = [
                "mask",
                "coor_x",
                "coor_y",
                "coor_z",
                "coor_x_bin",
                "coor_y_bin",
                "coor_z_bin",
                "region",
                "PM_R",
                "PM_xy",
                "PM_z",
                "PM_xy_noP",
                "PM_z_noP",
                "PM_T",
                "PM_T_noP",
                "centroid",
                "z",
                "trans_xy",
                "trans_z",
                "trans_LPnP",
                "rot",
                "bind",
            ]
            for loss_name in self.loss_names:
                self.register_parameter(
                    f"log_var_{loss_name}", nn.Parameter(torch.tensor([0.0], requires_grad=True, dtype=torch.float32))
                )

    def forward(
        self,
        x,
        # gt_xyz=None,
        # gt_xyz_bin=None,  # 没使用
        gt_kp_pro=None,
        gt_kp_3d=None,
        gt_skl_3d=None,
        gt_skl_pro=None,
        gt_skeleton=None,  # skeleton_flux
        gt_skeleton_mask=None,  # dilmask
        skeleton_orig=None,  # single-pixel skeleton

        gt_mask_trunc=None,
        gt_mask_visib=None,
        gt_mask_obj=None,
        gt_region=None,
        gt_allo_quat=None, # 没使用
        gt_ego_quat=None, # 没使用
        gt_allo_rot6d=None, # 没使用
        gt_ego_rot6d=None,
        f=None,
        gt_ego_rot=None,
        gt_points=None,
        sym_infos=None,
        gt_trans=None,
        gt_trans_ratio=None,
        roi_classes=None,
        roi_coord_2d=None,
        roi_cams=None, # 以下的在计算pose_tran_pred_centroid_2_abs (pre_ego_rot+pred_trans)
        roi_centers=None,
        roi_whs=None,
        roi_extents=None,
        resize_ratios=None,
        do_loss=False,
    ):
        cfg = self.cfg
        output_res = self.output_res
        r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
        t_head_cfg = cfg.MODEL.CDPN.TRANS_HEAD
        pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET

        # x.shape [bs, 3, 256, 256]
        if self.concat:  #False
            features, x_f64, x_f32, x_f16 = self.backbone(x)  # features.shape [bs, 2048, 8, 8]
            # joints.shape [bs, 1152, 64, 64]
            pre_skeleton, skeleton_flux, skeleton_single_pxiel,  mask,  pre_kp_pro, pre_kp_3d, out_skl_3d, out_skl_pro = self.rot_head_net(features, x_f64, x_f32, x_f16)
            # mask, coor_x, coor_y, coor_z, region = self.rot_head_net(features, x_f64, x_f32, x_f16)
        else:
            features = self.backbone(x)  # features.shape [bs, 2048, 8, 8]
            # joints.shape [bs, 1152, 64, 64]
            pre_skeleton, skeleton_flux, skeleton_single_pxiel, mask,  pre_kp_pro, pre_kp_3d, out_skl_3d, out_skl_pro = self.rot_head_net(features)
            # mask, coor_x, coor_y, coor_z, region = self.rot_head_net(features)

        # TODO: remove this trans_head_net
        # trans = self.trans_head_net(features)

        device = x.device
        bs = x.shape[0]
        num_classes = r_head_cfg.NUM_CLASSES
        norm_scale = cfg.NORM_SCALE

        out_res = cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES

        # if r_head_cfg.ROT_CLASS_AWARE:  # False
        #     assert roi_classes is not None
        #     coor_x = coor_x.view(bs, num_classes, self.r_out_dim // 3, out_res, out_res)  #shape的变化
        #     coor_x = coor_x[torch.arange(bs).to(device), roi_classes]
        #     coor_y = coor_y.view(bs, num_classes, self.r_out_dim // 3, out_res, out_res)
        #     coor_y = coor_y[torch.arange(bs).to(device), roi_classes]
        #     coor_z = coor_z.view(bs, num_classes, self.r_out_dim // 3, out_res, out_res)
        #     coor_z = coor_z[torch.arange(bs).to(device), roi_classes]

        if r_head_cfg.SKELETON_CLASS_AWARE:  # False
            assert roi_classes is not None
            pre_skeleton = pre_skeleton.view(bs, num_classes, self.pre_skeleton_out_dim, out_res, out_res)
            pre_skeleton = pre_skeleton[torch.arange(bs).to(device), roi_classes]

        if r_head_cfg.MASK_CLASS_AWARE:  # False
            assert roi_classes is not None
            mask = mask.view(bs, num_classes, self.mask_out_dim, out_res, out_res)
            mask = mask[torch.arange(bs).to(device), roi_classes]

        # if r_head_cfg.REGION_CLASS_AWARE: # False
        #     assert roi_classes is not None
        #     region = region.view(bs, num_classes, self.region_out_dim, out_res, out_res)
        #     region = region[torch.arange(bs).to(device), roi_classes]

        # -----------------------------------------------
        # get rot and trans from pnp_net
        # NOTE: use softmax for bins (the last dim is bg)
        # if coor_x.shape[1] > 1 and coor_y.shape[1] > 1 and coor_z.shape[1] > 1:
        #     coor_x_softmax = F.softmax(coor_x[:, :-1, :, :], dim=1)
        #     coor_y_softmax = F.softmax(coor_y[:, :-1, :, :], dim=1)
        #     coor_z_softmax = F.softmax(coor_z[:, :-1, :, :], dim=1)
        #     coor_feat = torch.cat([coor_x_softmax, coor_y_softmax, coor_z_softmax], dim=1)
        # else:
        #     coor_feat = torch.cat([coor_x, coor_y, coor_z], dim=1)  # BCHW

        if pnp_net_cfg.KP_3D_CA and cfg.SKELETON=='2D':
            bs, c_3d, w, h = pre_kp_3d.size()
            bs, c_pro, w, h = pre_kp_pro.size()
            # pre_kp_pro_to_pnp = torch.empty(pre_kp_pro.size())
            pre_kp_pro_to_pnp = pre_kp_pro.clone()
            pre_kp_3d_to_pnp = pre_kp_3d.clone()
            pre_kp_pro_to_pnp[pre_kp_pro < torch.exp(torch.tensor((-1.0) * (16)))]=0.0
            pre_kp_pro_to_pnp[pre_kp_pro >= torch.exp((torch.tensor((-1.0) * (16))))] = 1.0
            if c_3d != 3*c_pro:
                raise ValueError("这里有错，检查一下，kp_3d采用的是xxxxxyyyyyzzzzz的形式")
            # if cfg.MODEL.CDPN.ROT_HEAD.KP_PRESENT == 0:
            #     for i_c in range(c_pro):
            #         pre_kp_3d_to_pnp[:, i_c, :, :] = pre_kp_3d[:, i_c, :, :] * pre_kp_pro[:, i_c, :, :]
            #         pre_kp_3d_to_pnp[:, i_c + c_pro, :, :] = pre_kp_3d[:, i_c + c_pro, :, :] * pre_kp_pro[:, i_c, :, :]
            #         pre_kp_3d_to_pnp[:, i_c + 2 * c_pro, :, :] = pre_kp_3d[:, i_c + 2 * c_pro, :, :] * pre_kp_pro[:, i_c, :, :]
            # else:
            #     raise ValueError("这里有错，检查一下，kp_3d采用的是xxxxxyyyyyzzzzz的形式")
            if pnp_net_cfg.WITH_2D_COORD:  # true
                assert roi_coord_2d is not None
                # coor_feat = torch.cat([ pre_kp_3d, roi_coord_2d], dim=1)
                # coor_feat = torch.cat([pre_kp_pro, roi_coord_2d], dim=1)
                coor_feat = torch.cat([roi_coord_2d], dim=1)

        elif cfg.SKELETON=='2D':
            if pnp_net_cfg.WITH_2D_COORD:  # true
                assert roi_coord_2d is not None
                coor_feat = torch.cat([pre_kp_pro, pre_kp_3d, roi_coord_2d], dim=1)
        elif cfg.SKELETON=='3D':
            if pnp_net_cfg.WITH_2D_COORD:  # true
                assert roi_coord_2d is not None
                coor_feat = torch.cat([out_skl_pro, out_skl_3d, roi_coord_2d], dim=1)

            # divce = pre_kp_3d.device
            # skls = torch.empty(bs, 1, w, h)


        # # NOTE: for region, the 1st dim is bg
        # region_softmax = F.softmax(region[:, 1:, :, :], dim=1)

        # TODO:将flux的skeleton转为points,
        #  pre_skeleton是rot_head直接输出的，
        #  skeleton_flux是pre_skeleton上采样之后用于计算flux_loss的，用于后处理
        #  skeleton_single_pxiel是将skeleton_flux经过三个卷积层之后，将flux转为point，转为end-to-end,用于计算single-pixel-sleleton_loss的，
        #  ske_kp是skeleton_fluxs由flux[2通道]转为points[1通道]，ske_kp的值也是概率[0,1]
        #  ske_kp_d是ske_kp下采样之后的结果。
        #  gt_skeleton,  # skeleton_flux
        #  gt_skeleton_mask,  # dilmask
        #  skeleton_orig,  # single-pixel skeleton
        #  skeleton_atten 为用于融合的最终骨架
        # plt.imshow(skeleton_atten[0, 0, :, :].cpu().detach().numpy())

        #TODO:skeleton_atten 和 mask_atten的用处不大
        skeleton_atten = None
        # if pnp_net_cfg.SKELETON_ATTENTION != "none":  # none
        #     skeleton_atten = get_mask_prob(cfg, pre_skeleton)
        if pnp_net_cfg.SKELETON_2D_DETECTION:
            if pnp_net_cfg.SKELETON_ATTENTION != "none":  # none
                if r_head_cfg.SKE_FLUX2POINT == 'postpro':  # TODO:后处理的方法将flux转为point
                    #
                    ske_kp = ref.air_flight_full.flux_to_point(skeleton_flux, threshold=cfg.MODEL.CDPN.ROT_HEAD.SKE_THR,
                                                               dks=cfg.MODEL.CDPN.ROT_HEAD.SKE_DKS,
                                                               eks=cfg.MODEL.CDPN.ROT_HEAD.SKE_EKS)
                    pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                    downsample = lambda x: F.interpolate(x,
                                                         scale_factor=1 / cfg.MODEL.CDPN.ROT_HEAD.SKE_OUTPUT_RES_SCALE,
                                                         mode='bilinear', align_corners=False)
                    ske_kp_d = ske_kp
                    for times_d in range(int(pow(cfg.MODEL.CDPN.ROT_HEAD.SKE_OUTPUT_RES_SCALE, 0.5))):
                        ske_kp_d = pool2(ske_kp_d)
                elif r_head_cfg.SKE_FLUX2POINT == 'end2end':  # TODO:end-to-end的方法将flux转为point
                    pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                    downsample = lambda x: F.interpolate(x,
                                                         scale_factor=1 / cfg.MODEL.CDPN.ROT_HEAD.SKE_OUTPUT_RES_SCALE,
                                                         mode='bilinear', align_corners=False)
                    ske_kp_d = skeleton_single_pxiel.clone()
                    ske_kp_d[ske_kp_d < cfg.MODEL.CDPN.ROT_HEAD.SKE_THR] = 0.0
                    ske_kp_d[ske_kp_d >= cfg.MODEL.CDPN.ROT_HEAD.SKE_THR] = 1.0
                    skeleton_atten = ske_kp_d.clone()
                    skeleton_atten = downsample(skeleton_atten)
                    # for times_d in range(int(pow(cfg.MODEL.CDPN.ROT_HEAD.SKE_OUTPUT_RES_SCALE, 0.5))):
                    #     skeleton_atten = pool2(skeleton_atten)
                else:  # 不添加skeleton 错误
                    raise RuntimeError("ske is no postpro or end2end")


        mask_atten = None
        if pnp_net_cfg.MASK_ATTENTION != "none":  # none 计算每个像素的概率
            mask_atten = get_mask_prob(cfg, mask)

        # region_atten = None
        # if pnp_net_cfg.REGION_ATTENTION:  # True
        #     region_atten = region_softmax



        pred_rot_, pred_t_ = self.pnp_net(
            coor_feat, extents=roi_extents, skeleton_attention=pre_skeleton, mask_attention=mask_atten
        )
        if pnp_net_cfg.R_ONLY:  #  False  override trans pred False
            pred_t_ = self.trans_head_net(features)

        # convert pred_rot to rot mat -------------------------
        rot_type = pnp_net_cfg.ROT_TYPE  # 'allo_rot6d'
        if rot_type in ["ego_quat", "allo_quat"]:
            pred_rot_m = quat2mat_torch(pred_rot_)
        elif rot_type in ["ego_log_quat", "allo_log_quat"]:
            pred_rot_m = quat2mat_torch(quaternion_lf.qexp(pred_rot_))
        elif rot_type in ["ego_lie_vec", "allo_lie_vec"]:
            pred_rot_m = lie_algebra.lie_vec_to_rot(pred_rot_)
        elif rot_type in ["ego_rot6d", "allo_rot6d"]:
            pred_rot_m = ortho6d_to_mat_batch(pred_rot_)
        else:
            raise RuntimeError(f"Wrong pred_rot_ dim: {pred_rot_.shape}")
        # convert pred_rot_m and pred_t to ego pose -----------------------------
        if pnp_net_cfg.TRANS_TYPE == "centroid_z": # centroid_z
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
                roi_cams=roi_cams,  
                roi_centers=roi_centers,  
                resize_ratios=resize_ratios,
                roi_whs=roi_whs,
                eps=1e-4,
                is_allo="allo" in pnp_net_cfg.ROT_TYPE, # 'allo_rot6d'
                z_type=pnp_net_cfg.Z_TYPE,  # 'REL'
                # is_train=True
                is_train=do_loss,  # TODO: sometimes we need it to be differentiable during test
            )
        elif pnp_net_cfg.TRANS_TYPE == "centroid_z_abs":
            # abs 2d obj center and abs z
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z_abs(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
                roi_cams=roi_cams,
                eps=1e-4,
                is_allo="allo" in pnp_net_cfg.ROT_TYPE,
                # is_train=True
                is_train=do_loss,  # TODO: sometimes we need it to be differentiable during test
            )
        elif pnp_net_cfg.TRANS_TYPE == "trans": 
            # TODO: maybe denormalize trans
            pred_ego_rot, pred_trans = pose_from_pred(
                pred_rot_m, pred_t_, eps=1e-4, is_allo="allo" in pnp_net_cfg.ROT_TYPE, is_train=do_loss
            )
        else:
            raise ValueError(f"Unknown pnp_net trans type: {pnp_net_cfg.TRANS_TYPE}")

        if not do_loss:  # test
            out_dict = {"rot": pred_ego_rot, "trans": pred_trans}
            if cfg.TEST.USE_PNP:
                # TODO: move the pnp/ransac inside forward
                if pnp_net_cfg.SKELETON_ATTENTION != "none":
                    out_dict.update({"skeleton": skeleton_atten, "mask": mask, "pre_ky_pr": pre_kp_pro, "pre_ky_ed": pre_kp_3d})
                else:
                    out_dict.update({ "mask": mask, "pre_ky_pr": pre_kp_pro, "pre_ky_ed": pre_kp_3d})
        else:
            #自己添加的行
            out_dict = {"rot": pred_ego_rot, "trans": pred_trans}
            if pnp_net_cfg.SKELETON_ATTENTION != "none" and pnp_net_cfg.SKELETON_2D_DETECTION:
                out_dict.update({"rot": pred_ego_rot, "trans": pred_trans, "skeleton": skeleton_single_pxiel, "mask": mask, "pre_ky_pr": pre_kp_pro, "pre_ky_3d": pre_kp_3d})
            elif cfg.MODEL.CDPN.PNP_NET.SKELETON_3D_ATTENTION!='none':
                out_dict.update(
                    {"rot": pred_ego_rot, "trans": pred_trans, "mask": mask, "pre_skl_3d_pr": out_skl_pro, "pre_skl_3d": out_skl_3d})
            else:
                out_dict.update({"rot": pred_ego_rot, "trans": pred_trans, "mask": mask, "pre_ky_pr": pre_kp_pro, "pre_ky_3d": pre_kp_3d})
            assert (
                # (gt_xyz is not None)
                (gt_kp_3d is not None)
                and (gt_kp_pro is not None)
                and (gt_skeleton is not None)
                and (gt_trans is not None)
                and (gt_trans_ratio is not None)
                and (gt_mask_obj is not None)
                and (gt_mask_visib is not None)
                and (gt_mask_trunc is not None)
            )
            mean_re, mean_te = compute_mean_re_te(pred_trans * norm_scale, pred_rot_m, gt_trans * norm_scale, gt_ego_rot)
            vis_dict = {
                "vis/error_R": mean_re,
                "vis/error_t": mean_te *100,  # cm
                "vis/error_tx": np.abs(pred_trans[0, 0].detach().item() - gt_trans[0, 0].detach().item()) * 100,  # cm
                "vis/error_ty": np.abs(pred_trans[0, 1].detach().item() - gt_trans[0, 1].detach().item()) * 100,  # cm
                "vis/error_tz": np.abs(pred_trans[0, 2].detach().item() - gt_trans[0, 2].detach().item()) * 100,  # cm
                "vis/tx_pred": pred_trans[0, 0].detach().item(),
                "vis/ty_pred": pred_trans[0, 1].detach().item(),
                "vis/tz_pred": pred_trans[0, 2].detach().item(),
                "vis/tx_net": pred_t_[0, 0].detach().item(),
                "vis/ty_net": pred_t_[0, 1].detach().item(),
                "vis/tz_net": pred_t_[0, 2].detach().item(),
                "vis/tx_gt": gt_trans[0, 0].detach().item(),
                "vis/ty_gt": gt_trans[0, 1].detach().item(),
                "vis/tz_gt": gt_trans[0, 2].detach().item(),
                "vis/tx_rel_gt": gt_trans_ratio[0, 0].detach().item(),
                "vis/ty_rel_gt": gt_trans_ratio[0, 1].detach().item(),
                "vis/tz_rel_gt": gt_trans_ratio[0, 2].detach().item(),
            }

            loss_dict = self.ske_loss(
                cfg=self.cfg,
                out_mask=mask,
                gt_mask_trunc=gt_mask_trunc,
                gt_mask_visib=gt_mask_visib,
                gt_mask_obj=gt_mask_obj,
                # out_x=coor_x,
                # out_y=coor_y,
                # out_z=coor_z,
                # gt_xyz=gt_xyz,
                # gt_xyz_bin=gt_xyz_bin,

                out_kp_pro=pre_kp_pro,
                out_kp_3d=pre_kp_3d,
                out_skeleton=pre_skeleton,
                out_skeleton_E=skeleton_single_pxiel,
                out_skeleton_P=skeleton_flux,
                out_skl_3d=out_skl_3d,
                out_skl_3d_pro=out_skl_pro,
                gt_skeleton_P=gt_skeleton,
                gt_skeleton_mask_P=gt_skeleton_mask,
                gt_skeleton_E=skeleton_orig,
                gt_kp_pro=gt_kp_pro,
                gt_kp_3d=gt_kp_3d,
                gt_skl_3d=gt_skl_3d,
                gt_skl_3d_pro=gt_skl_pro,
                bs=bs,


                # out_region=region,
                # gt_region=gt_region,
                out_trans=pred_trans,
                gt_trans=gt_trans,
                out_rot=pred_ego_rot, 
                gt_rot=gt_ego_rot,
                out_centroid=pred_t_[:, :2],  # TODO: get these from trans head
                out_trans_z=pred_t_[:, 2],
                gt_trans_ratio=gt_trans_ratio,
                gt_points=gt_points,
                sym_infos=sym_infos,
                extents=roi_extents,
                # roi_classes=roi_classes,
            )

            if cfg.MODEL.CDPN.USE_MTL:
                for _name in self.loss_names:
                    if f"loss_{_name}" in loss_dict:
                        vis_dict[f"vis_lw/{_name}"] = torch.exp(-getattr(self, f"log_var_{_name}")).detach().item()
            for _k, _v in vis_dict.items():
                if "vis/" in _k or "vis_lw/" in _k:
                    if isinstance(_v, torch.Tensor):
                        _v = _v.item()
                    vis_dict[_k] = _v
            storage = get_event_storage()
            storage.put_scalars(**vis_dict)

            return out_dict, loss_dict
        return out_dict

    def ske_loss(
        self,
        cfg,
        out_mask,
        gt_mask_trunc,
        gt_mask_visib,
        gt_mask_obj,
        # out_x,
        # out_y,
        # out_z,
        # gt_xyz,
        # # gt_xyz_bin,
        # out_region,
        # gt_region,

        out_kp_pro,
        out_kp_3d,
        out_skl_3d_pro,
        out_skl_3d,
        out_skeleton,  # rot_head的输出
        out_skeleton_E,  # end2end的输出,singel_pixel-skeleton
        out_skeleton_P,  # postpro的输出,flux-skeleton
        gt_skeleton_P,  # flux
        gt_skeleton_mask_P,  # dilmask
        gt_skeleton_E,
        gt_kp_pro,
        gt_kp_3d,
        gt_skl_3d_pro,
        gt_skl_3d,
        bs,

        out_rot=None,
        gt_rot=None,
        out_trans=None,
        gt_trans=None,
        out_centroid=None,
        out_trans_z=None,
        gt_trans_ratio=None,
        gt_points=None,
        sym_infos=None,
        extents=None,

    ):
        r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
        t_head_cfg = cfg.MODEL.CDPN.TRANS_HEAD
        pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET

        loss_dict = {}

        gt_masks = {"trunc": gt_mask_trunc, "visib": gt_mask_visib, "obj": gt_mask_obj}

        # # rot xyz loss ----------------------------------
        # if not r_head_cfg.FREEZE: #进
        #     xyz_loss_type = r_head_cfg.XYZ_LOSS_TYPE
        #     gt_mask_xyz = gt_masks[r_head_cfg.XYZ_LOSS_MASK_GT] # visib
        #     if xyz_loss_type == "L1": # L1
        #         loss_func = nn.L1Loss(reduction="sum")
        #         loss_dict["loss_coor_x"] = loss_func(
        #             out_x * gt_mask_xyz[:, None], gt_xyz[:, 0:1] * gt_mask_xyz[:, None]
        #         ) / gt_mask_xyz.sum().float().clamp(min=1.0)
        #         loss_dict["loss_coor_y"] = loss_func(
        #             out_y * gt_mask_xyz[:, None], gt_xyz[:, 1:2] * gt_mask_xyz[:, None]
        #         ) / gt_mask_xyz.sum().float().clamp(min=1.0)
        #         loss_dict["loss_coor_z"] = loss_func(
        #             out_z * gt_mask_xyz[:, None], gt_xyz[:, 2:3] * gt_mask_xyz[:, None]
        #         ) / gt_mask_xyz.sum().float().clamp(min=1.0)
        #     elif xyz_loss_type == "CE_coor":
        #         gt_xyz_bin = gt_xyz_bin.long()
        #         loss_func = CrossEntropyHeatmapLoss(reduction="sum", weight=None)  # r_head_cfg.XYZ_BIN+1
        #         loss_dict["loss_coor_x"] = loss_func(
        #             out_x * gt_mask_xyz[:, None], gt_xyz_bin[:, 0] * gt_mask_xyz.long()
        #         ) / gt_mask_xyz.sum().float().clamp(min=1.0)
        #         loss_dict["loss_coor_y"] = loss_func(
        #             out_y * gt_mask_xyz[:, None], gt_xyz_bin[:, 1] * gt_mask_xyz.long()
        #         ) / gt_mask_xyz.sum().float().clamp(min=1.0)
        #         loss_dict["loss_coor_z"] = loss_func(
        #             out_z * gt_mask_xyz[:, None], gt_xyz_bin[:, 2] * gt_mask_xyz.long()
        #         ) / gt_mask_xyz.sum().float().clamp(min=1.0)
        #     else:
        #         raise NotImplementedError(f"unknown xyz loss type: {xyz_loss_type}")
        #     loss_dict["loss_coor_x"] *= r_head_cfg.XYZ_LW  #XYZ_LW=1
        #     loss_dict["loss_coor_y"] *= r_head_cfg.XYZ_LW
        #     loss_dict["loss_coor_z"] *= r_head_cfg.XYZ_LW

        # skeleton loss-------------------------------
        if pnp_net_cfg.SKELETON_2D_DETECTION and not r_head_cfg.FREEZE_ske and cfg.SKELETON=='2D':  # 运行
            # flux_loss
            skeleton_loss_type = r_head_cfg.SKELETON_LOSS_TYPE
            # gt_mask = gt_masks[r_head_cfg.MASK_LOSS_GT] # 'trunc'
            regionPos = (gt_skeleton_mask_P > 0) + 0
            regionNeg = (gt_skeleton_mask_P == 0) + 0
            bs, c, w, h = regionPos.shape

            sumPos = torch.sum(torch.sum(regionPos, axis=3), axis=2).view((bs, -1))
            sumNeg = torch.sum(torch.sum(regionNeg, axis=3), axis=2).view((bs, -1))
            weightPos = torch.zeros(bs, 2, w, h)
            weightNeg = torch.zeros(bs, 2, w, h)
            device = gt_skeleton_mask_P.device
            for bs_id in range(bs):
                weightPos[bs_id, 0, :, :] = sumNeg[bs_id, 0] / (sumPos[bs_id, 0] + sumNeg[bs_id, 0]) * regionPos[bs_id, :, :]
                weightPos[bs_id, 1, :, :] = sumNeg[bs_id, 0] / (sumPos[bs_id, 0] + sumNeg[bs_id, 0]) * regionPos[bs_id, :, :]
                weightNeg[bs_id, 0, :, :] = sumPos[bs_id, 0] / (sumPos[bs_id, 0] + sumNeg[bs_id, 0]) * regionNeg[bs_id, :, :]
                weightNeg[bs_id, 1, :, :] = sumPos[bs_id, 0] / (sumPos[bs_id, 0] + sumNeg[bs_id, 0]) * regionNeg[bs_id, :, :]
            weightske = weightPos + weightNeg
            weightske = weightske.to(device)
            sumske = torch.sum(torch.sum(weightske, dim=3), dim=2).view(bs, 2, 1, 1)

            if skeleton_loss_type == "L1":
                loss_dict["loss_skeleton_flux"] = nn.L1Loss(reduction="mean")(out_skeleton_P[:, 0, :, :] * weightske[:, 0, :, :],
                                                                              gt_skeleton_P[:, 0, :, :] * weightske[:, 0, :, :])
                loss_dict["loss_skeleton_flux"] += nn.L1Loss(reduction="mean")(out_skeleton_P[:, 1, :, :] * weightske[:, 1, :, :],
                                                                               gt_skeleton_P[:, 1, :, :] * weightske[:, 1, :, :])
                # loss_dict["loss_skeleton"] = nn.L1Loss(reduction="mean")(loss_skeleton[:, 0, :, :]*weightske/sumske, gt_skeleton* weightske/sumske)
            elif skeleton_loss_type == "MSE":
                loss_dict["loss_skeleton_flux"] = nn.MSELoss(reduction="mean")(
                    out_skeleton_P[:, 0, :, :] * pow(weightske[:, 0, :, :] / 2 , 0.5),
                    gt_skeleton_P[:, 0, :, :] * pow(weightske[:, 0, :, :] / 2 , 0.5))
                loss_dict["loss_skeleton_flux"] += nn.MSELoss(reduction="mean")(
                    out_skeleton_P[:, 1, :, :] * pow(weightske[:, 1, :, :] / 2 , 0.5),
                    gt_skeleton_P[:, 1, :, :] * pow(weightske[:, 1, :, :] / 2 , 0.5))
                # loss_dict["loss_skeleton"] = nn.MSELoss(reduction="mean")(
                #     loss_skeleton[:, 0, :, :] * pow(weightske, 0.5) / pow(sumske, 0.5), gt_skeleton * pow(weightske, 0.5) / pow(sumske, 0.5))
            elif skeleton_loss_type == "BCE":
                loss_dict["loss_skeleton_flux"] = nn.BCEWithLogitsLoss(reduction="mean")(out_skeleton_P[:, 0, :, :], gt_skeleton_P)
            elif skeleton_loss_type == "CE":
                loss_dict["loss_skeleton_flux"] = nn.CrossEntropyLoss(reduction="mean")(out_skeleton_P, gt_skeleton_P.long())
            else:
                raise NotImplementedError(f"unknown skeleton loss type: {skeleton_loss_type}")
            loss_dict["loss_skeleton_flux"] *= r_head_cfg.SKELETON_LW

            if r_head_cfg.SKE_FLUX2POINT == 'end2end':
                skeleton_loss_type = r_head_cfg.SKELETON_LOSS_TYPE
                # gt_mask = gt_masks[r_head_cfg.MASK_LOSS_GT] # 'trunc'
                regionPos = (gt_skeleton_E > 0) + 0
                regionNeg = (gt_skeleton_E== 0) + 0
                bs, w, h = regionPos.shape

                sumPos = torch.sum(torch.sum(regionPos, axis=2), axis=1).view((bs, -1))
                sumNeg = torch.sum(torch.sum(regionNeg, axis=2), axis=1).view((bs, -1))
                weightPos = torch.zeros(regionPos.shape)
                weightNeg = torch.zeros(regionNeg.shape)
                device = gt_skeleton_E.device
                for bs_id in range(bs):
                    weightPos[bs_id, :, :] = sumNeg[bs_id, 0] / (sumPos[bs_id, 0] + sumNeg[bs_id, 0]) * regionPos[bs_id, :, :]
                    weightNeg[bs_id, :, :] = sumPos[bs_id, 0] / (sumPos[bs_id, 0] + sumNeg[bs_id, 0]) * regionNeg[bs_id, :, :]
                weightske = weightPos + weightNeg
                weightske = weightske.to(device)
                sumske = torch.sum(torch.sum(weightske, dim=2), dim=1).view(bs, 1, 1)
                # plt.imshow(out_skeleton_E[0, 0, :, :].cpu().detach().numpy())
                # plt.show()

                if skeleton_loss_type == "L1":
                    loss_dict["loss_skeleton_single"] = nn.L1Loss(reduction="mean")(out_skeleton_E[:, 0, :, :] * weightske ,
                                                                                    gt_skeleton_E[:, :, :] * weightske )
                elif skeleton_loss_type == "MSE":
                    loss_dict["loss_skeleton_single"] = nn.MSELoss(reduction="mean")(out_skeleton_E[:, 0, :, :] * pow(weightske/ 2 , 0.5),
                                                                                     gt_skeleton_E[:, :, :] * pow(weightske/ 2 , 0.5))
                elif skeleton_loss_type == "BCE":
                    loss_dict["loss_skeleton_single"] = nn.BCEWithLogitsLoss(reduction="mean")(out_skeleton_E[:, 0, :, :],
                                                                                               gt_skeleton_E[:, :, :] )
                elif skeleton_loss_type == "CE":
                    loss_dict["loss_skeleton_single"] = nn.CrossEntropyLoss(reduction="mean")(out_skeleton_E, gt_skeleton_E.long())
                else:
                    raise NotImplementedError(f"unknown skeleton loss type: {skeleton_loss_type}")
                loss_dict["loss_skeleton_single"] *= r_head_cfg.SKELETON_LW

        # mask loss----------------------------------
        if not r_head_cfg.FREEZE_mask: # 运行
            mask_loss_type = r_head_cfg.MASK_LOSS_TYPE
            gt_mask = gt_masks[r_head_cfg.MASK_LOSS_GT] # 'trunc'
            regionPos_mask= (gt_mask > 0) + 0
            regionNeg_mask = (gt_mask == 0) + 0
            sumPos_mask = torch.sum(torch.sum(regionPos_mask, axis=2), axis=1).view((bs,-1))
            sumNeg_mask = torch.sum(torch.sum(regionNeg_mask, axis=2), axis=1).view((bs,-1))
            weightPos_mask = torch.zeros(regionPos_mask.shape)
            weightNeg_mask = torch.zeros(regionNeg_mask.shape)
            device_mask= gt_mask.device
            for bs_id_m in range(bs):
                weightPos_mask[bs_id_m, :, :] = sumNeg_mask[bs_id_m,0] / (sumPos_mask[bs_id_m,0] + sumNeg_mask[bs_id_m,0]) * regionPos_mask[bs_id_m, :, :]
                weightNeg_mask[bs_id_m, :, :] = sumPos_mask[bs_id_m,0] / (sumPos_mask[bs_id_m,0] + sumNeg_mask[bs_id_m,0]) * regionNeg_mask[bs_id_m, :, :]
            weight_mask = weightPos_mask + weightNeg_mask
            weight_mask =weight_mask.to(device_mask)
            sum_mask = torch.sum(torch.sum(weight_mask, dim=2), dim=1).view(bs, 1, 1)
            if mask_loss_type == "L1":
                loss_dict["loss_mask"] = nn.L1Loss(reduction="mean")(out_mask[:, 0, :, :] * weight_mask, gt_mask * weight_mask)
                # loss_dict["loss_mask"] = nn.L1Loss(reduction="mean")(out_mask[:, 0, :, :]*weight_mask/sum_mask, gt_mask*weight_mask/sum_mask)
            elif mask_loss_type == "MSE":
                loss_dict["loss_mask"] = nn.MSELoss(reduction="mean")(
                    out_mask[:, 0, :, :] * pow(weight_mask, 0.5) / 2, gt_mask * pow(weight_mask, 0.5) / 2)
                # loss_dict["loss_mask"] = nn.MSELoss(reduction="mean")(
                #     out_mask[:, 0, :, :] * pow(weight_mask, 0.5)/pow(sum_mask,0.5), gt_mask * pow(weight_mask, 0.5)/pow(sum_mask, 0.5))
            elif mask_loss_type == "BCE":
                loss_dict["loss_mask"] = nn.BCEWithLogitsLoss(reduction="mean")(out_mask[:, 0, :, :], gt_mask)
            elif mask_loss_type == "CE":
                loss_dict["loss_mask"] = nn.CrossEntropyLoss(reduction="mean")(out_mask, gt_mask.long())
            else:
                raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")
            loss_dict["loss_mask"] *= r_head_cfg.MASK_LW

        # kp_pro loss --------------------
        if not r_head_cfg.FREEZE_kp_pro and cfg.SKELETON=='2D': # 运行
            kp_pro_loss_type = r_head_cfg.KP_PRO_LOSS_TYPE
            gt_mask = gt_masks[r_head_cfg.KP_PRO_LOSS_MASK_GT] # visib
            if kp_pro_loss_type == "CE":
                loss_func = nn.CrossEntropyLoss(reduction="sum", weight=None)
                loss_dict["loss_kp_pro"] = loss_func(
                    out_kp_pro * gt_mask[:, None], gt_kp_pro * gt_mask
                ) / gt_mask.sum().float().clamp(min=1.0)
            elif kp_pro_loss_type == "MSE":
                loss_func = nn.MSELoss(reduction="sum")
                loss_dict["loss_kp_pro"] = loss_func(
                    out_kp_pro[:, 0, :, :] * gt_mask, gt_kp_pro[:, 0, :, :] * gt_mask
                ) / gt_mask.sum().float().clamp(min=1.0)
                for out_kp_pro_id in range(0, out_kp_pro.size(1)):
                    loss_dict["loss_kp_pro"] += loss_func(
                    out_kp_pro[:, out_kp_pro_id, :, :] * gt_mask, gt_kp_pro[:, out_kp_pro_id, :, :] * gt_mask
                ) / gt_mask.sum().float().clamp(min=1.0)
            else:
                raise NotImplementedError(f"unknown kp pro loss type: {kp_pro_loss_type}")
            loss_dict["loss_kp_pro"] *= r_head_cfg.KP_PRO_LW

        # kp_3d loss --------------------
        if not r_head_cfg.FREEZE_kp_3d and cfg.SKELETON=='2D': # 运行
            kp_3d_loss_type = r_head_cfg.KP_3D_LOSS_TYPE
            gt_mask = gt_masks[r_head_cfg.KP_3D_LOSS_MASK_GT] # visib
            if kp_3d_loss_type == "CE":
                loss_func = nn.CrossEntropyLoss(reduction="sum", weight=None)
                loss_dict["loss_kp_3d"] = loss_func(
                    out_kp_3d * gt_mask[:, None], gt_kp_3d * gt_mask
                ) / gt_mask.sum().float().clamp(min=1.0)
            elif kp_3d_loss_type == "MSE":
                loss_func = nn.MSELoss(reduction="sum")
                loss_dict["loss_kp_3d"] = loss_func(
                    out_kp_3d[:, 0, :, :] * gt_mask, gt_kp_3d[:, 0, :, :] * gt_mask
                ) / gt_mask.sum().float().clamp(min=1.0)
                for out_kp_3d_id in range(0, out_kp_3d.size(1)):
                    loss_dict["loss_kp_3d"] += loss_func(
                    out_kp_3d[:, out_kp_3d_id, :, :] * gt_mask, gt_kp_3d[:, out_kp_3d_id, :, :] * gt_mask
                ) / gt_mask.sum().float().clamp(min=1.0)
            else:
                raise NotImplementedError(f"unknown kp 3d loss type: {kp_3d_loss_type}")
            loss_dict["loss_kp_3d"] *= r_head_cfg.KP_3D_LW

        # skl_3d_pro loss --------------------
        if not r_head_cfg.FREEZE_skl_3d_pro and cfg.SKELETON=='3D': # 运行
            skl_3d_pro_loss_type = r_head_cfg.SKL_3D_PRO_LOSS_TYPE
            gt_mask = gt_masks[r_head_cfg.SKL_3D_PRO_LOSS_MASK_GT] # visib
            if skl_3d_pro_loss_type == "CE":
                loss_func = nn.CrossEntropyLoss(reduction="sum", weight=None)
                loss_dict["loss_skl_3d_pro"] = loss_func(
                    out_skl_3d_pro * gt_mask[:, None], gt_skl_3d_pro * gt_mask
                ) / gt_mask.sum().float().clamp(min=1.0)
            elif skl_3d_pro_loss_type == "MSE":
                loss_func = nn.MSELoss(reduction="sum")
                loss_dict["loss_skl_3d_pro"] = loss_func(
                    out_skl_3d_pro[:, 0, :, :] * gt_mask, gt_skl_3d_pro[:, 0, :, :] * gt_mask
                ) / gt_mask.sum().float().clamp(min=1.0)
                for out_skl_3d_pro_id in range(0, out_skl_3d_pro.size(1)):
                    loss_dict["loss_skl_3d_pro"] += loss_func(
                    out_skl_3d_pro[:, out_skl_3d_pro_id, :, :] * gt_mask, gt_skl_3d_pro[:, out_skl_3d_pro_id, :, :] * gt_mask
                ) / gt_mask.sum().float().clamp(min=1.0)
            else:
                raise NotImplementedError(f"unknown kp pro loss type: {skl_3d_pro_loss_type}")
            loss_dict["loss_skl_3d_pro"] *= r_head_cfg.SKL_3D_PRO_LW

        # skl_3d loss --------------------
        if not r_head_cfg.FREEZE_skl_3d and cfg.SKELETON=='3D': # 运行
            skl_3d_loss_type = r_head_cfg.SKL_3D_LOSS_TYPE
            gt_mask = gt_masks[r_head_cfg.SKL_3D_LOSS_MASK_GT] # visib
            if skl_3d_loss_type == "CE":
                loss_func = nn.CrossEntropyLoss(reduction="sum", weight=None)
                loss_dict["loss_skl_3d"] = loss_func(
                    out_skl_3d * gt_mask[:, None], gt_skl_3d * gt_mask
                ) / gt_mask.sum().float().clamp(min=1.0)
            elif skl_3d_loss_type == "MSE":
                loss_func = nn.MSELoss(reduction="sum")
                loss_dict["loss_skl_3d"] = loss_func(
                    out_skl_3d[:, 0, :, :] * gt_mask, gt_skl_3d[:, 0, :, :] * gt_mask
                ) / gt_mask.sum().float().clamp(min=1.0)
                for out_skl_3d_id in range(0, out_skl_3d.size(1)):
                    loss_dict["loss_skl_3d"] += loss_func(
                    out_skl_3d[:, out_skl_3d_id, :, :] * gt_mask, gt_skl_3d[:, out_skl_3d_id, :, :] * gt_mask
                ) / gt_mask.sum().float().clamp(min=1.0)
            else:
                raise NotImplementedError(f"unknown kp 3d loss type: {skl_3d_loss_type}")
            loss_dict["loss_skl_3d"] *= r_head_cfg.SKL_3D_LW

        # point matching loss ---------------
        if pnp_net_cfg.PM_LW > 0: # 运行
            assert (gt_points is not None) and (gt_trans is not None) and (gt_rot is not None)
            loss_func = PyPMLoss(
                loss_type=pnp_net_cfg.PM_LOSS_TYPE, # 'L1'
                beta=pnp_net_cfg.PM_SMOOTH_L1_BETA,
                reduction="mean",
                loss_weight=pnp_net_cfg.PM_LW,
                norm_by_extent=pnp_net_cfg.PM_NORM_BY_EXTENT,
                symmetric=pnp_net_cfg.PM_LOSS_SYM,
                disentangle_t=pnp_net_cfg.PM_DISENTANGLE_T,
                disentangle_z=pnp_net_cfg.PM_DISENTANGLE_Z,
                t_loss_use_points=pnp_net_cfg.PM_T_USE_POINTS,
                r_only=pnp_net_cfg.PM_R_ONLY,
            )
            loss_pm_dict = loss_func(
                pred_rots=out_rot,
                gt_rots=gt_rot,
                points=gt_points,
                pred_transes=out_trans,
                gt_transes=gt_trans,
                extents=extents,
                sym_infos=sym_infos,
            )
            loss_dict.update(loss_pm_dict)

        # rot_loss ----------
        if pnp_net_cfg.ROT_LW > 0: # 不运行
            if pnp_net_cfg.ROT_LOSS_TYPE == "angular":
                loss_dict["loss_rot"] = angular_distance(out_rot, gt_rot)
            elif pnp_net_cfg.ROT_LOSS_TYPE == "L2":
                loss_dict["loss_rot"] = rot_l2_loss(out_rot, gt_rot)
            else:
                raise ValueError(f"Unknown rot loss type: {pnp_net_cfg.ROT_LOSS_TYPE}")
            loss_dict["loss_rot"] *= pnp_net_cfg.ROT_LW

        # centroid loss -------------
        if pnp_net_cfg.CENTROID_LW > 0: #运行
            assert (
                pnp_net_cfg.TRANS_TYPE == "centroid_z"
            ), "centroid loss is only valid for predicting centroid2d_rel_delta"

            if pnp_net_cfg.CENTROID_LOSS_TYPE == "L1":  # 运行
                loss_dict["loss_centroid"] = nn.L1Loss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
            elif pnp_net_cfg.CENTROID_LOSS_TYPE == "L2":
                loss_dict["loss_centroid"] = L2Loss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
            elif pnp_net_cfg.CENTROID_LOSS_TYPE == "MSE":
                loss_dict["loss_centroid"] = nn.MSELoss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
            else:
                raise ValueError(f"Unknown centroid loss type: {pnp_net_cfg.CENTROID_LOSS_TYPE}")
            loss_dict["loss_centroid"] *= pnp_net_cfg.CENTROID_LW

        # z loss ------------------
        if pnp_net_cfg.Z_LW > 0: #运行 REL
            if pnp_net_cfg.Z_TYPE == "REL":
                gt_z = gt_trans_ratio[:, 2]
            elif pnp_net_cfg.Z_TYPE == "ABS":
                gt_z = gt_trans[:, 2]
            else:
                raise NotImplementedError

            if pnp_net_cfg.Z_LOSS_TYPE == "L1":  # 运行
                loss_dict["loss_z"] = nn.L1Loss(reduction="mean")(out_trans_z, gt_z)
            elif pnp_net_cfg.Z_LOSS_TYPE == "L2":
                loss_dict["loss_z"] = L2Loss(reduction="mean")(out_trans_z, gt_z)
            elif pnp_net_cfg.Z_LOSS_TYPE == "MSE":
                loss_dict["loss_z"] = nn.MSELoss(reduction="mean")(out_trans_z, gt_z)
            else:
                raise ValueError(f"Unknown z loss type: {pnp_net_cfg.Z_LOSS_TYPE}")
            loss_dict["loss_z"] *= pnp_net_cfg.Z_LW

        # trans loss ------------------
        if pnp_net_cfg.TRANS_LW > 0: # 不运行
            if pnp_net_cfg.TRANS_LOSS_DISENTANGLE:
                # NOTE: disentangle xy/z
                if pnp_net_cfg.TRANS_LOSS_TYPE == "L1":
                    loss_dict["loss_trans_xy"] = nn.L1Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.L1Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                elif pnp_net_cfg.TRANS_LOSS_TYPE == "L2":
                    loss_dict["loss_trans_xy"] = L2Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = L2Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                elif pnp_net_cfg.TRANS_LOSS_TYPE == "MSE":
                    loss_dict["loss_trans_xy"] = nn.MSELoss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.MSELoss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                else:
                    raise ValueError(f"Unknown trans loss type: {pnp_net_cfg.TRANS_LOSS_TYPE}")
                loss_dict["loss_trans_xy"] *= pnp_net_cfg.TRANS_LW
                loss_dict["loss_trans_z"] *= pnp_net_cfg.TRANS_LW
            else:
                if pnp_net_cfg.TRANS_LOSS_TYPE == "L1":
                    loss_dict["loss_trans_LPnP"] = nn.L1Loss(reduction="mean")(out_trans, gt_trans)
                elif pnp_net_cfg.TRANS_LOSS_TYPE == "L2":
                    loss_dict["loss_trans_LPnP"] = L2Loss(reduction="mean")(out_trans, gt_trans)

                elif pnp_net_cfg.TRANS_LOSS_TYPE == "MSE":
                    loss_dict["loss_trans_LPnP"] = nn.MSELoss(reduction="mean")(out_trans, gt_trans)
                else:
                    raise ValueError(f"Unknown trans loss type: {pnp_net_cfg.TRANS_LOSS_TYPE}")
                loss_dict["loss_trans_LPnP"] *= pnp_net_cfg.TRANS_LW

        # bind loss (R^T@t)
        if pnp_net_cfg.get("BIND_LW", 0.0) > 0.0: # 不运行
            pred_bind = torch.bmm(out_rot.permute(0, 2, 1), out_trans.view(-1, 3, 1)).view(-1, 3)
            gt_bind = torch.bmm(gt_rot.permute(0, 2, 1), gt_trans.view(-1, 3, 1)).view(-1, 3)
            if pnp_net_cfg.BIND_LOSS_TYPE == "L1":
                loss_dict["loss_bind"] = nn.L1Loss(reduction="mean")(pred_bind, gt_bind)
            elif pnp_net_cfg.BIND_LOSS_TYPE == "L2":
                loss_dict["loss_bind"] = L2Loss(reduction="mean")(pred_bind, gt_bind)
            elif pnp_net_cfg.CENTROID_LOSS_TYPE == "MSE":
                loss_dict["loss_bind"] = nn.MSELoss(reduction="mean")(pred_bind, gt_bind)
            else:
                raise ValueError(f"Unknown bind loss (R^T@t) type: {pnp_net_cfg.BIND_LOSS_TYPE}")
            loss_dict["loss_bind"] *= pnp_net_cfg.BIND_LW

        if cfg.MODEL.CDPN.USE_MTL: # 不运行
            for _k in loss_dict:
                _name = _k.replace("loss_", "log_var_")
                cur_log_var = getattr(self, _name)
                loss_dict[_k] = loss_dict[_k] * torch.exp(-cur_log_var) + torch.log(1 + torch.exp(cur_log_var))
        return loss_dict


def get_skeleton_mask_pro_3d_out_dim(cfg):
    r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
    t_head_cfg = cfg.MODEL.CDPN.TRANS_HEAD
    skeleton_loss_type = r_head_cfg.SKELETON_LOSS_TYPE  # L1
    mask_loss_type = r_head_cfg.MASK_LOSS_TYPE  # L1
    kp_pro_loss_type = r_head_cfg.KP_PRO_LOSS_TYPE  # L1
    kp_3d_loss_type = r_head_cfg.KP_3D_LOSS_TYPE  # L1
    kp_present = r_head_cfg.KP_PRESENT
    num_kp_3d = r_head_cfg.NUM_KP_3D  # 11 个

    if skeleton_loss_type in ["L1", "BCE", "MSE"]:
        skeleton_out_dim = 2
    elif skeleton_loss_type in ["CE"]:
        skeleton_out_dim = 4
    else:
        raise NotImplementedError(f"unknown skeleton loss type: {skeleton_loss_type}")

    if mask_loss_type in ["L1", "BCE", "MSE"]:
        mask_out_dim = 1
    elif mask_loss_type in ["CE"]:
        mask_out_dim = 2
    else:
        raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")

    if kp_pro_loss_type in ["MSE", "L1", "L2", "SmoothL1"]:
        kp_pro_out_dim = num_kp_3d
    elif kp_pro_loss_type in ["CE_coor", "CE"]:
        kp_pro_out_dim = num_kp_3d * (r_head_cfg.kp_pro_BIN + 1)  #这个没用
    else:
        raise NotImplementedError(f"unknown xyz loss type: {kp_pro_loss_type}")

    if kp_3d_loss_type in ["MSE", "L1", "L2", "SmoothL1"]:
        if kp_present == 0 or kp_present == 1:
            kp_3d_out_dim = 3 * num_kp_3d
        elif kp_present == 2:
            kp_3d_out_dim = 3
    elif kp_3d_loss_type in ["CE_coor", "CE"]:
        kp_3d_out_dim = 3 * (r_head_cfg.kp_3d_BIN + 1)
    else:
        raise NotImplementedError(f"unknown xyz loss type: {kp_3d_loss_type}")

    # region_out_dim = r_head_cfg.NUM_REGIONS + 1
    # at least 2 regions (with bg, at least 3 regions)
    # assert region_out_dim > 2, region_out_dim

    return skeleton_out_dim, mask_out_dim, kp_pro_out_dim, kp_3d_out_dim


def build_model_optimizer(cfg):
    backbone_cfg = cfg.MODEL.CDPN.BACKBONE
    r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
    t_head_cfg = cfg.MODEL.CDPN.TRANS_HEAD
    pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET
    ske_out_scale = cfg.MODEL.CDPN.ROT_HEAD.SKE_OUTPUT_RES_SCALE
    output_res = cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES

    if "resnet" in backbone_cfg.ARCH:
        params_lr_list = []
        # backbone net
        block_type, layers, channels, name = resnet_spec[backbone_cfg.NUM_LAYERS]
        backbone_net = ResNetBackboneNet(
            block_type, layers, backbone_cfg.INPUT_CHANNEL, freeze=backbone_cfg.FREEZE, rot_concat=r_head_cfg.ROT_CONCAT
        )
        if backbone_cfg.FREEZE:
            for param in backbone_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append(
                {
                    "params": filter(lambda p: p.requires_grad, backbone_net.parameters()),
                    "lr": float(cfg.SOLVER.BASE_LR),
                }
            )

        # rotation head net -----------------------------------------------------
        skeleton_out_dim, mask_out_dim, kp_pro_out_dim, kp_3d_out_dim = get_skeleton_mask_pro_3d_out_dim(cfg)  # 1  1 3 30or3
        rot_head_net = RotWithRegionHead(
            cfg,
            channels[-1],
            r_head_cfg.NUM_LAYERS,
            r_head_cfg.NUM_FILTERS,
            r_head_cfg.CONV_KERNEL_SIZE,
            r_head_cfg.OUT_CONV_KERNEL_SIZE,
            skeleton_output_dim=skeleton_out_dim,
            mask_output_dim=mask_out_dim,
            kp_pro_output_dim=kp_pro_out_dim,
            kp_3d_output_dim=kp_3d_out_dim,
            freeze=r_head_cfg.FREEZE,
            num_classes=r_head_cfg.NUM_CLASSES,
            skeleton_class_aware=r_head_cfg.SKELETON_CLASS_AWARE,
            mask_class_aware=r_head_cfg.MASK_CLASS_AWARE,
            kp_pro_class_aware=r_head_cfg.KP_PRO_CLASS_AWARE,
            kp_3d_class_aware=r_head_cfg.KP_3D_CLASS_AWARE,
            num_regions=r_head_cfg.NUM_REGIONS,
            norm=r_head_cfg.NORM,
            num_gn_groups=r_head_cfg.NUM_GN_GROUPS,
            ske_up_times=ske_out_scale,
            output_res = output_res,
        )
        if r_head_cfg.FREEZE:
            for param in rot_head_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append(
                {
                    "params": filter(lambda p: p.requires_grad, rot_head_net.parameters()),
                    "lr": float(cfg.SOLVER.BASE_LR),
                }
            )

        # translation head net --------------------------------------------------------
        if not t_head_cfg.ENABLED:
            trans_head_net = None
            assert not pnp_net_cfg.R_ONLY, "if pnp_net is R_ONLY, trans_head must be enabled!"
        else:
            trans_head_net = TransHeadNet(
                channels[-1],  # the channels of backbone output layer
                t_head_cfg.NUM_LAYERS,
                t_head_cfg.NUM_FILTERS,
                t_head_cfg.CONV_KERNEL_SIZE,
                t_head_cfg.OUT_CHANNEL,
                freeze=t_head_cfg.FREEZE,
                norm=t_head_cfg.NORM,
                num_gn_groups=t_head_cfg.NUM_GN_GROUPS,
            )
            if t_head_cfg.FREEZE:
                for param in trans_head_net.parameters():
                    with torch.no_grad():
                        param.requires_grad = False
            else:
                params_lr_list.append(
                    {
                        "params": filter(lambda p: p.requires_grad, trans_head_net.parameters()),
                        "lr": float(cfg.SOLVER.BASE_LR) * t_head_cfg.LR_MULT,
                    }
                )

        # -----------------------------------------------
        # if r_head_cfg.XYZ_LOSS_TYPE in ["CE_coor", "CE"]: # L1
        #     pnp_net_in_channel = r_out_dim - 3
        # else:
        if pnp_net_cfg.KP_3D_CA and cfg.SKELETON =='2D':  #  利用pre_kp_pro的值来优化pre_kp_3d，没有采用cat的形式，而是类似于etlwise.product的方法
            pnp_net_in_channel = kp_3d_out_dim
        elif cfg.SKELETON =='2D':
            pnp_net_in_channel = kp_pro_out_dim + kp_3d_out_dim
        elif cfg.SKELETON =='3D':
            pnp_net_in_channel = output_res*4

        if pnp_net_cfg.WITH_2D_COORD:  # True
            pnp_net_in_channel += 2

        if cfg.SKELETON =='2D' and pnp_net_cfg.SKELETON_ATTENTION in ["concat"]:  # True
            pnp_net_in_channel += 1

        if pnp_net_cfg.MASK_ATTENTION in ["concat"]:  # do not add dim for none/mul
            pnp_net_in_channel += 1



        if pnp_net_cfg.ROT_TYPE in ["allo_quat", "ego_quat"]:
            rot_dim = 4
        elif pnp_net_cfg.ROT_TYPE in ["allo_log_quat", "ego_log_quat", "allo_lie_vec", "ego_lie_vec"]:
            rot_dim = 3
        elif pnp_net_cfg.ROT_TYPE in ["allo_rot6d", "ego_rot6d"]:  # allo_rot6d
            rot_dim = 6
        else:
            raise ValueError(f"Unknown ROT_TYPE: {pnp_net_cfg.ROT_TYPE}")

        pnp_head_cfg = pnp_net_cfg.PNP_HEAD_CFG
        pnp_head_type = pnp_head_cfg.pop("type")
        if pnp_head_type == "ConvPnPNet":  # ConvPnPNet
            pnp_head_cfg.update(
                nIn=pnp_net_in_channel,
                rot_dim=rot_dim,
                num_regions=r_head_cfg.NUM_REGIONS,
                featdim=128,
                num_layers=3,
                mask_attention_type=pnp_net_cfg.MASK_ATTENTION,
                skeleton_attention_type=pnp_net_cfg.SKELETON_ATTENTION,

            )
            pnp_net = ConvPnPNet(**pnp_head_cfg)
        elif pnp_head_type == "PointPnPNet":
            pnp_head_cfg.update(nIn=pnp_net_in_channel, rot_dim=rot_dim, num_regions=r_head_cfg.NUM_REGIONS)
            pnp_net = PointPnPNet(**pnp_head_cfg)
        elif pnp_head_type == "SimplePointPnPNet":
            pnp_head_cfg.update(
                nIn=pnp_net_in_channel,
                rot_dim=rot_dim,
                num_regions=r_head_cfg.NUM_REGIONS,
                featdim=128,
                num_layers=3,
                mask_attention_type=pnp_net_cfg.MASK_ATTENTION,
                skeleton_attention_type=pnp_net_cfg.SKELETON_ATTENTION,
                # num_regions=r_head_cfg.NUM_REGIONS,
            )
            pnp_net = SimplePointPnPNet(**pnp_head_cfg)
        else:
            raise ValueError(f"Unknown pnp head type: {pnp_head_type}")

        if pnp_net_cfg.FREEZE:
            for param in pnp_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append(
                {
                    "params": filter(lambda p: p.requires_grad, pnp_net.parameters()),
                    "lr": float(cfg.SOLVER.BASE_LR) * pnp_net_cfg.LR_MULT,
                }
            )
        # ================================================

        # CDPN (Coordinates-based Disentangled Pose Network)
        model = ske(cfg, backbone_net, rot_head_net, trans_head_net=trans_head_net, pnp_net=pnp_net)
        if cfg.MODEL.CDPN.USE_MTL:
            params_lr_list.append(
                {
                    "params": filter(
                        lambda p: p.requires_grad,
                        [_param for _name, _param in model.named_parameters() if "log_var" in _name],
                    ),
                    "lr": float(cfg.SOLVER.BASE_LR),
                }
            )

        # get optimizer
        optimizer = build_optimizer_with_params(cfg, params_lr_list)

    if cfg.MODEL.WEIGHTS == "":
        ## backbone initialization
        backbone_pretrained = cfg.MODEL.CDPN.BACKBONE.get("PRETRAINED", "")
        if backbone_pretrained == "":
            logger.warning("Randomly initialize weights for backbone!")
        else:
            # initialize backbone with official ImageNet weights
            logger.info(f"load backbone weights from: {backbone_pretrained}")
            load_checkpoint(model.backbone, backbone_pretrained, strict=False, logger=logger)

    model.to(torch.device(cfg.MODEL.DEVICE))
    return model, optimizer
