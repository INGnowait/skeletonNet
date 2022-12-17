import torch.nn as nn
import torch
import math
from mmcv.cnn import normal_init, kaiming_init, constant_init
from core.utils.layer_utils import get_norm
from torch.nn.modules.batchnorm import _BatchNorm
from .resnet_backbone import resnet_spec
import matplotlib.pyplot as plt
from .model_utils import compute_mean_re_te, get_mask_prob


class RotWithRegionHead(nn.Module):
    def __init__(
        self,
        cfg,
        in_channels,
        num_layers=3,
        num_filters=256,
        kernel_size=3,
        output_kernel_size=1,
        skeleton_output_dim=1,
        mask_output_dim=1,
        kp_pro_output_dim=3,
        kp_3d_output_dim=3,
        freeze=False,
        num_classes=1,
        skeleton_class_aware=False,
        mask_class_aware=False,
        kp_pro_class_aware=False,
        kp_3d_class_aware=False,
        num_regions=8,
        norm="BN",
        num_gn_groups=32,
        ske_up_times=8,
        output_res=64,
    ):
        super().__init__()

        self.cfg = cfg
        self.freeze = freeze
        self.ske_up_times=ske_up_times
        self.output_res = output_res
        self.concat = cfg.MODEL.CDPN.ROT_HEAD.ROT_CONCAT
        assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, "Only support kenerl 2, 3 and 4"
        assert num_regions > 1, f"Only support num_regions > 1, but got {num_regions}"
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        assert output_kernel_size == 1 or output_kernel_size == 3, "Only support kenerl 1 and 3"
        if output_kernel_size == 1:
            pad = 0
        elif output_kernel_size == 3:
            pad = 1

        if self.concat:
            _, _, channels, _ = resnet_spec[cfg.MODEL.CDPN.BACKBONE.NUM_LAYERS]
            self.features = nn.ModuleList()
            self.features.append(
                nn.ConvTranspose2d(
                    in_channels,
                    num_filters,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
            self.features.append(nn.ReLU(inplace=True))
            for i in range(num_layers):
                self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
                self.features.append(
                    nn.Conv2d(
                        num_filters + channels[-2 - i], num_filters, kernel_size=3, stride=1, padding=1, bias=False
                    )
                )
                self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
                self.features.append(nn.ReLU(inplace=True))

                self.features.append(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
                )
                self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
                self.features.append(nn.ReLU(inplace=True))
        else:
            self.features = nn.ModuleList()
            self.features.append(
                nn.ConvTranspose2d(
                    in_channels,
                    num_filters,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
            self.features.append(nn.ReLU(inplace=True))
            for i in range(num_layers):
                # _in_channels = in_channels if i == 0 else num_filters
                # self.features.append(
                #    nn.ConvTranspose2d(_in_channels, num_filters, kernel_size=kernel_size, stride=2, padding=padding,
                #                       output_padding=output_padding, bias=False))
                # self.features.append(nn.BatchNorm2d(num_filters))
                # self.features.append(nn.ReLU(inplace=True))
                if i >= 1:
                    self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
                self.features.append(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
                )
                self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
                self.features.append(nn.ReLU(inplace=True))

                self.features.append(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
                )
                self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
                self.features.append(nn.ReLU(inplace=True))


        self.skeleton_output_dim = skeleton_output_dim
        if skeleton_class_aware:
            self.skeleton_output_dim *= num_classes

        self.mask_output_dim = mask_output_dim
        if mask_class_aware:
            self.mask_output_dim *= num_classes

        self.kp_pro_output_dim = kp_pro_output_dim
        if kp_pro_class_aware:
            self.kp_pro_output_dim *= num_classes

        self.kp_3d_output_dim = kp_3d_output_dim # add one channel for bg
        if kp_3d_class_aware:
            self.kp_3d_output_dim *= num_classes


        if self.cfg.SKELETON=='2D':
            self.features.append(
                nn.Conv2d(
                    num_filters,
                    self.skeleton_output_dim + self.mask_output_dim + self.kp_pro_output_dim + self.kp_3d_output_dim,
                    kernel_size=output_kernel_size,
                    padding=pad,
                    bias=True,
                )
            )
        elif self.cfg.SKELETON=='3D':
            self.features.append(
                nn.Conv2d(
                    num_filters,
                    self.mask_output_dim + self.output_res*4,
                    kernel_size=output_kernel_size,
                    padding=pad,
                    bias=True,
                )
            )

        if self.cfg.MODEL.CDPN.PNP_NET.SKELETON_2D_DETECTION:
            self.ske_branch1 = nn.Conv2d(2, num_filters, kernel_size=1, stride=1, padding=0, bias=False)
            self.ske_branch_norm1 = get_norm(norm, num_filters, num_gn_groups=num_gn_groups)
            self.ske_branch_re1 = nn.ReLU(inplace=True)

            self.ske_branch2 = nn.Conv2d(num_filters, num_filters, kernel_size=1, stride=1, padding=0, bias=False)
            self.ske_branch_norm2 = get_norm(norm, num_filters, num_gn_groups=num_gn_groups)
            self.ske_branch_re2 = nn.ReLU(inplace=True)

            self.ske_branch3 = nn.Conv2d(num_filters, num_filters, kernel_size=1, stride=1, padding=0, bias=False)
            self.ske_branch_norm3 = get_norm(norm, num_filters, num_gn_groups=num_gn_groups)
            self.ske_branch_re3 = nn.ReLU(inplace=True)

            self.ske_up_up_4_times_1 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.ske_up_conv_1 = nn.Conv2d(num_filters, num_filters, kernel_size=1, stride=1, padding=0, bias=False)
            self.ske_up_norm_1 = get_norm(norm, num_filters, num_gn_groups=num_gn_groups)
            self.ske_up_re_1 = nn.ReLU(inplace=True)

            self.ske_up_conv_2 = nn.Conv2d(num_filters, num_filters, kernel_size=1, stride=1, padding=0, bias=False)
            self.ske_up_norm_2 = get_norm(norm, num_filters, num_gn_groups=num_gn_groups)
            self.ske_up_re_2 = nn.ReLU(inplace=True)

            self.ske_up_conv_3 = nn.Conv2d(num_filters, 2, kernel_size=1, stride=1, padding=0, bias=False)
            self.ske_up_norm_3 = get_norm(norm, 2, num_gn_groups=num_gn_groups)

            self.ske_up_up_4_times_flux = nn.UpsamplingBilinear2d(scale_factor=2)

            self.ske_branch_E1 = nn.Conv2d(2, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
            self.ske_branch_E_norm1 = get_norm(norm, num_filters, num_gn_groups=num_gn_groups)
            self.ske_branch_E_re1 = nn.ReLU(inplace=True)

            self.ske_branch_E2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
            self.ske_branch_E_norm2 = get_norm(norm, num_filters, num_gn_groups=num_gn_groups)
            self.ske_branch_E_re2 = nn.ReLU(inplace=True)

            self.ske_branch_E3 = nn.Conv2d(num_filters, 1, kernel_size=3, stride=1, padding=1, bias=False)
            self.ske_branch_E_norm3 = get_norm(norm, 1, num_gn_groups=num_gn_groups)
            self.ske_branch_E_re3 = nn.ReLU(inplace=True)

            self.ske_up_up_4_times_single_pixel = nn.UpsamplingBilinear2d(scale_factor=2)

            self.ske_up_up_4_times_single_pixel_sigmod = nn.Sigmoid()
            self.kp_pro_sigmod = nn.Sigmoid()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

    def crop_1(self, d, size):  # use to keep same to the out the former layer
        d_h, d_w = d.size()[2:4]
        g_h, g_w = size[0], size[1]
        d1 = d[:, :, int(math.floor((d_h - g_h) / 2.0)):int(math.floor((d_h - g_h) / 2.0)) + g_h,
             int(math.floor((d_w - g_w) / 2.0)):int(math.floor((d_w - g_w) / 2.0)) + g_w]
        return d1

    def crop_2(self, d, region, up_times):  # use for crop the keep to input data
        x, y, h, w = region
        d1 = d[:, :, x:x + h*up_times, y:y + w*up_times]
        return d1

    def forward(self, x, x_f64=None, x_f32=None, x_f16=None):
        if self.concat:
            if self.freeze:
                with torch.no_grad():
                    for i, l in enumerate(self.features):
                        if i == 3:
                            x = l(torch.cat([x, x_f16], 1))
                        elif i == 12:
                            x = l(torch.cat([x, x_f32], 1))
                        elif i == 21:
                            x = l(torch.cat([x, x_f64], 1))
                        x = l(x)
                    return x.detach()
            else:
                for i, l in enumerate(self.features):
                    if i == 3:
                        x = torch.cat([x, x_f16], 1)
                    elif i == 12:
                        x = torch.cat([x, x_f32], 1)
                    elif i == 21:
                        x = torch.cat([x, x_f64], 1)
                    x = l(x)
                return x
        else:
            if self.freeze:
                with torch.no_grad():
                    for i, l in enumerate(self.features):
                        x = l(x)
                    skeleton = x[:, : self.skeleton_output_dim, :, :]
                    mask = x[:, self.skeleton_output_dim: self.mask_output_dim + self.skeleton_output_dim, :, :]
                    kp_pro = x[:,self.mask_output_dim + self.skeleton_output_dim:self.mask_output_dim + self.skeleton_output_dim + self.kp_pro_output_dim, :, :]
                    kp_3d = x[:, self.mask_output_dim + self.skeleton_output_dim + self.kp_pro_output_dim:, :, :]

                    return (skeleton.detach(), mask.detach(), kp_pro.detach(), kp_3d.detach())
            else:
                for i, l in enumerate(self.features):
                    x = l(x)
                mask = x[:, : self.mask_output_dim, :, :]
                if self.cfg.SKELETON=='2D':
                    skeleton_rot_head = x[:, self.mask_output_dim: self.skeleton_output_dim + self.mask_output_dim, :, :]
                    kp_pro = x[:, self.mask_output_dim + self.skeleton_output_dim:self.mask_output_dim + self.skeleton_output_dim + self.kp_pro_output_dim, :, :]
                    kp_3d = x[:, self.mask_output_dim + self.skeleton_output_dim + self.kp_pro_output_dim:self.mask_output_dim + self.skeleton_output_dim + + self.kp_pro_output_dim + self.kp_3d_output_dim, :, :]

                    # 将kp_pro和kp_3d(xxxxxyyyyyzzzzz)合在一起
                    # kp_pro_sig = self.kp_pro_sigmod(kp_pro)
                    # mask_sig = get_mask_prob(self.cfg, mask)
                    # kp_pro_sig_mask = kp_pro_sig * mask_sig

                    # x3_kp_pro = torch.cat([kp_pro_sig_mask, kp_pro_sig_mask, kp_pro_sig_mask], dim=1)
                    # kp_3d_pro= torch.mul(kp_3d, x3_kp_pro)
                    if self.cfg.MODEL.CDPN.PNP_NET.SKELETON_2D_DETECTION:
                        ske_branch1 = self.ske_branch1(skeleton_rot_head)
                        ske_branch_norm1 = self.ske_branch_norm1(ske_branch1)
                        ske_branch_re1 = self.ske_branch_re1(ske_branch_norm1)

                        ske_branch2 = self.ske_branch2(ske_branch_re1)
                        ske_branch_norm2 = self.ske_branch_norm2(ske_branch2)
                        ske_branch_re2 = self.ske_branch_re2(ske_branch_norm2)

                        ske_branch3 = self.ske_branch3(ske_branch_re2)
                        ske_branch_norm3 = self.ske_branch_norm3(ske_branch3)
                        ske_branch_re = self.ske_branch_re3(ske_branch_norm3)

                        ske_up_up1 = self.ske_up_up_4_times_1(ske_branch_re)
                        ske_up_conv1 = self.ske_up_conv_1(ske_up_up1)
                        ske_up_norm1 = self.ske_up_norm_1(ske_up_conv1)
                        ske_up_re1 = self.ske_up_re_1(ske_up_norm1)

                        ske_up_conv2 = self.ske_up_conv_2(ske_up_re1)
                        ske_up_norm2 = self.ske_up_norm_2(ske_up_conv2)
                        ske_up_re2 = self.ske_up_re_2(ske_up_norm2)

                        ske_up_conv3 = self.ske_up_conv_3(ske_up_re2)
                        ske_up_norm3 = self.ske_up_norm_3(ske_up_conv3)
                        # ske_up_re3 = self.ske_up_re_3(ske_up_norm3)

                        ske_up_up_4_times_flux = self.ske_up_up_4_times_flux(ske_up_norm3)

                        ske_branch_E1 = self.ske_branch_E1(ske_up_norm3)
                        ske_branch_E_norm1 = self.ske_branch_E_norm1(ske_branch_E1)
                        ske_branch_E_re1 = self.ske_branch_E_re1(ske_branch_E_norm1)

                        ske_branch_E2 = self.ske_branch_E2(ske_branch_E_re1)
                        ske_up_E_norm2 = self.ske_branch_E_norm2(ske_branch_E2)
                        ske_branch_E_re2 = self.ske_branch_E_re2(ske_up_E_norm2)

                        ske_branch_E3 = self.ske_branch_E3(ske_branch_E_re2)
                        ske_up_E_norm3 = self.ske_branch_E_norm3(ske_branch_E3)
                        # ske_branch_E_re3 = self.ske_branch_E_re3(ske_up_E_norm3)

                        ske_up_up_4_times_single_pixel = self.ske_up_up_4_times_single_pixel(ske_up_E_norm3)

                        skeleton_flux = self.crop_2(ske_up_up_4_times_flux, (0, 0) + skeleton_rot_head.size()[2:4],
                                                    self.ske_up_times)
                        skeleton_single = self.crop_2(ske_up_up_4_times_single_pixel,
                                                      (0, 0) + skeleton_rot_head.size()[2:4], self.ske_up_times)
                        ske_up_up_4_times_single_pixel_sigmod = self.ske_up_up_4_times_single_pixel_sigmod(
                            skeleton_single)

                    else:
                        skeleton_flux = None
                        ske_up_up_4_times_single_pixel_sigmod = None
                        skl_3d = None
                        skl_pro = None
                elif self.cfg.SKELETON=='3D':

                    skl_3d = x[:, self.mask_output_dim: self.mask_output_dim + self.output_res*3, :, :]
                    skl_pro =x[:, self.mask_output_dim  + self.output_res*3 : , :, :]



                    skeleton_rot_head  = None
                    skeleton_flux = None
                    ske_up_up_4_times_single_pixel_sigmod = None
                    kp_pro = None
                    kp_3d = None


                return skeleton_rot_head, skeleton_flux, ske_up_up_4_times_single_pixel_sigmod, mask, kp_pro, kp_3d, skl_3d, skl_pro



