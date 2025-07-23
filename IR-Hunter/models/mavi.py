# Copyright 2022 CircuitNet. All rights reserved.

import torch
import torch.nn as nn

from .InceptionEncoder import IncepEncoder
from .InceptionEncoder import generation_init_weights

from mmcv.cnn import constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

sys.path.append(os.path.abspath("/mnt/nfs/data/home/2220221950/zxj/Circuitnet3_1/models/networks"))
from models.networks.mask2former import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP

from collections import OrderedDict


def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    return missing_keys


class DoubleConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            DoubleConv3d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv2d(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv2d(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# 添加
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # global g_prediction_task
        # if ("DRC" == g_prediction_task):
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),  # nn.BatchNorm2d(out_channels),
            nn.PReLU(num_parameters=out_channels),  # nn.LeakyReLU(0.2, inplace=True),#nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),  # nn.BatchNorm2d(out_channels),
            nn.PReLU(num_parameters=out_channels))  # nn.LeakyReLU(0.2, inplace=True),#nn.ReLU(inplace=True)
        # elif ("Congestion" == g_prediction_task):
        # self.double_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.PReLU(num_parameters=out_channels),  # nn.LeakyReLU(0.2, inplace=True),#nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.PReLU(num_parameters=out_channels))  # nn.LeakyReLU(0.2, inplace=True),#nn.ReLU(inplace=True)
        # else:
        #     print("ERROR on prediction task!!")

    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MAVI(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=4,
                 bilinear=False,
                 **kwargs):
        super(MAVI, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv3d(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1

        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

        self.incepEncoder = IncepEncoder(True, 1, 512)
        self.Conv4Inception = DoubleConv(512, 512)

        # --- 新增 Attention/FFN/MLP 模块 ---
        # 在最深层 x4 部分加入 Self-Attention 和 FFN 模块
        self.self_attn_layer = SelfAttentionLayer(d_model=512, nhead=8, dropout=0.2,
                                                  normalize_before=False, use_layer_scale=True)
        self.ffn_layer = FFNLayer(d_model=512, dim_feedforward=2048, dropout=0.2,
                                  normalize_before=False, use_layer_scale=True)
        # 在 decoder 的 up1 部分加入 Cross-Attention 融合 skip 连接特征
        self.cross_attn_layer = CrossAttentionLayer(d_model=256, nhead=8, dropout=0.2,
                                                    normalize_before=False, use_layer_scale=False)
        # 在最后的 2D 特征上加入 MLP 进一步处理（可选）
        self.mlp = MLP(input_dim=64, hidden_dim=128, output_dim=64, num_layers=2)

    def forward(self, x):
        # 输入 x 形状: (B, in_channels, D, H, W)
        x_in = x[:, :, :self.out_channels, :, :]

        # Encoder 路径
        x1 = self.inc(x)  # (B, 64, D, H, W)
        x2 = self.down1(x1)  # (B, 128, D/2, H/2)
        x3 = self.down2(x2)  # (B, 256, D/4, H/4)
        x4 = self.down3(x3)  # (B, 512, D/8, H/8)

        # --- 先进行 Inception 处理 ---
        # 对深度维度求均值，得到 2D 特征
        x4B = self.incepEncoder(x4.mean(dim=2))  # (B, 512, H, W)
        x4C = self.Conv4Inception(x4B)  # (B, 512, H, W)

        # --- 然后对 x4C 进行自注意力和 FFN 处理 ---
        B, C, H, W = x4C.shape
        seq_len = H * W
        # 展平成序列形式： (seq_len, B, 512)
        x4_flat = x4C.view(B, C, seq_len).permute(2, 0, 1)
        x4_att = self.self_attn_layer(x4_flat)
        x4_ffn = self.ffn_layer(x4_att)
        # 恢复成原始 3D 形状
        x4_proc = x4_ffn.permute(1, 2, 0).contiguous().view(B, C, H, W)

        # --- Decoder 部分 ---
        # up1：将 x4_proc 与 x3.mean(dim=2) 融合
        x3_mean = x3.mean(dim=2)  # (B, 256, H', W')
        x_up1 = self.up1(x4_proc, x3_mean)  # (B, 256, H', W')
        # 对 up1 融合结果应用 Cross-Attention
        B_up, C_up, H_up, W_up = x_up1.shape
        seq_len_up = H_up * W_up
        x_up1_flat = x_up1.view(B_up, C_up, seq_len_up).permute(2, 0, 1)  # (seq_len_up, B, 256)
        x3_flat = x3_mean.view(B_up, x3_mean.size(1), -1).permute(2, 0, 1)  # (seq_len_skip, B, 256)
        x_up1_att = self.cross_attn_layer(x_up1_flat, x3_flat)
        x_up1 = x_up1_att.permute(1, 2, 0).contiguous().view(B_up, C_up, H_up, W_up)

        # up2 和 up3 按原结构上采样
        x2_mean = x2.mean(dim=2)  # (B, 128, H'', W'')
        x1_mean = x1.mean(dim=2)  # (B, 64, H''', W''')
        x_up2 = self.up2(x_up1, x2_mean)  # (B, 128, H'', W'')
        x_up3 = self.up3(x_up2, x1_mean)  # (B, 64, H''', W''')

        # 在 decoder 最后加入 MLP 对 2D 特征进一步处理
        # 将特征图每个位置的 64 维特征看作一个向量，经过 MLP 调整后再 reshape
        B_final, C_final, H_final, W_final = x_up3.shape
        x_flat = x_up3.view(B_final, C_final, -1).permute(2, 0, 1)  # (seq_len_final, B, 64)
        x_mlp = self.mlp(x_flat)  # (seq_len_final, B, 64)
        x_up3 = x_mlp.permute(1, 2, 0).contiguous().view(B_final, C_final, H_final, W_final)

        logits = self.outc(x_up3)  # (B, out_channels, H_final, W_final)
        # 与部分原始输入相乘（根据实际任务选择是否保留）
        logits = x_in.squeeze(1) * logits
        # 返回 2D 输出：对通道求和
        return torch.sum(logits, dim=1)

    # def forward(self, x):
    #     # 添加
    #     # print(x)
    #     # print(x.shape)
    #     # print(x.dim)
    #     x_in = x[:, :, :self.out_channels, :, :]
    #     x1 = self.inc(x)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     # print(x4)
    #     # print(x4.shape)
    #     # print(x4.dim)

    #     x4B = self.incepEncoder(x4.mean(dim=2))  # Insert the Inception Module at the bottleneck
    #     x4C = self.Conv4Inception(x4B)

    #     x = self.up1(x4C, x3.mean(dim=2))
    #     x = self.up2(x, x2.mean(dim=2))
    #     x = self.up3(x, x1.mean(dim=2))
    #     logits = self.outc(x)

    #     # logits = x_in.squeeze(1)*logits
    #     logits = x_in.squeeze(1)*logits
    #     return torch.sum(logits, dim=1)

    def init_weights(self, pretrained=None, strict=True, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m.weight, 1)
                    constant_init(m.bias, 0)

                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

