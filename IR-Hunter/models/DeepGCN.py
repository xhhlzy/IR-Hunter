# # 2022.06.17-Changed for building ViG model
# #            Huawei Technologies Co., Ltd. <foss@huawei.com>
# # !/usr/bin/env python
# # -*- coding: utf-8 -*-
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import Sequential as Seq
#
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.models.helpers import load_pretrained
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model
#
# from gcn_lib import Grapher, act_layer
#
#
# def _cfg(url='', **kwargs):
#     return {
#         'url': url,
#         'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
#         'crop_pct': .9, 'interpolation': 'bicubic',
#         'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
#         'first_conv': 'patch_embed.proj', 'classifier': 'head',
#         **kwargs
#     }
#
#
# default_cfgs = {
#     'vig_224_gelu': _cfg(
#         mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
#     ),
#     'vig_b_224_gelu': _cfg(
#         crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
#     ),
# }
#
#
# class FFN(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Sequential(
#             nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
#             nn.BatchNorm2d(hidden_features),
#         )
#         self.act = act_layer(act)
#         self.fc2 = nn.Sequential(
#             nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
#             nn.BatchNorm2d(out_features),
#         )
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#     def forward(self, x):
#         shortcut = x
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.fc2(x)
#         x = self.drop_path(x) + shortcut
#         return x  # .reshape(B, C, N, 1)
#
#
# class Stem(nn.Module):
#     """ Image to Visual Embedding
#     Overlap: https://arxiv.org/pdf/2106.13797.pdf
#     """
#
#     def __init__(self, img_size=256, in_dim=3, out_dim=768, act='relu'):
#         super().__init__()
#         self.convs = nn.Sequential(
#             nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1),
#             nn.BatchNorm2d(out_dim // 2),
#             act_layer(act),
#             nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
#             nn.BatchNorm2d(out_dim),
#             act_layer(act),
#             nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
#             nn.BatchNorm2d(out_dim),
#         )
#
#     def forward(self, x):
#         x = self.convs(x)
#         return x
#
#
# class Downsample(nn.Module):
#     """ Convolution-based downsample
#     """
#
#     def __init__(self, in_dim=3, out_dim=768):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
#             nn.BatchNorm2d(out_dim),
#         )
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x
#
# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(48, 4, kernel_size=1)
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# # 添加
# class DoubleConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super(DoubleConv2d, self).__init__()
#         if mid_channels is None:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.double_conv(x)
#
# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels, scale_factor=2, bilinear=True):
#         super(Up, self).__init__()
#         self.scale_factor = scale_factor
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
#             # 调整中间通道数，使其与输入通道数匹配
#             mid_channels = in_channels // 2 if in_channels // 2 > 0 else 1
#             self.conv = DoubleConv2d(in_channels, out_channels, mid_channels)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor)
#             self.conv = DoubleConv2d(out_channels, out_channels)
#
#     def forward(self, x):
#         x = self.up(x)
#         x = self.conv(x)
#         return x
#
#
#
#
# class DeepGCN(torch.nn.Module):
#     def __init__(self, opt):
#         super(DeepGCN, self).__init__()
#         print(opt)
#         k = opt.k
#         act = opt.act
#         norm = opt.norm
#         bias = opt.bias
#         epsilon = opt.epsilon
#         stochastic = opt.use_stochastic
#         conv = opt.conv
#         emb_dims = opt.emb_dims
#         drop_path = opt.drop_path
#
#         blocks = opt.blocks
#         self.n_blocks = sum(blocks)
#         channels = opt.channels
#         reduce_ratios = [4, 2, 1, 1]
#         dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule
#         num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
#         max_dilation = 49 // max(num_knn)
#
#         self.stem = Stem(out_dim=channels[0], act=act)
#         self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 256 // 4, 256 // 4))
#         HW = 256 // 4 * 256 // 4
#
#         self.backbone = nn.ModuleList([])
#         idx = 0
#         for i in range(len(blocks)):
#             if i > 0:
#                 self.backbone.append(Downsample(channels[i - 1], channels[i]))
#                 HW = HW // 4
#             for j in range(blocks[i]):
#                 self.backbone += [
#                     Seq(Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
#                                 bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
#                                 relative_pos=True),
#                         FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
#                         )]
#                 idx += 1
#         self.backbone = Seq(*self.backbone)
#
#         self.prediction = Seq(nn.Conv2d(channels[-1], 1024, 1, bias=True),
#                               nn.BatchNorm2d(1024),
#                               act_layer(act),
#                               nn.Dropout(opt.dropout),
#                               nn.Conv2d(1024, opt.n_classes, 1, bias=True))
#
#         self.input_conv = nn.Conv2d(3, 512, kernel_size=1)
#         self.model_init()
#
#
#         # 添加
#         bilinear = opt.bilinear
#         # # factor = 2 if bilinear else 1
#         # self.up1 = Up(512, 256, bilinear)
#         # self.up2 = Up(256, 128, bilinear)
#         # self.up3 = Up(128, 64, bilinear)
#         # self.outc = OutConv(64, 4)
#
#         self.up1 = Up(512, 256, scale_factor=8, bilinear=True)
#         self.up2 = Up(256, 128, scale_factor=2, bilinear=True)
#         self.up3 = Up(128, 3, scale_factor=2, bilinear=True)
#
#
#     def model_init(self):
#         for m in self.modules():
#             if isinstance(m, torch.nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#                 m.weight.requires_grad = True
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#                     m.bias.requires_grad = True
#
#     # def forward(self, inputs):
#     #     pool = nn.AdaptiveAvgPool3d((1, None, None))  # 将深度维度降为 1
#     #     inputs = pool(inputs)  # 输出形状为 [4, 1, 1, 256, 256]
#     #     inputs = inputs.squeeze(2)  # 移除深度维度，形状为 [4, 1, 256, 256]
#     #     inputs = inputs.repeat(1, 3, 1, 1)
#     #
#     #     x = self.stem(inputs) + self.pos_embed
#     #     B, C, H, W = x.shape
#     #     for i in range(len(self.backbone)):
#     #         x = self.backbone[i](x)
#     #
#     #     x = F.adaptive_avg_pool2d(x, 1)
#     #     return self.prediction(x).squeeze(-1).squeeze(-1)
#
#     # def forward(self, inputs):
#     #     pool = nn.AdaptiveAvgPool3d((1, None, None))  # 将深度维度降为 1
#     #     inputs = pool(inputs)  # 输出形状为 [4, 1, 1, 256, 256]
#     #     inputs = inputs.squeeze(2)  # 移除深度维度，形状为 [4, 1, 256, 256]
#     #     inputs = inputs.repeat(1, 3, 1, 1)
#     #
#     #     x = self.stem(inputs) + self.pos_embed
#     #     B, C, H, W = x.shape
#     #     for i in range(len(self.backbone)):
#     #         x = self.backbone[i](x)
#     #
#     #     inputs = inputs.view(x.shape)
#     #     x = inputs.squeeze(1) * x  # 形状: (batch_size, out_channels, height, width)
#     #     return torch.sum(x, dim=1)
#
#     def forward(self, inputs):
#         pool = nn.AdaptiveAvgPool3d((1, None, None))  # 将深度维度降为 1
#         inputs = pool(inputs)  # 输出形状为 [batch_size, 1, 1, height, width]
#         inputs = inputs.squeeze(2)  # 移除深度维度，形状为 [batch_size, 1, height, width]
#         inputs = inputs.repeat(1, 3, 1, 1)  # 扩展通道维度，形状为 [batch_size, 3, height, width]
#         # print(inputs.size)
#         # print(inputs.shape)
#
#         x = self.stem(inputs) + self.pos_embed
#         for block in self.backbone:
#             x = block(x)
#             # 添加
#             # print(x)
#             # print(x.shape)
#             # print(x.size)
#             # x = self.up1(x)
#             # x = self.up2(x)
#             # x = self.up3(x)
#         x = self.up1(x)
#         x = self.up2(x)
#         x = self.up3(x)
#         # print(x)
#         # print(x.shape)
#         # print(x.size)
#
#         # logits = self.up1(x)
#         logits = inputs * x
#         # print(logits)
#
#         # inputs_resized = self.input_conv(inputs)
#
#         # # 确保 inputs 和 x 在空间维度上的一致性
#         # inputs_resized = F.interpolate(inputs_resized, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
#         # x = inputs_resized * x  # 形状: (batch_size, out_channels, height, width)
#         return torch.sum(logits, dim=1)
#
#
# @register_model
# def pvig_ti_224_gelu(pretrained=False, **kwargs):
#     class OptInit:
#         def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
#             self.k = 9  # neighbor num (default:9)
#             self.conv = 'mr'  # graph conv layer {edge, mr}
#             self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
#             self.norm = 'batch'  # batch or instance normalization {batch, instance}
#             self.bias = True  # bias of conv layer True or False
#             self.dropout = 0.0  # dropout rate
#             self.use_dilation = True  # use dilated knn or not
#             self.epsilon = 0.2  # stochastic epsilon for gcn
#             self.use_stochastic = False  # stochastic for gcn, True or False
#             self.drop_path = drop_path_rate
#             self.blocks = [2, 2, 6, 2]  # number of basic blocks in the backbone
#             self.channels = [64, 128, 256, 512]  # number of channels of deep features
#             self.n_classes = num_classes  # Dimension of out_channels
#             self.emb_dims = 1024  # Dimension of embeddings
#             self.bilinear = True
#
#     opt = OptInit(**kwargs)
#     model = DeepGCN(opt)
#     model.default_cfg = default_cfgs['vig_224_gelu']
#     return model
#
#
# @register_model
# def pvig_s_224_gelu(pretrained=False, **kwargs):
#     class OptInit:
#         def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
#             self.k = 9  # neighbor num (default:9)
#             self.conv = 'mr'  # graph conv layer {edge, mr}
#             self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
#             self.norm = 'batch'  # batch or instance normalization {batch, instance}
#             self.bias = True  # bias of conv layer True or False
#             self.dropout = 0.0  # dropout rate
#             self.use_dilation = True  # use dilated knn or not
#             self.epsilon = 0.2  # stochastic epsilon for gcn
#             self.use_stochastic = False  # stochastic for gcn, True or False
#             self.drop_path = drop_path_rate
#             self.blocks = [2, 2, 6, 2]  # number of basic blocks in the backbone
#             self.channels = [80, 160, 400, 640]  # number of channels of deep features
#             self.n_classes = num_classes  # Dimension of out_channels
#             self.emb_dims = 1024  # Dimension of embeddings
#
#     opt = OptInit(**kwargs)
#     model = DeepGCN(opt)
#     model.default_cfg = default_cfgs['vig_224_gelu']
#     return model
#
#
# @register_model
# def pvig_m_224_gelu(pretrained=False, **kwargs):
#     class OptInit:
#         def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
#             self.k = 9  # neighbor num (default:9)
#             self.conv = 'mr'  # graph conv layer {edge, mr}
#             self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
#             self.norm = 'batch'  # batch or instance normalization {batch, instance}
#             self.bias = True  # bias of conv layer True or False
#             self.dropout = 0.0  # dropout rate
#             self.use_dilation = True  # use dilated knn or not
#             self.epsilon = 0.2  # stochastic epsilon for gcn
#             self.use_stochastic = False  # stochastic for gcn, True or False
#             self.drop_path = drop_path_rate
#             self.blocks = [2, 2, 16, 2]  # number of basic blocks in the backbone
#             self.channels = [96, 192, 384, 768]  # number of channels of deep features
#             self.n_classes = num_classes  # Dimension of out_channels
#             self.emb_dims = 1024  # Dimension of embeddings
#
#     opt = OptInit(**kwargs)
#     model = DeepGCN(opt)
#     model.default_cfg = default_cfgs['vig_224_gelu']
#     return model
#
#
# @register_model
# def pvig_b_224_gelu(pretrained=False, **kwargs):
#     class OptInit:
#         def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
#             self.k = 9  # neighbor num (default:9)
#             self.conv = 'mr'  # graph conv layer {edge, mr}
#             self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
#             self.norm = 'batch'  # batch or instance normalization {batch, instance}
#             self.bias = True  # bias of conv layer True or False
#             self.dropout = 0.0  # dropout rate
#             self.use_dilation = True  # use dilated knn or not
#             self.epsilon = 0.2  # stochastic epsilon for gcn
#             self.use_stochastic = False  # stochastic for gcn, True or False
#             self.drop_path = drop_path_rate
#             self.blocks = [2, 2, 18, 2]  # number of basic blocks in the backbone
#             self.channels = [128, 256, 512, 1024]  # number of channels of deep features
#             self.n_classes = num_classes  # Dimension of out_channels
#             self.emb_dims = 1024  # Dimension of embeddings
#
#     opt = OptInit(**kwargs)
#     model = DeepGCN(opt)
#     model.default_cfg = default_cfgs['vig_b_224_gelu']
#     return model