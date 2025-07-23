from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm


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


# 用于裁剪张量，使其尺寸与目标张量一致（只保留前面的部分）
def crop_tensor(x, target):
    # x, target 均为 (N, C, H, W)
    _, _, H_target, W_target = target.size()
    return x[:, :, :H_target, :W_target]


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=5, padding=2)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        # x: (N, 1, H, W)
        x0 = self.relu(self.conv1(x))        # (N, 64, H, W)
        x1 = self.max1(x0)                   # (N, 64, ceil(H/2), ceil(W/2))
        x1 = self.relu(self.conv2(x1))         # (N, 32, ceil(H/2), ceil(W/2))
        x2 = self.max2(x1)                   # (N, 32, ceil(H/4), ceil(W/4))
        x2 = self.relu(self.conv3(x2))         # (N, 16, ceil(H/4), ceil(W/4))
        x3 = self.max3(x2)                   # (N, 16, ceil(H/8), ceil(W/8))
        return (x0, x1, x2, x3)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv0 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=7, padding=3)
        self.relu = nn.ReLU(inplace=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=7, padding=3)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=3, padding=1)

    def forward(self, vals):
        # 输入 vals 为一个元组：(skip0, skip1, skip2, deep)
        skip0, skip1, skip2, deep = vals

        # 第一阶段
        x1 = self.conv0(deep)
        x1 = self.relu(x1)
        x1 = self.upsample1(x1)
        x1 = crop_tensor(x1, skip2)
        x1 = torch.cat([x1, skip2], dim=1)  # (N, 16+16, H2, W2)

        # 第二阶段
        x2 = self.conv1(x1)
        x2 = self.relu(x2)
        x2 = self.upsample2(x2)
        x2 = crop_tensor(x2, skip1)
        x2 = torch.cat([x2, skip1], dim=1)  # (N, 32+32, H1, W1)

        # 第三阶段
        x3 = self.conv2(x2)
        x3 = self.relu(x3)
        x3 = self.upsample3(x3)
        x3 = crop_tensor(x3, skip0)
        x3 = torch.cat([x3, skip0], dim=1)  # (N, 64+64, H, W)

        # 最后输出
        x4 = self.conv3(x3)
        x4 = self.relu(x4)
        return x4


class LSLayer(nn.Module):
    def __init__(self):
        super(LSLayer, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.Linear(256, 128)
        self.t_fc1 = nn.LazyLinear(64)
        self.t_fc2 = nn.Linear(64, 64)
        self.t_fc3 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(128 + 64, 256)
        self.fc4 = nn.Linear(256, 320)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        t = self.relu(self.t_fc1(t))
        t = self.relu(self.t_fc2(t))
        t = self.relu(self.t_fc3(t))
        x_cat = torch.cat([x, t], dim=1)
        x_cat = self.relu(self.fc3(x_cat))
        x_cat = self.relu(self.fc4(x_cat))
        return x_cat


class IREDGe(nn.Module):   # Autoencoder
    def __init__(self,
                 in_channels=1,
                 out_channels=4,
                 bilinear=False,
                 **kwargs):
        super(IREDGe, self).__init__()
        self.out_channels = out_channels
        self.encoder = Encoder()
        self.ls = LSLayer()
        # 新增 reduce 层，将 LSLayer 输出的 320 通道降到 16 通道
        self.reduce = nn.Conv2d(320, 16, kernel_size=1)
        self.decoder = Decoder()

    def forward(self, x, t=None):
        """
        输入:
          x: 5D 张量 (N, in_channels, D, H, W)
          t: 附加输入 (N, feature_dim)，若未传入，则默认生成全零张量（这里假设 feature_dim 为 10）
        """
        if t is None:
            t = torch.zeros(x.size(0), 10, device=x.device)
        # 1. 截取前 out_channels 个深度切片
        x_in = x[:, :, :self.out_channels, :, :]  # (N, in_channels, out_channels, H, W)
        # 2. 对 x 在深度维求平均，得到 (N, in_channels, H, W)
        x_2d = x.mean(dim=2)
        # 3. Encoder 提取多尺度特征
        ae_out = self.encoder(x_2d)  # 返回 (x0, x1, x2, x3)
        # 4. LSLayer 处理 encoder 最深层特征 x3 与附加输入 t，输出 (N,320)
        ls_vec = self.ls(ae_out[3], t)
        # 5. 将 ls_vec 转换为特征图：先扩展为 (N,320,1,1)，然后插值到与 skip2 相同的空间尺寸
        ls_map = ls_vec.unsqueeze(-1).unsqueeze(-1)  # (N,320,1,1)
        target_size = ae_out[2].shape[2:]  # 目标尺寸取自 skip2
        ls_map = F.interpolate(ls_map, size=target_size, mode='bilinear', align_corners=False)  # (N,320,H_target,W_target)
        ls_map = self.reduce(ls_map)  # (N,16,H_target,W_target)
        # 6. Decoder 使用 encoder 的跳跃连接和 ls_map 生成 logits (N,1,H,W)
        de_out = self.decoder((ae_out[0], ae_out[1], ae_out[2], ls_map))
        # 7. 将 x_in 的 in_channels 维 squeeze 掉，得到 (N, out_channels, H, W)
        logits = x_in.squeeze(1) * de_out
        # 8. 沿通道维求和，得到最终输出 (N, H, W)
        output = torch.sum(logits, dim=1)
        return output

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
