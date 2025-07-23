import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm
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
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys, err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [key for key in all_missing_keys if 'num_batches_tracked' not in key]

    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: ' +
                       f'{", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    return missing_keys


class powernet_change(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, bilinear=False, **kwargs):
        super(powernet_change, self).__init__()
        self.out_channels = out_channels

        # 卷积层定义保持不变
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU())

        # 添加自适应平均池化层固定特征图尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # 输出固定4x4特征图

        # 计算固定特征维度：8通道 * 4 * 4 = 128
        self.fc = nn.Linear(8 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # 输入维度处理保持不变
        x_in = x[:, :, :self.out_channels, :, :]
        x_2d = x.mean(dim=2)  # 假设输入x形状为(batch, 1, 4, H, W)

        # 前向传播流程
        out = self.layer1(x_2d)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # 添加自适应池化统一特征尺寸
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)

        # 全连接层
        out = F.relu(self.fc(out))
        out = self.fc2(out)

        # 空间扩展保持不变
        H, W = x_2d.size(2), x_2d.size(3)
        out_spatial = out.view(out.size(0), 1, 1, 1).expand(-1, self.out_channels, H, W)
        logits = x_in.squeeze(1) * out_spatial
        final = torch.sum(logits, dim=1)
        return final

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
            raise TypeError(f'"pretrained" must be a str or None. But received {type(pretrained)}.')
