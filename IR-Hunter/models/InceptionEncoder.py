import torch.nn as nn
import torch

g_prediction_type = "UnKnown"  # "Congestion" or "DRC"


def generation_init_weights(module):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):

            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    module.apply(init_func)


class create_encoder_single_conv(nn.Module):
    def __init__(self, in_chs, out_chs, kernel):
        super().__init__()
        global g_prediction_type
        assert kernel % 2 == 1
        # if ("DRC" == g_prediction_type):
        self.single_Conv = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=kernel, padding=(kernel - 1) // 2),
                                         nn.InstanceNorm2d(out_chs, affine=True),
                                         nn.PReLU(num_parameters=out_chs))  # nn.LeakyReLU(0.2, inplace=True))
        # elif ("Congestion" == g_prediction_type):
        # self.single_Conv = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=kernel, padding=(kernel - 1) // 2),
        #                                     nn.BatchNorm2d(out_chs),
        #                                     nn.PReLU(num_parameters=out_chs))
        # else:
        #     print("ERROR on prediction type!")

    def forward(self, x):
        out = self.single_Conv(x)
        return out


class EncoderInceptionModuleSignle(nn.Module):
    def __init__(self, channels):
        super().__init__()
        bn_ch = channels // 2
        self.bottleneck = create_encoder_single_conv(channels, bn_ch, 1)
        self.conv1 = create_encoder_single_conv(bn_ch, channels, 1)
        self.conv3 = create_encoder_single_conv(bn_ch, channels, 3)
        self.conv5 = create_encoder_single_conv(bn_ch, channels, 5)
        self.conv7 = create_encoder_single_conv(bn_ch, channels, 7)
        self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(5, stride=1, padding=2)

    def forward(self, x):
        bn = self.bottleneck(x)
        conv1 = self.conv1(bn)
        conv3 = self.conv3(bn)
        conv5 = self.conv5(bn)
        conv7 = self.conv7(bn)
        pool3 = self.pool3(x)
        pool5 = self.pool5(x)
        out = conv1 + conv3 + conv5 + conv7 + pool3 + pool5
        return out


class EncoderModule(nn.Module):
    def __init__(self, chs, repeat_num, use_inception):
        super().__init__()
        if use_inception:
            layers = [EncoderInceptionModuleSignle(chs) for i in range(repeat_num)]
        else:
            layers = [create_encoder_single_conv(chs, chs, 3) for i in range(repeat_num)]
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class IncepEncoder(nn.Module):

    def __init__(self, use_inception, repeat_per_module,middel_layer_size=512):
        global g_prediction_type
        super().__init__()
        # g_prediction_type = prediction_type
        self.encoderPart = EncoderModule(middel_layer_size, repeat_per_module, use_inception)

    def forward(self, x):
        out = self.encoderPart(x)
        return out

    def init_weights(self):
        """Initialize the weights."""
        generation_init_weights(self)
