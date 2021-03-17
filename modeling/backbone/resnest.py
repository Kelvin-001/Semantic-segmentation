import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair

# import torch.utils.model_zoo as model_zoo
# from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

"""Split-Attention"""

class SplAtConv2d(Module):
    """Split-Attention Conv2d
    基数cardinality=groups=1 groups对应nn.conv2d的一个参数，即特征层内的cardinal组数
    基数radix = 2  用于SplAtConv2d block中的特征通道数的放大倍数，即cardinal组内split组数
    reduction_factor =4 缩放系数用于fc2和fc3之间减少参数量
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True, radix=2, reduction_factor=4,
                 norm_layer=None, **kwargs):
        super(SplAtConv2d, self).__init__()
        # padding=1 => (1, 1)
        padding = _pair(padding)
        # reduction_factor主要用于减少三组卷积的通道数，进而减少网络的参数量
        # inter_channels 对应fc1层的输出通道数 (64*2//4, 32)=>32
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        # 注意这里使用了深度可分离卷积 groups != 1，实现对不同radix组的特征层进行分离的卷积操作
        self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                           groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        # [1,64,h,w] = [1,128,h,w]
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        x = self.relu(x)

        # rchannel通道数量
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            # [1, 128, h, w] = [[1,64,h,w], [1,64,h,w]]
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel//self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel//self.radix, dim=1)
            # [[1,64,h,w], [1,64,h,w]] => [1,64,h,w]
            gap = sum(splited) 
        else:
            gap = x
        # [1,64,h,w] => [1, 64, 1, 1]
        gap = F.adaptive_avg_pool2d(gap, 1)
        # [1, 64, 1, 1] => [1, 32, 1, 1]
        gap = self.fc1(gap)
        
        if self.use_bn:
            gap = self.bn1(gap)

        gap = self.relu(gap)
        # [1, 32, 1, 1] => [1, 128, 1, 1]

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        
        # attens [[1,64,1,1], [1,64,1,1]]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel//self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel//self.radix, dim=1)
            # [1,64,1,1]*[1,64,h,w] => [1,64,h,w]
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        # contiguous()这个函数，把tensor变成在内存中连续分布的形式
        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            # [1, 128, 1, 1] => [1, 2, 1, 64]
            # 分组进行softmax操作
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            # 对radix维度进行softmax操作
            x = F.softmax(x, dim=1)
            # [1, 2, 1, 64] => [1, 128]
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x
    
    
"""ResNet variants"""

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)

class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False, norm_layer=None):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.conv2 = SplAtConv2d(
            group_width, group_width, kernel_size=3,
            stride=stride, padding=dilation,
            dilation=dilation, groups=cardinality, bias=False,
            radix=radix, norm_layer=norm_layer)

        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        
        # if self.radix == 0:
        #     out = self.bn2(out)
        #     out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# class ResNet(nn.Module):
#     """ResNet Variants

#     Parameters
#     ----------
#     block : Block
#         Class for the residual block. Options are BasicBlockV1, BottleneckV1.
#     layers : list of int
#         Numbers of layers in each block
#     classes : int, default 1000
#         Number of classification classes.
#     dilated : bool, default False
#         Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
#         typically used in Semantic Segmentation.
#     norm_layer : object
#         Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
#         for Synchronized Cross-GPU BachNormalization).
#     """
#     # def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True):
#     def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64,
#                  num_classes=10, dilated=False, dilation=1,
#                  deep_stem=False, stem_width=64, avg_down=False,
#                  avd=False, avd_first=False,
#                  final_drop=0.0, norm_layer=nn.BatchNorm2d):
        
#         self.cardinality = groups
#         self.bottleneck_width = bottleneck_width
#         # ResNet-D params
#         self.inplanes = stem_width*2 if deep_stem else 64
#         self.avg_down = avg_down
#         # ResNeSt params
#         self.radix = radix
#         self.avd = avd
#         self.avd_first = avd_first

#         super(ResNet, self).__init__()
#         blocks = [1, 2, 4]
    
#         if deep_stem:
#             self.conv1 = nn.Sequential(
#                 nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
#                 norm_layer(stem_width),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
#                 norm_layer(stem_width),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False),
#             )
#         else:
#             self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
#         if dilated or dilation == 4:
#             self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
#                                            dilation=2, norm_layer=norm_layer)
#             # self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
#             #                                 dilation=4, norm_layer=norm_layer)
#             self.layer4 = self._make_layer(block, 512, blocks=blocks, stride=1,
#                                             dilation=4, norm_layer=norm_layer)
#         elif dilation==2:
#             self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
#                                            dilation=1, norm_layer=norm_layer)
#             # self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
#             #                                 dilation=2, norm_layer=norm_layer)
#             self.layer4 = self._make_layer(block, 512, blocks=blocks, stride=1,
#                                             dilation=4, norm_layer=norm_layer)
#         else:
#             self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
#                                            norm_layer=norm_layer)
#             # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
#             #                                 norm_layer=norm_layer)
#             self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=2,
#                                             norm_layer=norm_layer)
#         self.avgpool = GlobalAvgPool2d()
#         self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, norm_layer):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, is_first=True):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             down_layers = []
#             if self.avg_down:
#                 if dilation == 1:
#                     down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
#                                                     ceil_mode=True, count_include_pad=False))
#                 else:
#                     down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
#                                                     ceil_mode=True, count_include_pad=False))
#                 down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
#                                              kernel_size=1, stride=1, bias=False))
#             else:
#                 down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
#                                              kernel_size=1, stride=stride, bias=False))
#             down_layers.append(norm_layer(planes * block.expansion))
#             downsample = nn.Sequential(*down_layers)

#         layers = []
#         if dilation == 1 or dilation == 2:
#             layers.append(block(self.inplanes, planes, stride, downsample=downsample,
#                                 radix=self.radix, cardinality=self.cardinality,
#                                 bottleneck_width=self.bottleneck_width,
#                                 avd=self.avd, avd_first=self.avd_first,
#                                 dilation=1, is_first=is_first,
#                                 norm_layer=norm_layer))
#         elif dilation == 4:
#             layers.append(block(self.inplanes, planes, stride, downsample=downsample,
#                                 radix=self.radix, cardinality=self.cardinality,
#                                 bottleneck_width=self.bottleneck_width,
#                                 avd=self.avd, avd_first=self.avd_first,
#                                 dilation=2, is_first=is_first,
#                                 norm_layer=norm_layer))
#         else:
#             raise RuntimeError("=> unknown dilation size: {}".format(dilation))

#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes,
#                                 radix=self.radix, cardinality=self.cardinality,
#                                 bottleneck_width=self.bottleneck_width,
#                                 avd=self.avd, avd_first=self.avd_first,
#                                 dilation=dilation,
#                                 norm_layer=norm_layer))

#         return nn.Sequential(*layers)
    
#     def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, is_first=True):
        
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             down_layers = []
#             if self.avg_down:
#                 if dilation == 1:
#                     down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
#                                                     ceil_mode=True, count_include_pad=False))
#                 else:
#                     down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
#                                                     ceil_mode=True, count_include_pad=False))
#                 down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
#                                               kernel_size=1, stride=1, bias=False))
#             else:
#                 down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
#                                               kernel_size=1, stride=stride, bias=False))
#             down_layers.append(norm_layer(planes * block.expansion))
#             downsample = nn.Sequential(*down_layers)

#         layers = []
#         if dilation == 1 or dilation == 2:
#             layers.append(block(self.inplanes, planes, stride, downsample=downsample,
#                                 radix=self.radix, cardinality=self.cardinality,
#                                 bottleneck_width=self.bottleneck_width,
#                                 avd=self.avd, avd_first=self.avd_first,
#                                 dilation=blocks[0]*dilation, is_first=is_first,
#                                 norm_layer=norm_layer))
#         elif dilation == 4:
#             layers.append(block(self.inplanes, planes, stride, downsample=downsample,
#                                 radix=self.radix, cardinality=self.cardinality,
#                                 bottleneck_width=self.bottleneck_width,
#                                 avd=self.avd, avd_first=self.avd_first,
#                                 dilation=blocks[0]*dilation, is_first=is_first,
#                                 norm_layer=norm_layer))
#         else:
#             raise RuntimeError("=> unknown dilation size: {}".format(dilation))

#         self.inplanes = planes * block.expansion
#         for i in range(1, len(blocks)):
#             layers.append(block(self.inplanes, planes,
#                                 radix=self.radix, cardinality=self.cardinality,
#                                 bottleneck_width=self.bottleneck_width,
#                                 avd=self.avd, avd_first=self.avd_first,
#                                 dilation=blocks[i]*dilation,
#                                 norm_layer=norm_layer))

#         return nn.Sequential(*layers)
    
class ResNet(nn.Module):

    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64,
                 num_classes=10, output_stride = 16,
                 deep_stem=False, stem_width=64, avg_down=False,
                 avd=False, avd_first=False,
                 final_drop=0.0, norm_layer=nn.BatchNorm2d):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], norm_layer=norm_layer)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], norm_layer=norm_layer)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self._init_weight()

        # if pretrained:
        #     self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False), 
                norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first,
                                norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=1, stride=1,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=blocks[0]*dilation, is_first=is_first,
                                norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=blocks[i]*dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x, low_level_feat

        # x = self.avgpool(x)
        # #x = x.view(x.size(0), -1)
        # x = torch.flatten(x, 1)
        # if self.drop:
        #     x = self.drop(x)
        # x = self.fc(x)

        # return x


"""ResNeSt models"""

_url_format = 'https://s3.us-west-1.wasabisys.com/resnest/torch/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

def resnest101(output_stride, norm_layer, pretrained=False):
    # model = ResNet(Bottleneck, [3, 4, 23, 3],
    #                radix=2, groups=1, bottleneck_width=64,
    #                deep_stem=True, stem_width=64, avg_down=True,
    #                avd=True, avd_first=False, **kwargs)
    model = ResNet(Bottleneck, [3, 4, 23, 3], radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True, avd=True, avd_first=False)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest101'], progress=True, check_hash=True))
    return model

if __name__ == "__main__":
    import torch
    import numpy as np
    import cv2
    from PIL import Image
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    input_path = 'E:/ubuntu_Shared_folder/Fengyun Satellite Competition/cloud_data/img_jpg/20200101_0330.jpg'
    img = Image.open(input_path).convert('RGB')
    xoff = 0
    yoff = 0
    # xoff = np.random.randint(0,725)
    # yoff = np.random.randint(0,944)
    img = img.crop((xoff, yoff, xoff + 981, yoff + 981))
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img = np.array(img).astype(np.float32)
    print(img.shape)
    img = img.transpose((2, 0, 1))
    input = torch.from_numpy(img).float()
    input = torch.unsqueeze(input, 0)
    print(input.size())
    model = resnest101(output_stride=16, norm_layer=nn.BatchNorm2d, pretrained=False)
    # model = AlignedXception(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=16)
    
    st1 = model.conv1(input)
    print(st1.detach().numpy().shape)
    st2 = model.bn1(st1)
    print(st2.detach().numpy().shape)
    st3 = model.relu(st2)
    print(st3.detach().numpy().shape)
    st4 = model.maxpool(st3)
    print(st4.detach().numpy().shape)
    st5 = model.layer1(st4)
    print(st5.detach().numpy().shape)
    st6 = model.layer2(st5)
    print(st6.detach().numpy().shape)
    st7 = model.layer3(st6)
    print(st7.detach().numpy().shape)
    st8 = model.layer4(st7)
    print(st8.detach().numpy().shape)
    
    # output, low_level_feat = model(input)
    # print(output.size())
    # print(low_level_feat.size())
