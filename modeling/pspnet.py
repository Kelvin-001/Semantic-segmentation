import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
affine_par = True
import functools
import sys, os

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual      
        out = self.relu_inplace(out)

        return out

class PSPModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle
    
class PSPNet(nn.Module):
    def __init__(self, n_channels, n_filters, num_classes, block = Bottleneck, layers = [3, 4, 23, 3]):
        self.inplanes = 128
        super(PSPNet, self).__init__()
        blocks = [1, 2, 4]
        strides = [1, 2, 2, 1]
        dilations = [1, 1, 1, 2]
        self.conv1 = nn.Conv2d(n_channels, n_filters, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(n_filters * 2)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)

        self.pspmodule = PSPModule(2048, 512)
        self.head = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )				

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.relu1(self.bn1(self.conv1(input)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        # x_dsn = self.dsn(x3)
        x4 = self.layer4(x3)
        #x = self.head(x4)
        x_feat_after_psp = self.pspmodule(x4)
        x = self.head(x_feat_after_psp)
        output = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return output

if __name__ == "__main__":
    import torch
    model = PSPNet(3, 64, 10)
    input = torch.rand(1, 3, 512, 512)
    st1 = model.relu1(model.bn1(model.conv1(input)))
    print(st1.detach().numpy().shape)
    st2 = model.relu2(model.bn2(model.conv2(st1)))
    print(st2.detach().numpy().shape)
    st3 = model.relu3(model.bn3(model.conv3(st2)))
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
    st9 = model.pspmodule(st8)
    print(st9.detach().numpy().shape)
    st10 = model.head(st9)
    print(st10.detach().numpy().shape)
    # output = model(input)
    # print(output.size())
    