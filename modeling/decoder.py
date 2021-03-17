import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn' or backbone == 'resnest':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        #self.upSample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        #self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn1 = BatchNorm(256)
        #self.relu = nn.ReLU()
        #self.dropout1 = nn.Dropout(0.5)
        #self.conv2 = nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn2 = BatchNorm(128)
        #self.conv3 = nn.Conv2d(224, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn3 = BatchNorm(64)
        #self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #self.dropout2 = nn.Dropout(0.1)
        #self.conv5 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
        #                               nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
        #                               nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        #x = self.upSample(x)
        #x = torch.cat((x, feat_block2), dim=1)
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu(x)
        #x = self.dropout1(x)

        #x = self.upSample(x)
        #x = torch.cat((x, low_level_feat), dim=1)
        #x = self.conv2(x)
        #x = self.bn2(x)
        #x = self.relu(x)
        #x = self.dropout1(x)

        #x = self.upSample(x)
        #x = torch.cat((x, feat_conv1, feat_conv2), dim=1)
        #x = self.conv3(x)
        #x = self.bn3(x)
        #x = self.relu(x)
        #x = self.dropout1(x)

        #x = self.conv4(x)
        #x = self.bn3(x)
        #x = self.relu(x)
        #x = self.dropout2(x)
        #x = self.conv5(x)
        
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)
