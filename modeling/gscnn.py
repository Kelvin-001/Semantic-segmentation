import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import modeling.snorm as sn
import modeling.weightstd as ws
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

def ConvSnRelu(in_channels, out_channels, using_movavg=True, using_bn=True, last_gamma=False, kernel_size=(3, 3), stride=1, padding=1):
    conv = ws.Conv2dws(in_channels, out_channels, kernel_size, stride, padding)
    norm = sn.SwitchNorm2d(out_channels, using_movavg=using_movavg, using_bn=using_bn, last_gamma=last_gamma)
    relu = nn.ReLU()
    return nn.Sequential(conv, norm, relu)

# def CropConcat(a, b):
#     diffY = a.size()[2] - b.size()[2]
#     diffX = a.size()[3] - b.size()[3]
#     a = F.pad(a, (-(diffX//2), -(diffX-diffX//2), -(diffY//2), -(diffY-diffY//2)))
#     return torch.cat((a, b), dim=1)

class Resnet_BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Resnet_BasicBlock, self).__init__()
        self.conv1 = ws.Conv2dws(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = sn.SwitchNorm2d(planes)
        self.conv2 = ws.Conv2dws(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = sn.SwitchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ASPP(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(ASPP, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        
        return self.relu(x)
    
class GatedSpatialConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GatedSpatialConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, 
                                                 False, _pair(0), groups, bias, 'zeros')

        self.gate_conv = nn.Sequential(
            sn.SwitchNorm2d(in_channels+1),
            ws.Conv2dws(in_channels+1, in_channels+1, 1),
            nn.ReLU(), 
            ws.Conv2dws(in_channels+1, 1, 1),
            sn.SwitchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, input_features, gating_features):
        """
        :param input_features:  [NxCxHxW]  featuers comming from the shape branch (canny branch).
        :param gating_features: [Nx1xHxW] features comming from the texture branch (resnet). Only one channel feature map.
        :return:
        """
        alphas = self.gate_conv(torch.cat([input_features, gating_features], dim=1))

        input_features = (input_features * (alphas + 1)) 
        return F.conv2d(input_features, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class GSCNN(nn.Module):
    
    def __init__(self, n_channels, n_filters, n_class, using_movavg, using_bn):
        
        super(GSCNN, self).__init__()
        
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        # down1
        self.convd1_1 = ConvSnRelu(n_channels, n_filters, using_movavg, using_bn)
        self.convd1_2 = ConvSnRelu(n_filters, n_filters, using_movavg, using_bn)

        # down2
        self.convd2_1 = ConvSnRelu(n_filters, n_filters*2, using_movavg, using_bn)
        self.convd2_2 = ConvSnRelu(n_filters*2, n_filters*2, using_movavg, using_bn)

        # down3
        self.convd3_1 = ConvSnRelu(n_filters*2, n_filters*4, using_movavg, using_bn)
        self.convd3_2 = ConvSnRelu(n_filters*4, n_filters*4, using_movavg, using_bn)

        # down4
        self.convd4_1 = ConvSnRelu(n_filters*4, n_filters*8, using_movavg, using_bn)
        self.convd4_2 = ConvSnRelu(n_filters*8, n_filters*8, using_movavg, using_bn)

        ## center
        self.conv5_1 = ConvSnRelu(n_filters*8, n_filters*16, using_movavg, using_bn)
        self.conv5_2 = ConvSnRelu(n_filters*16, n_filters*16, using_movavg, using_bn)
        
        # down6
        #self.convd6_1 = ConvSnRelu(n_filters*16, n_filters*32, using_movavg, using_bn)
        #self.convd6_2 = ConvSnRelu(n_filters*32, n_filters*32, using_movavg, using_bn)

        ## center
        #self.conv7_1 = ConvSnRelu(n_filters*32, n_filters*64, using_movavg, using_bn)
        #self.conv7_2 = ConvSnRelu(n_filters*64, n_filters*64, using_movavg, using_bn)
        
        self.dsn1 = ws.Conv2dws(64, 1, 1)
        self.dsn3 = ws.Conv2dws(256, 1, 1)
        self.dsn4 = ws.Conv2dws(512, 1, 1)
        self.dsn5 = ws.Conv2dws(1024, 1, 1)

        self.res1 = Resnet_BasicBlock(64, 64, stride=1, downsample=None)
        self.d1 = ws.Conv2dws(64, 32, 1)
        self.res2 = Resnet_BasicBlock(32, 32, stride=1, downsample=None)
        self.d2 = ws.Conv2dws(32, 16, 1)
        self.res3 = Resnet_BasicBlock(16, 16, stride=1, downsample=None)
        self.d3 = ws.Conv2dws(16, 8, 1)
        self.fuse = ws.Conv2dws(8, 1, kernel_size=1, padding=0, bias=False)
        self.cw = ws.Conv2dws(2, 1, kernel_size=1, padding=0, bias=False)
        
        self.gate1 = GatedSpatialConv2d(32, 32)
        self.gate2 = GatedSpatialConv2d(16, 16)
        self.gate3 = GatedSpatialConv2d(8, 8)
        
        ## ------------------aspp------------------
        self.aspp1 = ASPP(1024, 256, 1, padding=0, dilation=1, BatchNorm=nn.BatchNorm2d)    # 0 1    1024  nn.BatchNorm2d
        self.aspp2 = ASPP(1024, 256, 3, padding=6, dilation=6, BatchNorm=nn.BatchNorm2d)    # 2
        self.aspp3 = ASPP(1024, 256, 3, padding=12, dilation=12, BatchNorm=nn.BatchNorm2d)    # 6
        self.aspp4 = ASPP(1024, 256, 3, padding=18, dilation=18, BatchNorm=nn.BatchNorm2d)    # 12
      
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), 
                                            nn.Conv2d(1024,256,1,stride=1,bias=False), 
                                            nn.BatchNorm2d(256),
                                            nn.ReLU())
        self.edge_conv = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), 
                                      nn.Conv2d(1,256,1,stride=1,bias=False), 
                                      nn.BatchNorm2d(256),
                                      nn.ReLU())
        
        self.bot_aspp = ws.Conv2dws(1280 + 256, 256, kernel_size=1, bias=False)
            # ConvSnRelu(1536, 256, using_movavg=1, using_bn=1, kernel_size=1, padding=0)
        self.bot_fine = ws.Conv2dws(128, 48, kernel_size=1, bias=False)
        self.final_seg = nn.Sequential(ws.Conv2dws(256 + 48, 256, kernel_size=3, padding=1, bias=False),
                                      sn.SwitchNorm2d(256),
                                      nn.ReLU(inplace=True),
                                      ws.Conv2dws(256, 256, kernel_size=3, padding=1, bias=False),
                                      sn.SwitchNorm2d(256),
                                      nn.ReLU(inplace=True),
                                      ws.Conv2dws(256, n_class, kernel_size=1, bias=False))

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input, gts=None):
    
        x_size = input.size()
        
        # down1
        x = self.convd1_1(input)
        convd1 = self.convd1_2(x)
        x = self.maxPool(convd1)
        
        # down2
        x = self.convd2_1(x)
        convd2 = self.convd2_2(x)
        x = self.maxPool(convd2)
        
        # down3
        x = self.convd3_1(x)
        convd3 = self.convd3_2(x)
        # x = self.maxPool(convd3)
        
        # down4
        x = self.convd4_1(x)
        convd4 = self.convd4_2(x)
        # x = self.maxPool(convd4)

        # down5
        x = self.conv5_1(x)
        conv5 = self.conv5_2(x)
        # x = self.maxPool(convd5)

        # down6
        #x = self.convd6_1(x)
        #convd6 = self.convd6_2(x)
        # x = self.maxPool(convd6)
        
        # center
        #x = self.conv7_1(x)
        #conv7 = self.conv7_2(x)
        
        m1f = convd1
        #m1f = F.interpolate(convd1, x_size[2:], mode='bilinear', align_corners=True)
        s3 = F.interpolate(self.dsn3(convd3), x_size[2:], mode='bilinear', align_corners=True)
        s4 = F.interpolate(self.dsn4(convd4), x_size[2:], mode='bilinear', align_corners=True)
        s5 = F.interpolate(self.dsn5(conv5), x_size[2:], mode='bilinear', align_corners=True)

        im_arr = input.cpu().numpy().transpose((0,2,3,1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i],10,100)
        canny = torch.from_numpy(canny).cuda().float()
        
        # 1
        cs = self.res1(m1f)
        #cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)
        cs = self.d1(cs)
        cs = self.gate1(cs, s3)
        # 2
        cs = self.res2(cs)
        #cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)
        cs = self.d2(cs)
        cs = self.gate2(cs, s4)
        # 3
        cs = self.res3(cs)
        #cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)
        cs = self.d3(cs)
        cs = self.gate3(cs, s5)
        
        cs = self.fuse(cs)
        #cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)
        
        edge_out = self.sigmoid(cs)
        cat = torch.cat((edge_out, canny), dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)

        # aspp
        #conv7_f = F.interpolate(conv7, size=x_size[2:], mode='bilinear', align_corners=True)
        x1 = self.aspp1(conv5)
        x2 = self.aspp2(conv5)
        x3 = self.aspp3(conv5)
        x4 = self.aspp4(conv5)
        x5 = self.global_avg_pool(conv5)
        x5 = F.interpolate(x5, size=conv5.size()[2:], mode='bilinear', align_corners=True)
        edge = self.edge_conv(acts)
        edge = F.interpolate(edge, size=conv5.size()[2:],mode='bilinear',align_corners=True)
        #print(x1.shape, x5.shape, edge.shape)
        #import sys
        #sys.exit()
        x_aspp = torch.cat((x1, x2, x3, x4, x5, edge), dim = 1)
        
        dec0_up = self.bot_aspp(x_aspp)
        dec0_fine = self.bot_fine(convd2)
        dec0_up = F.interpolate(dec0_up, convd2.size()[2:], mode='bilinear', align_corners=True)
        dec0 = torch.cat([dec0_fine, dec0_up], dim = 1)
        dec1 = self.final_seg(dec0)
        seg_out = F.interpolate(dec1, x_size[2:], mode='bilinear') 
        
        # if self.training:
        #     return self.criterion((seg_out, edge_out), gts)              
        # else:
        #     return seg_out, edge_out
        return seg_out
        
        
        
        
        
        
