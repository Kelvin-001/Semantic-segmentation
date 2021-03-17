import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import modeling.snorm as sn
import modeling.weightstd as ws
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

class ASPP(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(ASPP, self).__init__()
        self.atrous_conv = ws.Conv2dws(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self.init_weight()
    
    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        
        return self.relu(x)
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, ws.Conv2dws):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, sn.SwitchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

def ConvSnRelu(in_channels, out_channels, using_movavg=True, using_bn=True, last_gamma=False, kernel_size=3, stride=1, padding=1):
    conv = ws.Conv2dws(in_channels, out_channels, kernel_size, stride, padding)
    norm = sn.SwitchNorm2d(out_channels, using_movavg=using_movavg, using_bn=using_bn, last_gamma=last_gamma)
    relu = nn.ReLU()
    return nn.Sequential(conv, norm, relu)

def CropConcat(a, b):
    diffY = a.size()[2] - b.size()[2]
    diffX = a.size()[3] - b.size()[3]
    a = F.pad(a, (-(diffX//2), -(diffX-diffX//2), -(diffY//2), -(diffY-diffY//2)))
    return torch.cat((a, b), dim=1)


class UNet3p_res_edge2_aspp_SNws(nn.Module):
    def __init__(self, n_channels, n_filters, n_class, using_movavg, using_bn):
        super(UNet3p_res_edge2_aspp_SNws, self).__init__()


        ## ------------------encoder------------------
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.maxPool4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.maxPool8 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.upSample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upSample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upSample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.upSample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)

        # down1
        self.convd1_1 = ConvSnRelu(n_channels, n_filters, using_movavg, using_bn)
        #self.convd1_2 = ConvSnRelu(n_filters, n_filters, using_movavg, using_bn)
        self.convd1_2 = Resnet_BasicBlock(n_filters, n_filters, stride=1, downsample=None)

        # down2
        self.convd2_1 = ConvSnRelu(n_filters, n_filters*2, using_movavg, using_bn)
        #self.convd2_2 = ConvSnRelu(n_filters*2, n_filters*2, using_movavg, using_bn)
        self.convd2_2 = Resnet_BasicBlock(n_filters*2, n_filters*2, stride=1, downsample=None)

        # down3
        self.convd3_1 = ConvSnRelu(n_filters*2, n_filters*4, using_movavg, using_bn)
        #self.convd3_2 = ConvSnRelu(n_filters*4, n_filters*4, using_movavg, using_bn)
        self.convd3_2 = Resnet_BasicBlock(n_filters*4, n_filters*4, stride=1, downsample=None)

        # down4
        self.convd4_1 = ConvSnRelu(n_filters*4, n_filters*8, using_movavg, using_bn)
        #self.convd4_2 = ConvSnRelu(n_filters*8, n_filters*8, using_movavg, using_bn)
        self.convd4_2 = Resnet_BasicBlock(n_filters*8, n_filters*8, stride=1, downsample=None)

        ## center
        self.conv5_1 = ConvSnRelu(n_filters*8, n_filters*16, using_movavg, using_bn)
        #self.conv5_2 = ConvSnRelu(n_filters*16, n_filters*16, using_movavg, using_bn)
        self.conv5_2 = Resnet_BasicBlock(n_filters*16, n_filters*16, stride=1, downsample=None)
        
        
        ## ------------------edge------------------.
        self.dsn3 = ws.Conv2dws(256, 1, 1)
        self.dsn4 = ws.Conv2dws(512, 1, 1)
        self.dsn5 = ws.Conv2dws(1024, 1, 1)

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
        self.edge_conv = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), 
                                      nn.Conv2d(1,64,1,stride=1,bias=False), 
                                      nn.BatchNorm2d(64),
                                      nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        
        ## ------------------aspp------------------
        
        self.aspp1 = ASPP(320, 64, 1, padding=0, dilation=1, BatchNorm=sn.SwitchNorm2d)    # 0 1    1024  nn.BatchNorm2d
        self.aspp2 = ASPP(320, 64, 3, padding=2, dilation=2, BatchNorm=sn.SwitchNorm2d)    # 2
        self.aspp3 = ASPP(320, 64, 3, padding=3, dilation=3, BatchNorm=sn.SwitchNorm2d)    # 6
        self.aspp4 = ASPP(320, 64, 3, padding=6, dilation=6, BatchNorm=sn.SwitchNorm2d)    # 12
        self.aspp5 = ASPP(320, 64, 3, padding=12, dilation=12, BatchNorm=sn.SwitchNorm2d)    # 18
        #self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), 
        #                                     nn.Conv2d(1024,256,1,stride=1,bias=False), 
        #                                     nn.BatchNorm2d(256),
        #                                     nn.ReLU())

        #self.conva = nn.Sequential(nn.Conv2d(1280, n_filters, 1, bias=False),
        #                           nn.BatchNorm2d(n_filters),
        #                           nn.ReLU(),
        #                           nn.Conv2d(n_filters, n_filters, 3, bias=False)),
        #                           nn.BatchNorm2d(n_filters),
        #                           nn.ReLU())
        self.conva = ConvSnRelu(320 + 64, n_filters, using_movavg, using_bn, kernel_size=1, padding=0)
        #                           ConvSnRelu(n_filters, n_filters, using_movavg, using_bn))
        
        #self.conv1 = nn.Conv2d(1280, 512, 1, bias=False)
        #self.bn1 = nn.BatchNorm2d(512)
        #self.bn1 = sn.SwitchNorm2d(1024)
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.5)


        ## ------------------unet3+_decoder------------------
        
        self.Cat_channels = n_filters
        self.Fusion_channels = 5 * n_filters    # 5 * n_filters
        
        # up
        self.convd1_cat = ConvSnRelu(n_filters, self.Cat_channels, using_movavg, using_bn)
        self.convd2_cat = ConvSnRelu(n_filters*2, self.Cat_channels, using_movavg, using_bn)
        self.convd3_cat = ConvSnRelu(n_filters*4, self.Cat_channels, using_movavg, using_bn)
        self.convd4_cat = ConvSnRelu(n_filters*8, self.Cat_channels, using_movavg, using_bn)
        self.conv5_cat = ConvSnRelu(n_filters*16, self.Cat_channels, using_movavg, using_bn)
        self.convu_cat = ConvSnRelu(self.Fusion_channels, self.Cat_channels, using_movavg, using_bn)
        self.conv_fusion = ConvSnRelu(self.Fusion_channels, self.Fusion_channels, using_movavg, using_bn)

        self.output_seg_map = nn.Conv2d(n_filters, n_class, kernel_size=1, stride=1, padding=0)



    def forward(self, input):
        
        x_size = input.size()
        
        ## ------------------encoder------------------
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
        x = self.maxPool(convd3)

        # down4
        x = self.convd4_1(x)
        #x = self.conv4_2(x)
        convd4 = self.convd4_2(x)      
        
        # center
        x = self.maxPool(convd4)
        x = self.conv5_1(x)
        #x = self.conv5_2(x)
        conv5 = self.conv5_2(x)
        
        ## ------------------unet3+_decoder------------------
        # up4
        convu4_d1 = self.convd1_cat(self.maxPool8(convd1))
        convu4_d2 = self.convd2_cat(self.maxPool4(convd2))
        convu4_d3 = self.convd3_cat(self.maxPool(convd3))
        convu4_d4 = self.convd4_cat(convd4)
        convu4_5 = self.conv5_cat(self.upSample(conv5))
        #convu4_a = self.convd1_cat(self.upSample(conva))
        convu4 = self.conv_fusion(torch.cat((convu4_d1, convu4_d2, convu4_d3, convu4_d4, convu4_5), 1))
        
        # up3
        convu3_d1 = self.convd1_cat(self.maxPool4(convd1))
        convu3_d2 = self.convd2_cat(self.maxPool(convd2))
        convu3_d3 = self.convd3_cat(convd3)
        convu3_u4 = self.convu_cat(self.upSample(convu4))
        convu3_5 = self.conv5_cat(self.upSample4(conv5))
        #convu3_a = self.convd1_cat(self.upSample4(conva))
        convu3 = self.conv_fusion(torch.cat((convu3_d1, convu3_d2, convu3_d3, convu3_u4,convu3_5), 1))
        
        # up2
        convu2_d1 = self.convd1_cat(self.maxPool(convd1))
        convu2_d2 = self.convd2_cat(convd2)
        convu2_u3 = self.convu_cat(self.upSample(convu3))
        convu2_u4 = self.convu_cat(self.upSample4(convu4))
        convu2_5 = self.conv5_cat(self.upSample8(conv5))
        #convu2_a = self.convd1_cat(self.upSample8(conva))
        convu2 = self.conv_fusion(torch.cat((convu2_d1, convu2_d2, convu2_u3, convu2_u4,convu2_5), 1))
        
        # up1
        convu1_d1 = self.convd1_cat(convd1)
        convu1_u2 = self.convu_cat(self.upSample(convu2))
        convu1_u3 = self.convu_cat(self.upSample4(convu3))
        convu1_u4 = self.convu_cat(self.upSample8(convu4))
        convu1_5 = self.conv5_cat(self.upSample16(conv5))
        #convu1_a = self.convd1_cat(self.upSample16(conva))
        convu1 = self.conv_fusion(torch.cat((convu1_d1, convu1_u2, convu1_u3, convu1_u4,convu1_5), 1))

        
        ## ------------------edge------------------        
        s3 = self.dsn3(convd3)
        s4 = self.dsn4(convd4)
        s5 = self.dsn5(conv5)

        im_arr = input.cpu().numpy().transpose((0,2,3,1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i],10,100)
        canny = torch.from_numpy(canny).cuda().float()
        
        # 1
        c1 = convd1
        d1 = self.d1(c1)
        d1 = self.maxPool4(d1)
        gate1 = self.gate1(d1, s3)
        # 2
        c2 = self.res2(gate1)
        d2 = self.d2(c2)
        d2 = self.maxPool(d2)
        gate2 = self.gate2(d2, s4)
        # 3
        c3 = self.res3(gate2)
        d3 = self.d3(c3)
        d3 = self.maxPool(d3)
        gate3 = self.gate3(d3, s5)
        
        fuse = self.fuse(gate3)
        fuse = F.interpolate(fuse, x_size[2:], mode='bilinear', align_corners=True)
        
        edge_out = self.sigmoid(fuse)
        cat = torch.cat((edge_out, canny), dim=1)
        cw = self.cw(cat)
        acts = self.sigmoid(cw)
        
        ## ------------------aspp------------------
        x1 = self.aspp1(convu1)
        x2 = self.aspp2(convu1)
        x3 = self.aspp3(convu1)
        x4 = self.aspp4(convu1)
        x5 = self.aspp5(convu1)
        edge = self.edge_conv(acts)
        edge = F.interpolate(edge, size=x_size[2:], mode='bilinear', align_corners=True)
        #x5 = self.global_avg_pool(conv5)
        #x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x_aspp = torch.cat((x1, x2, x3, x4, x5, edge), dim=1)
        conva = self.conva(x_aspp)
        #x = self.dropout(x)
        
        output = self.output_seg_map(conva)
        

        return output
