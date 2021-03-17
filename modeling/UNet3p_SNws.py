import torch
import torch.nn as nn
import torch.nn.functional as F
import modeling.snorm as sn
import modeling.weightstd as ws

class ASPP(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(ASPP, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self.init_weight()
    
    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        
        return self.relu(x)
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, sn.SwitchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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


class UNet3p_SNws(nn.Module):
    def __init__(self, n_channels, n_filters, n_class, using_movavg, using_bn):
        super(UNet3p_SNws, self).__init__()


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
        


        ## ------------------unet3+_decoder------------------
        
        self.Cat_channels = n_filters
        self.Fusion_channels = 5 * n_filters
        
        # up
        self.convd1_cat = ConvSnRelu(n_filters, self.Cat_channels, using_movavg, using_bn)
        self.convd2_cat = ConvSnRelu(n_filters*2, self.Cat_channels, using_movavg, using_bn)
        self.convd3_cat = ConvSnRelu(n_filters*4, self.Cat_channels, using_movavg, using_bn)
        self.convd4_cat = ConvSnRelu(n_filters*8, self.Cat_channels, using_movavg, using_bn)
        self.conv5_cat = ConvSnRelu(n_filters*16, self.Cat_channels, using_movavg, using_bn)
        self.convu_cat = ConvSnRelu(self.Fusion_channels, self.Cat_channels, using_movavg, using_bn)
        self.conv_fusion = ConvSnRelu(self.Fusion_channels, self.Fusion_channels, using_movavg, using_bn)

        self.output_seg_map = nn.Conv2d(self.Fusion_channels, n_class, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        ## ------------------encoder------------------
        # down1
        x = self.convd1_1(x)
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
        convu4 = self.conv_fusion(torch.cat((convu4_d1, convu4_d2, convu4_d3, convu4_d4,convu4_5), 1))
        
        # up3
        convu3_d1 = self.convd1_cat(self.maxPool4(convd1))
        convu3_d2 = self.convd2_cat(self.maxPool(convd2))
        convu3_d3 = self.convd3_cat(convd3)
        convu3_u4 = self.convu_cat(self.upSample(convu4))
        convu3_5 = self.conv5_cat(self.upSample4(conv5))
        convu3 = self.conv_fusion(torch.cat((convu3_d1, convu3_d2, convu3_d3, convu3_u4,convu3_5), 1))
        
        # up2
        convu2_d1 = self.convd1_cat(self.maxPool(convd1))
        convu2_d2 = self.convd2_cat(convd2)
        convu2_u3 = self.convu_cat(self.upSample(convu3))
        convu2_u4 = self.convu_cat(self.upSample4(convu4))
        convu2_5 = self.conv5_cat(self.upSample8(conv5))
        convu2 = self.conv_fusion(torch.cat((convu2_d1, convu2_d2, convu2_u3, convu2_u4,convu2_5), 1))
        
        # up3
        convu1_d1 = self.convd1_cat(convd1)
        convu1_u2 = self.convu_cat(self.upSample(convu2))
        convu1_u3 = self.convu_cat(self.upSample4(convu3))
        convu1_u4 = self.convu_cat(self.upSample8(convu4))
        convu1_5 = self.conv5_cat(self.upSample16(conv5))
        convu1 = self.conv_fusion(torch.cat((convu1_d1, convu1_u2, convu1_u3, convu1_u4,convu1_5), 1))
        
        output = self.output_seg_map(convu1)

        return output
    
