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

def ConvSnRelu(in_channels, out_channels, using_movavg=True, using_bn=True, last_gamma=False, kernel_size=(3, 3), stride=1, padding=1):
    conv = ws.Conv2dws(in_channels, out_channels, kernel_size, stride, padding)
    norm = sn.SwitchNorm2d(out_channels, using_movavg=using_movavg, using_bn=using_bn, last_gamma=last_gamma)
    relu = nn.ReLU()
    return nn.Sequential(conv, norm, relu)

def CropConcat(a, b):
    diffY = a.size()[2] - b.size()[2]
    diffX = a.size()[3] - b.size()[3]
    a = F.pad(a, (-(diffX//2), -(diffX-diffX//2), -(diffY//2), -(diffY-diffY//2)))
    return torch.cat((a, b), dim=1)


class UNet_SNws(nn.Module):
    def __init__(self, n_channels, n_filters, n_class, using_movavg, using_bn):
        super(UNet_SNws, self).__init__()

        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, ceil_mode=True)
        self.upSample = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False)

        # down1
        self.conv1_1 = ConvSnRelu(n_channels, n_filters, using_movavg, using_bn)
        self.conv1_2 = ConvSnRelu(n_filters, n_filters, using_movavg, using_bn)

        # down2
        self.conv2_1 = ConvSnRelu(n_filters, n_filters*2, using_movavg, using_bn)
        self.conv2_2 = ConvSnRelu(n_filters*2, n_filters*2, using_movavg, using_bn)

        # down3
        self.conv3_1 = ConvSnRelu(n_filters*2, n_filters*4, using_movavg, using_bn)
        self.conv3_2 = ConvSnRelu(n_filters*4, n_filters*4, using_movavg, using_bn)

        # center
        self.conv4_1 = ConvSnRelu(n_filters*4, n_filters*8, using_movavg, using_bn)
        self.conv4_2 = ConvSnRelu(n_filters*8, n_filters*8, using_movavg, using_bn)

        ## add two
        self.conv_d1 = ConvSnRelu(n_filters*8, n_filters*16, using_movavg, using_bn)
        self.conv_d2 = ConvSnRelu(n_filters*16, n_filters*16, using_movavg, using_bn)
        

        self.conv_u1 = ConvSnRelu(n_filters*16, n_filters*8, using_movavg, using_bn, kernel_size=(1, 1), padding=0)
        self.conv_u2 = ConvSnRelu(n_filters*8+n_filters*8, n_filters*8, using_movavg, using_bn)
        self.conv_u3 = ConvSnRelu(n_filters*8, n_filters*8, using_movavg, using_bn)

        # up1
        self.conv5_1 = ConvSnRelu(n_filters*8, n_filters*4, using_movavg, using_bn, kernel_size=(1, 1), padding=0)
        # Crop + concat step between these 2
        self.conv5_2 = ConvSnRelu(n_filters*4+n_filters*4, n_filters*4, using_movavg, using_bn)
        self.conv5_3 = ConvSnRelu(n_filters*4, n_filters*4, using_movavg, using_bn)

        # up2
        self.conv6_1 = ConvSnRelu(n_filters*4, n_filters*2, using_movavg, using_bn, kernel_size=(1, 1), padding=0)
        # Crop + concat step between these 2
        self.conv6_2 = ConvSnRelu(n_filters*2+n_filters*2, n_filters*2, using_movavg, using_bn)
        self.conv6_3 = ConvSnRelu(n_filters*2, n_filters*2, using_movavg, using_bn)

        # up3
        self.conv7_1 = ConvSnRelu(n_filters*2, n_filters, using_movavg, using_bn, kernel_size=(1, 1), padding=0)
        # Crop + concat step between these 2
        self.conv7_2 = ConvSnRelu(n_filters+n_filters, n_filters, using_movavg, using_bn)
        self.conv7_3 = ConvSnRelu(n_filters, n_filters, using_movavg, using_bn)

        # 1x1 convolution at the last layer
        self.output_seg_map = nn.Conv2d(n_filters, n_class, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        # down1
        x = self.conv1_1(x)
        conv1 = self.conv1_2(x)
        x = self.maxPool(conv1)

        # down2
        x = self.conv2_1(x)
        conv2 = self.conv2_2(x)
        x = self.maxPool(conv2)

        # down3
        x = self.conv3_1(x)
        conv3 = self.conv3_2(x)
        x = self.maxPool(conv3)

        # center
        x = self.conv4_1(x)
        #x = self.conv4_2(x)
        conv4 = self.conv4_2(x)      
  
        # add
        x = self.maxPool(conv4)
        x = self.conv_d1(x)
        x = self.conv_d2(x)
        #convd = self.conv_d2(x)

        x = self.upSample(x)
        x = self.conv_u1(x)
        x = CropConcat(conv4, x)
        x = self.conv_u2(x)
        x = self.conv_u3(x)

        # up1
        x = self.upSample(x)
        #x = F.interpolate(x, size=(128,128), mode='bilinear',align_corners=True)
        x = self.conv5_1(x)
        x = CropConcat(conv3, x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)

        # up2
        x = self.upSample(x)
        x = self.conv6_1(x)
        x = CropConcat(conv2, x)
        x = self.conv6_2(x)
        x = self.conv6_3(x)

        # up3
        x = self.upSample(x)
        x = self.conv7_1(x)
        x = CropConcat(conv1, x)
        x = self.conv7_2(x)
        x = self.conv7_3(x)

        output = self.output_seg_map(x)

        return output
    
