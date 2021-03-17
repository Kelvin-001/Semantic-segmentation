import torch
import torch.nn as nn
import torch.nn.functional as F
import modeling.snorm as sn
import modeling.weightstd as ws
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial 
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c 
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
    '''
    def __init__(self, in_channels, key_channels, scale=1):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU()
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU()
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU()
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)   

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self, in_channels, key_channels, scale=1):
        super(ObjectAttentionBlock2D, self).__init__(in_channels, key_channels, scale)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.1):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels, scale)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output

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


class UNet3p_res_ocr_SNws(nn.Module):
    def __init__(self, n_channels, n_filters, n_class, using_movavg, using_bn):
        super(UNet3p_res_ocr_SNws, self).__init__()


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
        
        
        # OCR
        ocr_mid_channels = 512
        ocr_key_channels = 256

        self.aux_head = nn.Sequential(
            nn.Conv2d(5 * n_filters, 5 * n_filters, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(5 * n_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(5 * n_filters, n_class, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(5 * n_filters, ocr_mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True)
        )
        self.ocr_gather_head = SpatialGather_Module(n_class)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, n_class, kernel_size=1, stride=1, padding=0, bias=True)

        


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
        
        # up3
        convu1_d1 = self.convd1_cat(convd1)
        convu1_u2 = self.convu_cat(self.upSample(convu2))
        convu1_u3 = self.convu_cat(self.upSample4(convu3))
        convu1_u4 = self.convu_cat(self.upSample8(convu4))
        convu1_5 = self.conv5_cat(self.upSample16(conv5))
        #convu1_a = self.convd1_cat(self.upSample16(conva))
        convu1 = self.conv_fusion(torch.cat((convu1_d1, convu1_u2, convu1_u3, convu1_u4,convu1_5), 1))
        
        # ocr
        out_aux = self.aux_head(convu1)
        # compute contrast feature
        feats = self.conv3x3_ocr(convu1)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        output = self.cls_head(feats)

        # out_aux_seg.append(out_aux)
        # out_aux_seg.append(out)

        # return out_aux_seg
        return output
    
