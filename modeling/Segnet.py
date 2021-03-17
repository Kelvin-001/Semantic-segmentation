from torch import nn
from torch.nn import functional as F
import torch as t

def ConvBnRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    norm = nn.BatchNorm2d(out_channels)
    relu = nn.ReLU(inplace=True)
    return nn.Sequential(conv, norm, relu)

class Segnet(nn.Module):
    def __init__(self, n_channels, n_filters, num_classes):
        super(Segnet, self).__init__()

        # Encoder
        self.conv11 = ConvBnRelu(n_channels, n_filters)
        self.conv12 = ConvBnRelu(n_filters, n_filters)

        self.conv21 = ConvBnRelu(n_filters, n_filters * 2)
        self.conv22 = ConvBnRelu(n_filters * 2, n_filters * 2)

        self.conv31 = ConvBnRelu(n_filters * 2, n_filters * 4)
        self.conv32 = ConvBnRelu(n_filters * 4, n_filters * 4)
        self.conv33 = ConvBnRelu(n_filters * 4, n_filters * 4)

        self.conv41 = ConvBnRelu(n_filters * 4, n_filters * 8)
        self.conv42 = ConvBnRelu(n_filters * 8, n_filters * 8)
        self.conv43 = ConvBnRelu(n_filters * 8, n_filters * 8)

        self.conv51 = ConvBnRelu(n_filters * 8, n_filters * 8)
        self.conv52 = ConvBnRelu(n_filters * 8, n_filters * 8)
        self.conv53 = ConvBnRelu(n_filters * 8, n_filters * 8)
        
        # Decoder
        self.conv53d = ConvBnRelu(n_filters * 8, n_filters * 8)
        self.conv52d = ConvBnRelu(n_filters * 8, n_filters * 8)
        self.conv51d = ConvBnRelu(n_filters * 8, n_filters * 8)

        self.conv43d = ConvBnRelu(n_filters * 8, n_filters * 8)
        self.conv42d = ConvBnRelu(n_filters * 8, n_filters * 8)
        self.conv41d = ConvBnRelu(n_filters * 8, n_filters * 4)

        self.conv33d = ConvBnRelu(n_filters * 4, n_filters * 4)
        self.conv32d = ConvBnRelu(n_filters * 4, n_filters * 4)
        self.conv31d = ConvBnRelu(n_filters * 4, n_filters * 2)

        self.conv22d = ConvBnRelu(n_filters * 2, n_filters * 2)
        self.conv21d = ConvBnRelu(n_filters * 2, n_filters)

        self.conv12d = ConvBnRelu(n_filters, n_filters)
        self.conv11d = ConvBnRelu(n_filters, num_classes)

    def forward(self, x):
        # Stage 1
        x11 = self.conv11(x)
        x12 = self.conv12(x11)
        x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        x21 = self.conv21(x1p)
        x22 = self.conv22(x21)
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        x31 = self.conv31(x2p)
        x32 = self.conv32(x31)
        x33 = self.conv33(x32)
        x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=2, return_indices=True)

        # Stage 4
        x41 = self.conv41(x3p)
        x42 = self.conv42(x41)
        x43 = self.conv43(x42)
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2, return_indices=True)

        # Stage 5
        x51 = self.conv51(x4p)
        x52 = self.conv52(x51)
        x53 = self.conv53(x52)
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2, return_indices=True)

        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x53d = self.conv53d(x5d)
        x52d = self.conv52d(x53d)
        x51d = self.conv51d(x52d)

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x43d = self.conv43d(x4d)
        x42d = self.conv42d(x43d)
        x41d = self.conv41d(x42d)

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x33d = self.conv33d(x3d)
        x32d = self.conv32d(x33d)
        x31d = self.conv31d(x32d)

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = self.conv22d(x2d)
        x21d = self.conv21d(x22d)

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = self.conv12d(x1d)
        x11d = self.conv11d(x12d)
        output = x11d

        return output

if __name__ == "__main__":
    import torch
    model = Segnet(14, 64, 10)
    x = torch.rand(1, 14, 512, 512)
    x11 = model.conv11(x)
    x12 = model.conv12(x11)
    x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2, return_indices=True)
    print(x12.detach().numpy().shape)
    print(x11.detach().numpy().shape)
    print(x1p.detach().numpy().shape, id1.detach().numpy().shape)
    x21 = model.conv21(x1p)
    x22 = model.conv22(x21)
    x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)
    print(x21.detach().numpy().shape)
    print(x22.detach().numpy().shape)
    print(x2p.detach().numpy().shape)
    x31 = model.conv31(x2p)
    x32 = model.conv32(x31)
    x33 = model.conv33(x32)
    x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=2, return_indices=True)
    print(x33.detach().numpy().shape)
    print(x3p.detach().numpy().shape)
    x41 = model.conv41(x3p)
    x42 = model.conv42(x41)
    x43 = model.conv43(x42)
    x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2, return_indices=True)
    print(x43.detach().numpy().shape)
    print(x4p.detach().numpy().shape)
    x51 = model.conv51(x4p)
    x52 = model.conv52(x51)
    x53 = model.conv53(x52)
    x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2, return_indices=True)
    print(x53.detach().numpy().shape)
    print(x5p.detach().numpy().shape)