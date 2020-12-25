from torch.nn import *


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,bias=bias)

def trans_conv3x3(in_planes, out_planes, stride=2, bias=False):
    return ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=1, bias=bias)

def conv3x3_bn(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return Sequential(
        Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias),
        BatchNorm2d(out_planes),
    )

def conv1x1_bn(in_planes, out_planes, stride=2, bias=False):  #****************
    """1x1 convolution"""
    return Sequential(
        Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias),
        BatchNorm2d(out_planes)
    )

def trans_conv3x3_bn(in_planes, out_planes, stride=2, bias=False):
    return Sequential(
        ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=1, output_padding=1, bias=bias),
        BatchNorm2d(out_planes),
    )

def conv3x3_do(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return Sequential(Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias),
                      Dropout2d())

def conv1x1_do(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return Sequential(Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,bias=bias),
                      Dropout2d())

def trans_conv3x3_do(in_planes, out_planes, stride=2, bias=False):
    return Sequential(ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=1, bias=bias),
                      Dropout2d())

def conv3x3_bn_do(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return Sequential(
        Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias),
                      Dropout2d(),
        BatchNorm2d(out_planes),
    )

def conv1x1_bn_do(in_planes, out_planes, stride=2, bias=False):  #****************
    """1x1 convolution"""
    return Sequential(
        Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias),
                      Dropout2d(),
        BatchNorm2d(out_planes)
    )

def trans_conv3x3_bn_do(in_planes, out_planes, stride=2, bias=False):
    return Sequential(
        ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=1, output_padding=1, bias=bias),
                      Dropout2d(),
        BatchNorm2d(out_planes),
    )

class DownsampleResBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes):
        super(DownsampleResBlock, self).__init__()

        self.conv_bn1 = conv3x3_bn(inplanes,planes,stride=2)
        self.relu = ReLU(inplace=True)
        self.conv_bn2 = conv3x3_bn(planes,planes)
        self.downsample = conv1x1_bn(inplanes,planes,stride=2)

    def forward(self, x):

        out = self.conv_bn1(x)   #downsample the main stream to 1/2 resolution
        out = self.relu(out)

        out = self.conv_bn2(out)

        identity = self.downsample(x) #downsample the shortcut

        out += identity
        out = self.relu(out)

        return out

class ResBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()

        self.conv_bn1 = conv3x3_bn(inplanes,planes)
        self.relu = ReLU(inplace=True)
        self.conv_bn2 = conv3x3_bn(planes,planes)

    def forward(self, x):

        out = self.conv_bn1(x)   #the main stream to 1/2 resolution
        out = self.relu(out)

        out = self.conv_bn2(out)

        identity = x #the shortcut

        out += identity
        out = self.relu(out)

        return out

class BasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv_bn1 = conv3x3_bn(inplanes,planes,stride)
        self.relu = ReLU(inplace=True)
        self.conv_bn2 = conv3x3_bn(planes,planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_bn1(x)   #downsample the main stream to 1/2 resolution
        out = self.relu(out)

        out = self.conv_bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x) #downsample the shortcut

        out += identity
        out = self.relu(out)

        return out

class BasicBlockDo(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockDo, self).__init__()

        self.conv_bn1 = conv3x3_bn_do(inplanes,planes,stride)
        self.relu = ReLU(inplace=True)
        self.conv_bn2 = conv3x3_bn_do(planes,planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_bn1(x)   #downsample the main stream to 1/2 resolution
        out = self.relu(out)

        out = self.conv_bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x) #downsample the shortcut

        out += identity
        out = self.relu(out)

        return out