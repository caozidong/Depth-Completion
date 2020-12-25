import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils.visualize_sparse import *


class DepthEncoder(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(DepthEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)
        dl_padding = (filter_size - 1)

        self.init = nn.Sequential(nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding))

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),

                                  )

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),

                                  )

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale=2, pre_x=None):


        ### input

        x0 = self.init(input)
        if pre_x is not None:
            x0 = x0 + F.interpolate(pre_x, scale_factor=scale, mode='bilinear', align_corners=True)

        ### hourglass with short cuts connections between encoder and decoder
        x1 = self.enc1(x0) #1/2 input size
        x2 = self.enc2(x1) # 1/4 input size

        return x0, x1, x2

class RGBEncoder(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(RGBEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)

        self.init = nn.Sequential(nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding))

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc3 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc4 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale=2, pre_x=None):

        ### input

        x0 = self.init(input)
        if pre_x is not None:
            x0 = x0 + F.interpolate(pre_x, scale_factor=scale, mode='bilinear', align_corners=True)

        ### hourglass with short cuts connections between encoder and decoder
        x1 = self.enc1(x0)  # 1/2 input size
        x2 = self.enc2(x1)  # 1/4 input size
        x3 = self.enc3(x2)  # 1/8 input size
        x4 = self.enc4(x3)  # 1/16 input size

        return x0, x1, x2, x3, x4

class DepthDecoder(nn.Module):
    def __init__(self, layers, filter_size):
        super(DepthDecoder, self).__init__()
        padding = int((filter_size-1)/2)

        self.dec2 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers//2, layers//2, filter_size, stride=2, padding=padding, output_padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers//2, layers//2, filter_size, stride=1, padding=padding),
                                  )

        self.dec1 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers//2, layers//2, filter_size, stride=2, padding=padding, output_padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers//2, layers//2, filter_size, stride=1, padding=padding),
                                  )

        self.prdct = nn.Sequential(nn.ReLU(),
                                     nn.Conv2d(layers//2, layers//2, filter_size, stride=1, padding=padding),
                                     nn.ReLU(),
                                     nn.Conv2d(layers//2, 1, filter_size, stride=1, padding=padding))

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, pre_dx, pre_cx):

        x2 = pre_dx[2] + pre_cx[2]#torch.cat((pre_dx[2], pre_cx[2]), 1)
        x1 = pre_dx[1] + pre_cx[1]#torch.cat((pre_dx[1], pre_cx[1]), 1)
        x0 = pre_dx[0] + pre_cx[0]


        x3 = self.dec2(x2) # 1/2 input size
        x4 = self.dec1(x1+x3) #1/1 input size

        ### prediction
        output_d = self.prdct(x4+x0)


        return x4, output_d

class RGBDecoder(nn.Module):
    def __init__(self, layers, filter_size):
        super(RGBDecoder, self).__init__()
        padding = int((filter_size-1)/2)

        self.dec4 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers, layers, filter_size, stride=2, padding=padding,
                                                     output_padding=padding), )

        self.dec3 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers, layers, filter_size, stride=2, padding=padding,
                                                     output_padding=padding), )

        self.dec2 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers, layers, filter_size, stride=2, padding=padding, output_padding=padding),)

        self.dec1 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers, layers//2, filter_size, stride=2, padding=padding, output_padding=padding),
                                        )

        self.prdct = nn.Sequential(nn.ReLU(),
                                     nn.Conv2d(layers//2, layers//2, filter_size, stride=1, padding=padding),
                                     nn.ReLU(),
                                     nn.Conv2d(layers//2, 1, filter_size, stride=1, padding=padding))

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, pre_cx):

        x4 = self.dec4(pre_cx[4]) # 1/8 output size
        x3 = self.dec3(pre_cx[3]+x4)  # 1/4 output size
        x2 = self.dec2(pre_cx[2]+x3) # 1/2 input size
        x1 = self.dec1(pre_cx[1]+x2) #1/1 input size

        ### prediction
        output_d = self.prdct(x1)


        return output_d

class Hourglass(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(Hourglass, self).__init__()
        padding = int((filter_size-1)/2)

        self.init = nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding)

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers*2, filter_size, stride=1, padding=padding),)

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers*2, layers*2, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers*2, layers*2, filter_size, stride=1, padding=padding),)

        self.dec2 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers*2, layers*2, filter_size, stride=2, padding=padding, output_padding=padding),)

        self.dec1 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers*2, layers, filter_size, stride=2, padding=padding, output_padding=padding),
                                        )

        self.prdct = nn.Sequential(nn.ReLU(),
                                     nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),
                                     nn.ReLU(),
                                     nn.Conv2d(layers, 1, filter_size, stride=1, padding=padding))

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale, pre_output=None):


        ### input

        input = self.init(input)
        if pre_output is not None:
            input = input + F.interpolate(pre_output, scale_factor=scale, mode='bilinear', align_corners=True)

        ### hourglass with short cuts connections between encoder and decoder
        x1 = self.enc1(input) #1/2 input size
        x2 = self.enc2(x1) # 1/4 input size
        x3 = self.dec2(x2) # 1/2 input size
        x4 = self.dec1(x1+x3) #1/1 input size

        ### prediction
        output_d = self.prdct(x4+input)


        return x4, output_d

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()

        denc_layers = 32
        cenc_layers = 32
        ddcd_layers = denc_layers+cenc_layers

        self.rgb_encoder = RGBEncoder(3, cenc_layers, 3)
        #self.rgb_decoder = RGBDecoder(cenc_layers, 3)

        self.depth_encoder1 = DepthEncoder(1, denc_layers, 3)
        self.depth_decoder1 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder(ddcd_layers, 3)


    def forward(self, input_d, input_rgb, C):

        enc_c = self.rgb_encoder(input_rgb)
        #rgb_d11 = self.rgb_decoder(enc_c)

        ## for the 1/4 res
        input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        enc_d14 = self.depth_encoder1(input_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])


        ## for the 1/2 res
        input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12 = F.interpolate(dcd_d14[1], scale_factor=2, mode='bilinear', align_corners=True)
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])


        ## for the 1/1 res
        predict_d11 = F.interpolate(dcd_d12[1] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
        input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = dcd_d11[1] + predict_d11
        output_d12 = predict_d11
        output_d14 = F.interpolate(dcd_d14[1], scale_factor=4, mode='bilinear', align_corners=True)

        '''plt.subplot(311)
        plt.imshow(dcd_d14[1][0, 0, :, :].data.cpu())
        plt.subplot(312)
        plt.imshow(dcd_d12[1][0, 0, :, :].data.cpu())
        plt.subplot(313)
        plt.imshow(dcd_d11[1][0, 0, :, :].data.cpu())
        plt.show()'''

        return  output_d11






