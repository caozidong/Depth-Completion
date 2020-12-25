########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.visualize_sparse import colorize
import numpy as np


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)

        sigma_x = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = torch.ne(target, 0).float().cuda()
        loss = F.smooth_l1_loss(outputs * val_pixels, target * val_pixels, reduction='none')
        return torch.mean(loss)

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = (target > 0).float().cuda()
        err = (target * val_pixels - outputs * val_pixels) ** 2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.mean(torch.sqrt(loss / cnt))

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = torch.ne(target, 0).float().cuda()
        loss = target * val_pixels - outputs * val_pixels
        return torch.sum(torch.abs(loss)) / torch.sum(val_pixels)

class MAESmoothLoss(nn.Module):
    def __init__(self):
        super(MAESmoothLoss, self).__init__()

    def gradient_x(self, img):
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gx
    def gradient_y(self, img):
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gy

    def forward(self, outputs, target, img, ifsmooth=1, *args):
        val_pixels = torch.ne(target, 0).float().cuda()
        loss = target * val_pixels - outputs * val_pixels
        gt_loss = torch.sum(torch.abs(loss)) / torch.sum(val_pixels)
        if ifsmooth == 1:
            disp_gradients_x = self.gradient_x(self.gradient_x(outputs))
            disp_gradients_y = self.gradient_y(self.gradient_y(outputs))

            image_gradients_x = self.gradient_x(self.gradient_x(img))
            image_gradients_y = self.gradient_y(self.gradient_y(img))

            weights_x = torch.exp(-1 * torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
            weights_y = torch.exp(-1 * torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

            smoothness_x = torch.abs(disp_gradients_x) * weights_x
            smoothness_y = torch.abs(disp_gradients_y) * weights_y
            smooth_loss = 0.5 * ((smoothness_x).mean() + (smoothness_y).mean())

            w1, w2 = 0.8, 0.2
            return w1* gt_loss + w2 * smooth_loss
        else:
            return gt_loss

class MAEPaintLoss_v1(nn.Module):
    '''MAE Loss + simplified Relative Loss'''
    def __init__(self):
        super(MAEPaintLoss_v1, self).__init__()

    def forward(self, outputs, target, inputs, disp, *args):
        assert (disp > 1)
        val_pixels_1 = torch.ne(target, 0).float().cuda()
        val_pixels_2 = torch.eq(inputs, 0).float().cuda()
        outputs = outputs * val_pixels_1
        loss_abs = torch.sum(torch.abs((target - outputs) * val_pixels_2))

        loss_rel = torch.zeros(1).float().cuda()
        for i in range(0, disp):
            for j in range(0, disp):
                if i == 0 and j != 0:
                    loss_rel = loss_rel + torch.sum(torch.abs(((target[:, :, :, j:] - target[:, :, :, :-j]) - (outputs[:, :, :, j:] - outputs[:, :, :, :-j])) \
                               * val_pixels_1[:, :, :, j:] * val_pixels_1[:, :, :, :-j]))
                elif j == 0 and i != 0:
                    loss_rel = loss_rel + torch.sum(torch.abs(((target[:, :, i:, :] - target[:, :, :-i, :]) - (outputs[:, :, i:, :] - outputs[:, :, :-i, :])) \
                               * val_pixels_1[:, :, i:, :] * val_pixels_1[:, :, :-i, :]))
                elif i != 0 and j != 0:
                    loss_rel = loss_rel + torch.sum(torch.abs(((target[:, :, i:, j:] - target[:, :, :-i, :-j]) - (outputs[:, :, i:, j:] - outputs[:, :, :-i, :-j])) \
                         * val_pixels_1[:, :, i:, j:] * val_pixels_1[:, :, :-i, :-j]))
                    loss_rel = loss_rel + torch.sum(torch.abs(((target[:, :, i:, :-j] - target[:, :, :-i, j:]) - (outputs[:, :, i:, :-j] - outputs[:, :, :-i, j:])) \
                         * val_pixels_1[:, :, i:, :-j] * val_pixels_1[:, :, :-i, j:]))

        weight_abs, weight_rel = 0.1, 0.9
        loss = weight_abs * loss_abs / (torch.sum(val_pixels_1 * val_pixels_2) + 1) + weight_rel * (loss_rel / ((disp - 1) ** 2)) / torch.sum(val_pixels_1)
        return loss

class MAEPaintLoss_v2(nn.Module):
    '''MAE Loss + Relative Loss + SSIMLoss'''
    def __init__(self):
        super(MAEPaintLoss_v2, self).__init__()

    def forward(self, outputs, target, inputs, disp, *args):
        assert (disp > 1)
        val_pixels_1 = torch.ne(target, 0).float().cuda()
        val_pixels_2 = torch.eq(inputs, 0).float().cuda()
        val_pixels_3 = val_pixels_1 * val_pixels_2
        val_pixels_4 = torch.ne(inputs, 0).float().cuda()
        outputs = outputs * val_pixels_1

        loss_abs = torch.sum(torch.abs((target - outputs) * val_pixels_2))

        loss_rel = torch.zeros(1).float().cuda()
        for i in range(0, disp):
            for j in range(0, disp):
                if i == 0 and j != 0:
                    loss_rel = loss_rel + torch.sum((((target[:, :, :, j:] - target[:, :, :, :-j]) - (outputs[:, :, :, j:] - outputs[:, :, :, :-j])) *
                            val_pixels_1[:, :, :, j:] * val_pixels_1[:, :, :, :-j] * (1 / np.sqrt(i ** 2 + j ** 2)) *
                            (10 * val_pixels_3[:, :, :, j:] * val_pixels_4[:, :, :, :-j] + 10 * val_pixels_3[:, :, :, :-j] * val_pixels_4[:, :, :, j:] + 1)) ** 2)

                elif j == 0 and i != 0:
                    loss_rel = loss_rel + torch.sum((((target[:, :, i:, :] - target[:, :, :-i, :]) - (outputs[:, :, i:, :] - outputs[:, :, :-i, :])) *
                            val_pixels_1[:, :, i:, :] * val_pixels_1[:, :, :-i, :] * (1 / np.sqrt(i ** 2 + j ** 2)) *
                            (10 * val_pixels_3[:, :, :-i, :] * val_pixels_4[:, :, i:, :] + 10 * val_pixels_3[:, :, i:, :] * val_pixels_4[:, :, :-i, :] + 1)) ** 2)

                elif i != 0 and j != 0:
                    loss_rel = loss_rel + torch.sum((((target[:, :, i:, j:] - target[:, :, :-i, :-j]) - (outputs[:, :, i:, j:] - outputs[:, :, :-i, :-j])) *
                            val_pixels_1[:, :, i:, j:] * val_pixels_1[:, :, :-i, :-j] *
                            (10 * val_pixels_3[:, :, :-i, :-j] * val_pixels_4[:, :, i:, j:] + 10 * val_pixels_3[:, :, i:, j:] * val_pixels_4[:, :, :-i, :-j] + 1)) ** 2)

                    loss_rel = loss_rel + torch.sum((((target[:, :, i:, :-j] - target[:, :, :-i, j:]) - (outputs[:, :, i:, :-j] - outputs[:, :, :-i, j:])) *
                            val_pixels_1[:, :, i:, :-j] * val_pixels_1[:, :, :-i, j:] *
                            (10 * val_pixels_3[:, :, :-i, j:] * val_pixels_4[:, :, i:, :-j] + 10 * val_pixels_3[:, :, i:, :-j] * val_pixels_4[:, :, :-i, j:] + 1)) ** 2)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(outputs, 3, 1, 1)
        mu_y = F.avg_pool2d(target, 3, 1, 1)

        sigma_x = F.avg_pool2d(outputs ** 2, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(target ** 2, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(outputs * target, 3, 1, 1) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        loss_ssim = torch.mean(torch.clamp((1 - SSIM) / 2, 0, 1))

        weight_abs, weight_rel, weight_ssim = 0.1, 0.8, 0.1
        loss = weight_abs * loss_abs / (torch.sum(val_pixels_1 * val_pixels_2) + 1) + weight_rel * (loss_rel / ((disp - 1) ** 2)) / torch.sum(val_pixels_1) + weight_ssim * loss_ssim
        return loss

