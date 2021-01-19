import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.visualize_sparse import colorize
import numpy as np

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

