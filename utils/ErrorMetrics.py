import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = (target > 0).float().cuda()
        err = torch.abs(target * val_pixels - outputs * val_pixels)
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.mean(loss / cnt)

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = (target > 0).float().cuda()
        err = (target * val_pixels - outputs * val_pixels) ** 2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.mean(torch.sqrt(loss / cnt))

class Deltas(nn.Module):
    def __init__(self):
        super(Deltas, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = (target > 0).float().cuda()
        rel = torch.max((target * val_pixels) / (outputs * val_pixels + 1e-3),
                        (outputs * val_pixels) / (target * val_pixels))

        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)

        def del_i(i):
            r = (rel < 1.01 ** i).float()
            delta = torch.sum(r.view(r.size(0), 1, -1), -1, keepdim=True) / cnt
            return torch.mean(delta)

        return del_i(1), del_i(2), del_i(3)

class Huber(nn.Module):
    def __init__(self):
        super(Huber, self).__init__()

    def forward(self, outputs, target, delta=5):
        l1_loss = F.l1_loss(outputs, target, reduce=False)
        mse_loss = F.mse_loss(outputs, target, reduce=False)

        mask = (l1_loss < delta).float()

        loss = (0.5 * mse_loss) * mask + delta * (l1_loss - 0.5 * delta) * (1 - mask)

        return torch.mean(loss)

class EPE_metric(nn.Module):
    def __init__(self):
        super(EPE_metric, self).__init__()

    def forward(self, outputs, target):
        mask = (target > 0)
        outputs, target = outputs[mask], target[mask]
        err = torch.abs(target - outputs)
        loss = torch.mean(err)
        return loss

class D1_metric(nn.Module):
    def __init__(self):
        super(D1_metric, self).__init__()

    def forward(self, outputs, target):
        mask = (target > 0)
        outputs, target = outputs[mask], target[mask]
        E = torch.abs(outputs - target)
        # err_mask = (E > 3) & (E / target.abs() > 0.05)
        err_mask = (E > 3)
        return torch.mean(err_mask.float())

class Thres_metric(nn.Module):
    def __init__(self):
        super(Thres_metric, self).__init__()

    def forward(self, outputs, target):
        mask_tar = (target > 0)
        mask_out = (outputs > 0)
        mask = mask_tar * mask_out
        # mask = (target > 0)
        thres = 3
        assert isinstance(thres, (int, float))
        outputs, target = outputs[mask], target[mask]
        E = torch.abs(target - outputs)
        err_mask = (E > thres)
        return torch.mean(err_mask.float())

class Deltas_Paint(nn.Module):
    def __init__(self):
        super(Deltas_Paint, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = (target > 0).float().cuda()
        rel = torch.max((target * val_pixels) / (outputs * val_pixels + 1e-3),
                        (outputs * val_pixels) / (target * val_pixels))

        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)

        def del_i(i):
            r = (rel < 1.25 ** i).float()
            delta = torch.sum(r.view(r.size(0), 1, -1), -1, keepdim=True) / cnt
            return torch.mean(delta)

        def del_j(i):
            r = (rel < i).float()
            delta = torch.sum(r.view(r.size(0), 1, -1), -1, keepdim=True) / cnt
            return torch.mean(delta)

        return del_j(1.05), del_j(1.10), del_i(1), del_i(2), del_i(3), cnt

class SSIM_Metric(nn.Module):
    def __init__(self):
        super(SSIM_Metric, self).__init__()

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

        # return torch.clamp((1 - SSIM) / 2, 0, 1)
        return SSIM.mean(), torch.tensor([torch.numel(x)])

class MAE_Paint(nn.Module):
    def __init__(self):
        super(MAE_Paint, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = (target > 0).float().cuda()
        err = torch.abs(target * val_pixels - outputs * val_pixels)
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.mean(loss / torch.numel(outputs)), torch.tensor([torch.numel(outputs)])

class RMSE_Paint(nn.Module):
    def __init__(self):
        super(RMSE_Paint, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = (target > 0).float().cuda()
        err = (target * val_pixels - outputs * val_pixels) ** 2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return (loss / torch.numel(outputs)), torch.tensor([torch.numel(outputs)])