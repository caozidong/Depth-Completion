from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import glob
import torchvision
import matplotlib.pyplot as plt
import random
from random import choice
from numpy import linalg as LA
import cv2
from torchvision import transforms
np.seterr(divide='ignore', invalid='ignore')


class MatterportDataset(Dataset):
    def __init__(self, data_path, gt_path, rgb_path, setname='train', transform=None, norm_factor=256, invert_depth=False,
                 rgb2gray=False, flip=1):
        self.data_path = data_path
        self.gt_path = gt_path
        self.rgb_path = rgb_path
        self.setname = setname
        self.transform = transform
        self.norm_factor = norm_factor
        self.invert_depth = invert_depth
        self.rgb2gray = rgb2gray
        self.flip = flip

        # self.data = list(sorted(glob.iglob(self.data_path + "/*/undistorted_depth_images/*.png", recursive=True)))
        # self.gt = list(sorted(glob.iglob(self.gt_path + "/*/mesh_images/*.png", recursive=True)))
        # self.rgb = list(sorted(glob.iglob(self.rgb_path + "/*/undistorted_color_images/*.jpg", recursive=True)))

        if setname == 'train':
            data = []
            gt = []
            rgb = []
            train_list = './dataloader/mp_train_list_noup.txt'
            f = open(train_list)
            for line in f:
                line_rgb = line.strip('\n')
                line_data = line_rgb.replace('undistorted_color_images', 'undistorted_depth_images').replace('jpg', 'png')
                line_data = line_data[:-8] + 'd' + line_data[-7:]
                line_gt = line_data.replace('undistorted_depth_images', 'mesh_images').replace(line_data.split('/')[-1], line_data.split('/')[-1].split('.')[0] + '_mesh_depth.png')
                data.append(self.rgb_path + line_data)
                gt.append(self.rgb_path + line_gt)
                rgb.append(self.rgb_path + line_rgb)
            self.data = data
            self.gt = gt
            self.rgb = rgb

        if setname == 'val':
            data = []
            gt = []
            rgb = []
            test_list = './dataloader/mp_test_list_horizontal.txt'
            f = open(test_list)
            for line in f:
                line_rgb = line.strip('\n')
                line_data = line_rgb.replace('undistorted_color_images', 'undistorted_depth_images').replace('jpg', 'png')
                line_data = line_data[:-8] + 'd' + line_data[-7:]
                line_gt = line_data.replace('undistorted_depth_images', 'mesh_images').replace(line_data.split('/')[-1], line_data.split('/')[-1].split('.')[0] + '_mesh_depth.png')
                data.append(self.rgb_path + line_data)
                gt.append(self.rgb_path + line_gt)
                rgb.append(self.rgb_path + line_rgb)
            self.data = data
            self.gt = gt
            # self.data = list(sorted(glob.glob('./experiments/val_input_d_epoch_9/' + '*.npy', recursive=True)))
            # self.gt = list(sorted(glob.glob('./experiments/val_labels_epoch_9/' + '*.npy', recursive=True)))
            self.rgb = rgb

    def __len__(self):
        return len(self.data)  #the length of data list

    def mask_create(self, x, is_dilate=False, is_random=False):
        #Shape of x: [H, W]
        H, W = x.shape
        FULL_KERNEL_15 = np.ones((15, 15), np.uint8)
        x_all = np.array((x > 0), dtype=np.float32)

        #Dilate
        C_x = np.array((x == 0), dtype=np.float32)
        if is_dilate:
            x_dilate = cv2.dilate(C_x, FULL_KERNEL_15)
            if 0 not in x_dilate:
                x_dilate = C_x
            x_dilate = 1 - x_dilate
            x_all = x_all * x_dilate

        #Random mask
        if is_random:
            mask_x = random.randint(0, W - 150)
            mask_y = random.randint(0, H - 150)
            mask_size_x = random.randint(100, 150)
            mask_size_y = random.randint(100, 150)
            mask = np.ones_like(x)
            mask[mask_y: mask_y + mask_size_y, mask_x: mask_x + mask_size_x] = 0
            x_all = x_all * mask

        '''plt.plot()
        plt.subplot(2, 1, 1)
        plt.imshow(mask)
        plt.subplot(2, 1, 2)
        plt.imshow(x_all)
        plt.show()
        assert False'''

        return x_all

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        # Read images and convert them to 4D floats
        data = Image.open(str(self.data[item]))
        gt = Image.open(str(self.gt[item]))
        # data = np.load(self.data[item])
        # gt = np.load(self.gt[item])
        # data = Image.fromarray(data)
        # gt = Image.fromarray(gt)

        # Read RGB images
        if self.setname == 'train' or self.setname == 'val':
            rgb = Image.open(str(self.rgb[item]))

        if self.rgb2gray:
            t = torchvision.transforms.Grayscale(1)
            rgb = t(rgb)

        # Apply transformations if given
        if self.transform is not None:
            data = self.transform(data)
            gt = self.transform(gt)
            rgb = self.transform(rgb)

        if self.flip and random.randint(0, 1):
            # data = data.transpose(Image.FLIP_TOP_BOTTOM)
            data = data.transpose(Image.FLIP_TOP_BOTTOM)
            gt = gt.transpose(Image.FLIP_TOP_BOTTOM)
            rgb = rgb.transpose(Image.FLIP_TOP_BOTTOM)

        # Resize all the images
        # transform = transforms.Resize((256, 320))
        # data = transform(data)
        # gt = transform(gt)
        # rgb = transform(rgb)
        data = data.resize((320, 256))
        gt = gt.resize((320, 256))
        rgb = rgb.resize((320, 256))

        # Convert to numpy
        data = np.array(data, dtype=np.float32)
        data = data / 4000
        data[data < 0.1] = 0
        gt = np.array(gt, dtype=np.float32)
        gt = gt / 4000
        gt[gt < 0.1] = 0

        '''plt.imshow(data)
        plt.show()
        assert False'''

        gt_fake = np.copy(data)

        mask = self.mask_create(data, is_dilate=True, is_random=False)
        data_fake = data * mask

        # define the certainty
        C = (data > 0).astype(float)
        # print(np.min(data[data > 0]))

        # Normalize the data
        #data = data / self.norm_factor
        #gt = gt / self.norm_factor

        # Expand dims into Pytorch format
        data = np.expand_dims(data, 0)
        gt = np.expand_dims(gt, 0)
        C = np.expand_dims(C, 0)
        data_fake = np.expand_dims(data_fake, 0)
        gt_fake = np.expand_dims(gt_fake, 0)

        # Convert to Pytorch Tensors
        data = torch.tensor(data, dtype=torch.float)
        gt = torch.tensor(gt, dtype=torch.float)
        C = torch.tensor(C, dtype=torch.float)
        data_fake = torch.tensor(data_fake, dtype=torch.float)
        gt_fake = torch.tensor(gt_fake, dtype=torch.float)


        # Convert RGB image to tensor
        rgb = np.array(rgb, dtype=np.float16)
        rgb /= 255
        if self.rgb2gray:
            rgb = np.expand_dims(rgb, 0)
        else:
            rgb = np.transpose(rgb, (2, 0, 1))
        rgb = torch.tensor(rgb, dtype=torch.float)

        if self.setname == 'train':
            return data_fake, C, gt_fake, item, rgb
        elif self.setname == 'val':
            return data, C, gt, item, rgb


