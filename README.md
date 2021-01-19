# Self-supervised-Depth-Completion

## Introduction

Depth sensing is crucial for many computer vision applications. Commodity-level RGB-D cameras are often unable to sense depth in distant, reflective and transparent regions,
resulting in large missing areas. As the acquisition of depth annotations in missing areas is tedious, we propose a selfsupervised method for the task of completing depth values of missing areas. Specifically, we sample the incomplete raw depth map via an adaptive sampling strategy to generate a more incomplete depth map as the input and use the raw depth map as the training label. To enable the network to propagate long-range depth information to fill large invalid areas, we
further propose a relative consistency loss during training. Experiments validate the effectiveness of our self-supervised method, which outperforms previous unsupervised methods and even can compete with some supervised methods.

## Data Preparation
In this repository, we only train and test on Matterport3D dataset.
```bash
dataset/
├── 1LXtFkjw3qL
│   ├── mesh_images
│   │   ├── 0b22fa63d0f54a529c525afbf2e8bb25_d0_0_mesh_depth.png
│   ├── undistorted_color_images
│   │   ├── 0b22fa63d0f54a529c525afbf2e8bb25_i0_0.png
│   │   └── ...
│   └── undistorted_depth_images
│       ├── 0b22fa63d0f54a529c525afbf2e8bb25_d0_0.png
│       └── ...
└─── 1pXnuDYAj8r
     └── ...
```
In our method, we resize all images to 320x256, including training and testing processes. We use three types of data, which are shown below:
- mesh_images: rendered ground truth from multi_view reconstruction
- undistorted_color_images: RGB images aligned with raw depth images
- undistorted_depth_images: raw depth images captured with a matterport camera

You need to download  [matterport3D](https://github.com/niessner/Matterport) and follow [yinda](https://github.com/yindaz/DeepCompletionRelease)'s repository to get above data. Training list `./dataloader/mp_test_list_horizontal.txt` and testing list `./dataloader/mp_train_list_noup.txt` are provided by Yu-Kai.(https://github.com/tsunghan-mama/Depth-Completion)'s

## Environment Setup
Python 3.5, Pytorch 1.1.0

## Training and testing
After being ready for the dataset, we can begin to train the model. Note that we resize the images with the nearest interpolation. 

Change the parameters in `params.json`, such as dataset_dir, loss and batch-size.

- In training, the inputs are RGB images and sampled depth images(RGB and data_fake in `MatterportDataset`), and the training labels are raw depth images(gt_fake in `MatterportDataset`).

- In testing, the inputs are RGB images and raw depth images(RGB and data in `MatterportDataset`), and the ground truth is the mesh_images(gt in `MatterportDataset`).

We trained about 30 epoches to get the final model.

