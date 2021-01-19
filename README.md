# Self-supervised-Depth-Completion

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

You need to download  [matterport3D](https://github.com/niessner/Matterport) and follow [yindaZ](https://github.com/yindaz/DeepCompletionRelease)'s repository to get above data.
