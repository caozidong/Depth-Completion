from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def colorize_roi(data, dilate, save_path, show):
    # args:
    # data:  (np.float16) a numpy array
    # dilate: (bool) whether to dilate the valid data points
    #         True -- for spase data
    #         False -- for dense data
    # save_path: (string or None)
    #           string -- save the visualized image to save_path
    #           None -- skip saving
    # show: (bool)

    H, W = np.shape(data)

    color_map = np.full((H, W, 3), 221, np.uint8)

    valid = (data>0)


    max_value = np.amax(data[valid])
    min_value = np.amin(data[valid])
    valid_roi = (data>min_value)&(data<max_value)


    color_map[valid, 0] = 68
    color_map[valid, 1] = 1
    color_map[valid, 2]= 84


    color_map[valid_roi, 0] = 253
    color_map[valid_roi, 1] = 231
    color_map[valid_roi, 2] = 36



    if (dilate):
        valid = np.tile((data>0).reshape(H,W,1),(1,1,3))
        valid_neig = np.concatenate((valid[1:, :, :], np.zeros((1, W, 3), np.bool)), axis=0)
        valid_curt = valid
        valid_curt[0,:,:]=0
        color_map[valid_neig] = color_map[valid_curt]

        '''valid_neig = np.concatenate((valid[:, 1:,  :], np.zeros((H, 1, 3), np.bool)), axis=1)
        valid_curt = valid
        valid_curt[:, 0, :] = 0
        color_map[valid_neig] = color_map[valid_curt]

        valid_neig = np.concatenate((np.zeros((1, W, 3), np.bool), valid[:-1, :, :]), axis=0)
        valid_curt = valid
        valid_curt[-1, :, :] = 0
        color_map[valid_neig] = color_map[valid_curt]

        valid_neig = np.concatenate((np.zeros((H, 1, 3), np.bool), valid[:, :-1, :]), axis=1)
        valid_curt = valid
        valid_curt[:, -1, :] = 0
        color_map[valid_neig] = color_map[valid_curt]'''

    if (save_path):
        cv2.imwrite(save_path, color_map)

    if (show):
        plt.imshow(color_map)
        plt.show()


def colorize(data, dilate, save_path, show):
    # args:
    # data:  (np.float16) a numpy array
    # dilate: (bool) whether to dilate the valid data points
    #         True -- for spase data
    #         False -- for dense data
    # save_path: (string or None)
    #           string -- save the visualized image to save_path
    #           None -- skip saving
    # show: (bool)

    H, W = np.shape(data)

    color_map = np.full((H, W, 3), 221, np.uint8)

    valid = (data>0)

    max_data = np.amax(data[valid])
    min_data = np.amin(data[valid])
    bin_width = (max_data - min_data) / 10.


    valid = (data >= min_data) & (data < min_data + bin_width)
    color_map[valid, 0] = 0
    color_map[valid, 1] = ((data[valid] - min_data) / (bin_width) * 255).astype(np.uint8)
    color_map[valid, 2]= 255

    valid = (data >= min_data + bin_width) & (data < min_data + 4 * bin_width)
    color_map[valid, 0] = ((data[valid] - min_data - bin_width) / (3*bin_width) * 255).astype(np.uint8)
    color_map[valid, 1] = 255
    color_map[valid, 2] = 255 - ((data[valid] - min_data - bin_width) / (3*bin_width) * 255).astype(np.uint8)

    valid = (data >= min_data + 4 * bin_width) & (data <= max_data)
    color_map[valid, 0] = 255
    color_map[valid, 1] = 255 - ((data[valid] - min_data - 4 * bin_width) / (6*bin_width) * 255).astype(np.uint8)
    color_map[valid, 2] = 0

    if (dilate):
        valid = np.tile((data>0).reshape(H,W,1),(1,1,3))
        valid_neig = np.concatenate((valid[1:, :, :], np.zeros((1, W, 3), np.bool)), axis=0)
        valid_curt = valid
        valid_curt[0,:,:]=0
        color_map[valid_neig] = color_map[valid_curt]

        '''valid_neig = np.concatenate((valid[:, 1:,  :], np.zeros((H, 1, 3), np.bool)), axis=1)
        valid_curt = valid
        valid_curt[:, 0, :] = 0
        color_map[valid_neig] = color_map[valid_curt]

        valid_neig = np.concatenate((np.zeros((1, W, 3), np.bool), valid[:-1, :, :]), axis=0)
        valid_curt = valid
        valid_curt[-1, :, :] = 0
        color_map[valid_neig] = color_map[valid_curt]

        valid_neig = np.concatenate((np.zeros((H, 1, 3), np.bool), valid[:, :-1, :]), axis=1)
        valid_curt = valid
        valid_curt[:, -1, :] = 0
        color_map[valid_neig] = color_map[valid_curt]'''

    if (save_path):
        cv2.imwrite(save_path, color_map)

    if (show):
        plt.imshow(color_map)
        plt.show()


