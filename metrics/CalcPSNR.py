# -*- coding: utf-8 -*-
import os
import math
import numpy as np
# import skimage.metrics.peak_signal_noise_ratio as psnr_skimage
import cv2
import glob
import datetime


def calc_psnr(folder_Gen, folder_GT, result_save_path, epoch):

    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    PSNR, total_psnr, avg_psnr = 0.0, 0.0, 0.0
    epoch_result = result_save_path + 'PSNR_epoch_' + str(epoch) + '.csv'
    epochfile = open(epoch_result, 'w')
    epochfile.write('image_name' + ','+ 'psnr' + '\n')

    total_result = result_save_path + 'PSNR_total_results_epoch_avgpsnr.csv'
    totalfile = open(total_result, 'a+')

    crop_border = 4
    test_Y = False  # True: test Y channel only; False: test RGB channels

    img_list = sorted(glob.glob(folder_Gen + '/*'))

    if test_Y:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    starttime = datetime.datetime.now()

    for i, img_path in enumerate(img_list):
        im_Gen = cv2.imread(img_path) / 255.

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        imgName, _, _ = base_name.rsplit('_', 2)
        # print(base_name)
        # print(imgName)
        GT_imgName = imgName + '.png'
        # print(GT_imgName)
        im_GT = cv2.imread(os.path.join(folder_GT, GT_imgName)) / 255.

        if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
            im_GT_in = bgr2ycbcr(im_GT)
            im_Gen_in = bgr2ycbcr(im_Gen)
        else:
            im_GT_in = im_GT
            im_Gen_in = im_Gen

        # crop borders
        if im_GT_in.ndim == 3:
            cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
        elif im_GT_in.ndim == 2:
            cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
            cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
        else:
            raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))

        # calculate PSNR and SSIM
        PSNR = calculate_psnr(cropped_GT * 255, cropped_Gen * 255)

        # print("=== PSNR is {:.4f} of {:>3d}-th img ===".format(PSNR, i))

        epochfile.write(GT_imgName + ',' + str(round(PSNR, 6)) + '\n')

        total_psnr += PSNR
        if i % 50 == 0:
            print("=== PSNR is processing {:>3d}-th image ===".format(i))

    endtime = datetime.datetime.now()
    print("======================= Complete the PSNR test of {:>3d} images, take {} seconds ======================= ".format(i+1, (endtime - starttime).seconds))
    avg_psnr = total_psnr / i
    epochfile.write('Average' + ',' + str(round(avg_psnr, 6)) + '\n')
    epochfile.close()
    totalfile.write(str(epoch) + ',' + str(round(avg_psnr, 6)) + '\n')
    totalfile.close()
    return avg_psnr


def calculate_psnr(img1, img2, data_range=255):
    # img1 and img2 have range [0, 255]
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2, dtype=np.float64)
    if mse == 0:
        return float('inf')
    # return 20 * math.log10(255.0 / math.sqrt(mse))
    return 10 * np.log10((data_range ** 2)/ mse)


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)