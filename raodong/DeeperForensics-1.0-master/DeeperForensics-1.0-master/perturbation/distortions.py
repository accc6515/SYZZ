import math
import os
import random

import cv2
import numpy as np


def bgr2ycbcr(img_bgr):
    img_bgr = img_bgr.astype(np.float32)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
    img_ycbcr = img_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    # to [16/255, 235/255]
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0
    # to [16/255, 240/255]
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0

    return img_ycbcr


def ycbcr2bgr(img_ycbcr):
    img_ycbcr = img_ycbcr.astype(np.float32)
    # to [0, 1]
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * 255.0 - 16) / (235 - 16)
    # to [0, 1]
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * 255.0 - 16) / (240 - 16)
    img_ycrcb = img_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
    img_bgr = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2BGR)

    return img_bgr

#色彩饱和度扰动CS
def color_saturation(img, param):
    ycbcr = bgr2ycbcr(img)
    ycbcr[:, :, 1] = 0.5 + (ycbcr[:, :, 1] - 0.5) * param
    ycbcr[:, :, 2] = 0.5 + (ycbcr[:, :, 2] - 0.5) * param
    img = ycbcr2bgr(ycbcr).astype(np.uint8)

    return img

#对比度扰动CC
def color_contrast(img, param):
    img = img.astype(np.float32) * param
    img = img.astype(np.uint8)

    return img

#块状遮挡扰动BW
def block_wise(img, param):
    # Block size adapts to image size, minimum 4x4, maximum 32x32
    min_block = 4
    max_block = 32
    width = max(min_block, min(max_block, min(img.shape[0], img.shape[1]) // 16))
    block = np.ones((width, width, 3), dtype=int) * 128
    # Number of blocks adapts to image area, but at least 1
    num_blocks = max(1, int((img.shape[0] * img.shape[1]) / (256 * 256) * param))
    for _ in range(num_blocks):
        if img.shape[0] <= width or img.shape[1] <= width:
            continue
        r_w = random.randint(0, img.shape[1] - width)
        r_h = random.randint(0, img.shape[0] - width)
        img[r_h:r_h + width, r_w:r_w + width, :] = block

    return img

#加高斯噪声（彩色）GNC
def gaussian_noise_color(img, param):
    ycbcr = bgr2ycbcr(img) / 255
    size_a = ycbcr.shape
    b = (ycbcr + math.sqrt(param) *
         np.random.randn(size_a[0], size_a[1], size_a[2])) * 255
    b = ycbcr2bgr(b)
    img = np.clip(b, 0, 255).astype(np.uint8)

    return img

#高斯模糊GB
def gaussian_blur(img, param):
    img = cv2.GaussianBlur(img, (param, param), param * 1.0 / 6)

    return img

#JPEG压缩（缩放模拟）JPEG
def jpeg_compression(img, param):
    h, w, _ = img.shape
    s_h = h // param
    s_w = w // param
    img = cv2.resize(img, (s_w, s_h))
    img = cv2.resize(img, (w, h))

    return img

#视频压缩（整段处理）VC
def video_compression(vid_in, vid_out, param):
    cmd = f'ffmpeg -i {vid_in} -crf {param} -y {vid_out}'
    os.system(cmd)

    return
