#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author Chenlei
# @date 4/12/21
# @fileName image_enhance.py
# Copyright 2021 ruyueshi@qq.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import glob
import time
import numpy as np
import cv2


def singleScaleRetinex(img, sigma):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))

    return retinex


def multiScaleRetinex(img, sigma_list):
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)

    retinex = retinex / len(sigma_list)

    return retinex


def colorRestoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)

    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))

    return color_restoration


def simplestColorBalance(img, low_clip, high_clip):
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c

        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img


def MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):
    img = np.float64(img) + 1.0

    img_retinex = multiScaleRetinex(img, sigma_list)

    img_color = colorRestoration(img, alpha, beta)
    img_msrcr = G * (img_retinex * img_color + b)

    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                             255

    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)

    return img_msrcr


def automatedMSRCR(img, sigma_list):
    img = np.float64(img) + 1.0

    img_retinex = multiScaleRetinex(img, sigma_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break

        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break

        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255

    img_retinex = np.uint8(img_retinex)

    return img_retinex


def MSRCP(img, sigma_list, low_clip, high_clip):
    img = np.float64(img) + 1.0

    intensity = np.sum(img, axis=2) / img.shape[2]

    retinex = multiScaleRetinex(intensity, sigma_list)

    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)

    intensity1 = simplestColorBalance(retinex, low_clip, high_clip)

    intensity1 = (intensity1 - np.min(intensity1)) / \
                 (np.max(intensity1) - np.min(intensity1)) * \
                 255.0 + 1.0

    img_msrcp = np.zeros_like(img)

    for y in range(img_msrcp.shape[0]):
        for x in range(img_msrcp.shape[1]):
            B = np.max(img[y, x])
            A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]

    img_msrcp = np.uint8(img_msrcp - 1.0)

    return img_msrcp


def test_retinex():
    from CoreApp.config import BASE_DIR
    img_list = glob.glob(os.path.join(BASE_DIR, 'exampleImages/05_double_orthogonal_ammeter/*[!sa][!kp].[jp][pn]g'))
    vis_save_path = os.path.join(BASE_DIR, 'data/result/image_enhance/05_double_orthogonal_ammeter')
    if not os.path.exists(vis_save_path):
        os.makedirs(vis_save_path)

    config = {"sigma_list": [15, 80, 200],
              "G": 5.0,
              "b": 25.0,
              "alpha": 125.0,
              "beta": 46.0,
              "low_clip": 0.01,
              "high_clip": 0.99}

    for img_path in img_list:
        print("****** process", img_path)
        img_name = os.path.basename(img_path).split('.')[0]
        suffix = os.path.basename(img_path).split('.')[1]
        img = cv2.imread(img_path)

        print('msrcr processing......', end='')
        start = time.time()
        img_msrcr = MSRCR(
            img,
            config['sigma_list'],
            config['G'],
            config['b'],
            config['alpha'],
            config['beta'],
            config['low_clip'],
            config['high_clip']
        )
        end = time.time()
        print(' cost {} s'.format(end - start))
        cv2.imwrite(os.path.join(vis_save_path, img_name + "_MSRCR_retinex." + suffix), img_msrcr)

        print('amsrcr processing......', end='')
        start = time.time()
        img_amsrcr = automatedMSRCR(
            img,
            config['sigma_list']
        )
        end = time.time()
        print(' cost {} s'.format(end - start))
        cv2.imwrite(os.path.join(vis_save_path, img_name + "_AutomatedMSRCR_retinex." + suffix), img_amsrcr)

        print('msrcp processing......', end='')
        start = time.time()
        img_msrcp = MSRCP(
            img,
            config['sigma_list'],
            config['low_clip'],
            config['high_clip']
        )
        end = time.time()
        print(' cost {} s'.format(end - start))
        cv2.imwrite(os.path.join(vis_save_path, img_name + "_MSRCP_retinex." + suffix), img_msrcp)


if __name__ == '__main__':
    test_retinex()