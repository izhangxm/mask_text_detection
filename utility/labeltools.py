#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author tccw
# @date 2021/4/3
# @fileName labeltools.py
# Copyright 2017 izhangxm@gmail.com. All Rights Reserved.
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
import json
import os

import cv2
import numpy as np

from CoreApp.utils.image_tool import show_cvimg_in_sciview


def convert_label_file_to_text_info(json_path):
    json_str = open(json_path, 'r', encoding='utf-8').read()
    text_info = conver_label_string_to_text_info(json_str)
    return text_info


def conver_label_string_to_text_info(json_str):
    text_info = []
    label = json.loads(json_str)
    for key, val in label['results'].items():
        p = val['points']
        location: list = p[0] + p[1] + p[2] + p[3]
        location.append(val['label'])
        location.append(1.0)
        text_info.append(location)
    return text_info


def conver_my_color_mask_to_labelmap(cv_img):
    # pass
    b, g, r = cv2.split(cv_img)
    h, w = cv_img.shape[:2]
    pointer_indexs = np.where(r >= 200)
    scale_indexs = np.where(g >= 200)
    label_map = np.zeros((h, w), dtype=np.uint8)
    label_map[pointer_indexs] = 1
    label_map[scale_indexs] = 2
    return label_map


def conver_my_color_mask_with_white_background_to_labelmap(cv_img):
    # pass
    b, g, r = cv2.split(cv_img)
    h, w = cv_img.shape[:2]
    pointer_indexs = np.where(g <= 200)
    scale_indexs = np.where((r <= 200) & (g > 200))
    label_map = np.zeros((h, w), dtype=np.uint8)
    label_map[pointer_indexs] = 1
    label_map[scale_indexs] = 2
    # show_cvimg_in_sciview(label_map * 125)
    return label_map


def conver_my_color_mask_with_black_background_to_labelmap(cv_img):
    # pass
    b, g, r = cv2.split(cv_img)
    h, w = cv_img.shape[:2]
    pointer_indexs = np.where((r > 150) & (g < 200))
    scale_indexs = np.where(g > 200)
    label_map = np.zeros((h, w), dtype=np.uint8)
    label_map[pointer_indexs] = 1
    label_map[scale_indexs] = 2
    # img_debug = label_map * 125
    # show_cvimg_in_sciview(img_debug)
    return label_map


# 判断单个文字检测结果是否在指定表的区域内，需要完全包括在内才可以
def is_text_in_this_meter_area(text, detect_res):
    #  text 分别是4个脚的xy坐标，分别是左上，右上，右下，左下，顺时针
    #  [394.2, 1445.4, 660.6, 1468.8, 646.2, 1623.6000000000001, 379.8, 1600.2, '1LV', 0.9991672]
    #  detect就是 左上和右下对角坐标
    # [ 1013, 274, 1644, 909, 0.98046875, 5 ]

    meter_min_x, meter_min_y, meter_max_x, meter_max_y = min(detect_res[0], detect_res[2]), min(detect_res[1],
                                                                                                detect_res[3]), \
                                                         max(detect_res[0], detect_res[2]), max(detect_res[1],
                                                                                                detect_res[3])
    for i, v in enumerate(text[:8]):
        if i % 2 == 0:
            # value of x
            if v < meter_min_x or v > meter_max_x:
                return False
        else:
            # value of y
            if v < meter_min_y or v > meter_max_y:
                return False
    return True


# 判断单个文字检测结果是否属于表的区域，只要在任意一个表内就为True
def is_text_in_all_meter_areas(text, detect_results):
    for det in detect_results:
        _r = is_text_in_this_meter_area(text=text, detect_res=det)
        if _r:
            return True
    return False


# 得到这个表区域内所有的结果
def get_this_meter_texts(texts, detect_res):
    rt = []
    for text in texts:
        _r = is_text_in_this_meter_area(text=text, detect_res=detect_res)
        if _r:
            rt.append(text)
    return rt


# 得到所有表内部的文字检测结果
def get_all_meter_texts(texts, detect_results):
    rt = []
    for text in texts:
        _r = is_text_in_all_meter_areas(text=text, detect_results=detect_results)
        if _r:
            rt.append(text)
    return rt


# 得到所有表【外部】的文字检测结果
def get_all_outer_texts(texts, detect_results):
    rt = []
    for text in texts:
        _r = is_text_in_all_meter_areas(text=text, detect_results=detect_results)
        if not _r:
            rt.append(text)
    return rt


if __name__ == "__main__":
    # img_path = os.path.join(BASE_DIR, 'exampleImages/01_standard_ammeter/IMG_20210222_100320_mask.png')
    # img = cv2.imread(img_path)
    # label_map = conver_my_color_mask_to_labelmap(img)
    # show_cvimg_in_sciview(label_map * 125)
    # print("OKK")

    img_dir = '/home/chenlei/data/Desktop/meter_recognition/label_0608/new_meter_cropped_1'
    for f in os.listdir(img_dir):
        if 'mask' not in f:
            continue
        img = cv2.imread(os.path.join(img_dir, f))
        label_map = conver_my_color_mask_to_labelmap(img)
        show_cvimg_in_sciview(label_map * 125)
        save_name = 'labelmap'.join(f.split('mask'))
        cv2.imwrite(os.path.join(img_dir, save_name), label_map)
