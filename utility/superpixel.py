#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# @date 2021/12/24
# @fileName superpixel.py
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
import numpy as np
from skimage import segmentation

__all__ = ['vis_boundaries', 'vis_superpixel', 'get_boundaries_from_label_map', 'expand_map_to_img']


def vis_boundaries(img, b_map: np.ndarray, color=((253, 252, 58))):
    C = 1
    if len(img.shape) == 3:
        H, W, C = img.shape
        if len(b_map.shape) == 2:
            b_map = expand_map_to_img(b_map, c=C)
    if len(b_map.shape) == 3:
        b_map = b_map[:, :, 0]
    b_map = b_map.astype(np.bool8)

    if C == 1:
        img[b_map] = 0
    else:
        img[b_map] = color
    return img


def vis_superpixel(img_base, label_map: np.ndarray, color=((253, 252, 58))):
    b_map = get_boundaries_from_label_map(label_map, c=3)
    vis = vis_boundaries(img_base, b_map, color=color)
    return vis


def get_boundaries_from_label_map(label_map, c=3):
    boundaries_map = segmentation.find_boundaries(label_map)
    b_img_map = expand_map_to_img(boundaries_map, c)
    return b_img_map


def expand_map_to_img(a_map, c=3):
    masked_map = np.expand_dims(a_map, axis=2)
    masked_map = np.concatenate([masked_map for _ in range(c)], axis=2)
    return masked_map
