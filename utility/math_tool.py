#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# @date 2021/11/3
# @fileName math_tool.py
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


def xiangxian(x, y):
    if y == 0 and x > 0:
        return "A"
    if x == 0 and y > 0:
        return "B"
    if y == 0 and x < 0:
        return "C"
    if x == 0 and y < 0:
        return "D"

    if x > 0 and y > 0:
        return 1
    elif x > 0 and y < 0:
        return 4
    elif x < 0 and y < 0:
        return 3
    elif x < 0 and y > 0:
        return 2


def calc_theta(_x, _y, center):
    x0, y0 = center
    x, y = _x - x0, _y - y0
    xx = xiangxian(x, y)
    table = {"A": 0, "B": 90, "C": 180, "D": 270}
    if xx in table:
        return table[xx]
    r = np.sqrt((x - 0) ** 2 + (y - 0) ** 2)
    if xx == 1 or xx == 4:
        _theta = (np.arctan(y / x) / (2 * np.pi) * 360 + 360) % 360
    elif xx == 2:
        _theta = np.arccos(x / r) / (2 * np.pi) * 360
    else:
        _theta = 180 - np.arcsin(y / r) / (2 * np.pi) * 360
    return _theta


def calculate_point_angle_with_positiveX_use_center(point, center=[0,0]):
    return calc_theta(point[0], point[1], center)


def get_my_cmp(center, start_angle):
    def _cmp(ele1, ele2):
        x1, y1 = ele1['p'][0], ele1['p'][1]
        x2, y2 = ele2['p'][0], ele2['p'][1]
        _theta1 = calc_theta(x1, y1, center)
        _theta2 = calc_theta(x2, y2, center)
        theta1 = (_theta1 - start_angle + 360) % 360
        theta2 = (_theta2 - start_angle + 360) % 360
        return theta1 - theta2

    return _cmp
