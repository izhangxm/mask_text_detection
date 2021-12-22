#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# @date 2021/11/2
# @fileName colortool.py
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
from math import pow
from colormath import color_objects
from colormath import color_conversions
import math


def bgr2lab(bgr_pixel):
    r = bgr_pixel[2] / 255.0  # rgb range: 0 ~ 1
    g = bgr_pixel[1] / 255.0
    b = bgr_pixel[0] / 255.0

    # gamma 2.2
    if r > 0.04045:
        r = pow((r + 0.055) / 1.055, 2.4)
    else:
        r = r / 12.92

    if g > 0.04045:
        g = pow((g + 0.055) / 1.055, 2.4)
    else:
        g = g / 12.92

    if b > 0.04045:
        b = pow((b + 0.055) / 1.055, 2.4)
    else:
        b = b / 12.92

    # sRGB
    X = r * 0.436052025 + g * 0.385081593 + b * 0.143087414
    Y = r * 0.222491598 + g * 0.716886060 + b * 0.060621486
    Z = r * 0.013929122 + g * 0.097097002 + b * 0.714185470

    # XYZ range: 0~100
    X = X * 100.000
    Y = Y * 100.000
    Z = Z * 100.000

    # Reference White Point

    ref_X = 96.4221
    ref_Y = 100.000
    ref_Z = 82.5211

    X = X / ref_X
    Y = Y / ref_Y
    Z = Z / ref_Z

    # Lab
    if X > 0.008856:
        X = pow(X, 1 / 3.000)
    else:
        X = (7.787 * X) + (16 / 116.000)

    if Y > 0.008856:
        Y = pow(Y, 1 / 3.000)
    else:
        Y = (7.787 * Y) + (16 / 116.000)

    if Z > 0.008856:
        Z = pow(Z, 1 / 3.000)
    else:
        Z = (7.787 * Z) + (16 / 116.000)

    Lab_L = round((116.000 * Y) - 16.000, 2)
    Lab_a = round(500.000 * (X - Y), 2)
    Lab_b = round(200.000 * (Y - Z), 2)
    return Lab_L, Lab_a, Lab_b

def bgr2Labobject(bgr_pixel):
    rgb_color = color_objects.sRGBColor(bgr_pixel[2], bgr_pixel[1], bgr_pixel[0])
    lab_color = color_conversions.convert_color(rgb_color, color_objects.LabColor)
    return lab_color


def colorDistance(bgr_pixel1, bgr_pixel2):
    B_1, G_1, R_1, = bgr_pixel1
    B_2, G_2, R_2, = bgr_pixel2
    rmean = (R_1 + R_2) / 2
    R = R_1 - R_2
    G = G_1 - G_2
    B = B_1 - B_2
    return math.sqrt((2 + rmean / 256) * (R ** 2) + 4 * (G ** 2) + (2 + (255 - rmean) / 256) * (B ** 2))


from colormath.color_diff import delta_e_cie1976
from colormath.color_objects import LabColor


def color_delta_e(bgr1, bgr2):
    lab1 = bgr2Labobject(bgr1)
    lab2 = bgr2Labobject(bgr2)
    delta_e = delta_e_cie1976(lab1, lab2)

    return delta_e



if __name__ == '__main__':
    p = [0, 0, 255]
    print("Lab: " + str(p))
    lab = bgr2lab(p)
    print("Lab: " + str(lab))

    print(colorDistance([128, 128, 255], [158, 128, 255]))
    print(colorDistance([128, 128, 255], [29, 128, 255]))
    print(colorDistance([128, 128, 255], [0, 0, 255]))

    print('*' * 20)

    print(color_delta_e([128, 128, 255], [158, 128, 255]))
    print(color_delta_e([128, 128, 255], [29, 128, 255]))
    print(color_delta_e([128, 128, 255], [0, 0, 255]))
    print(color_delta_e([0, 0, 0], [0, 0, 255]))

    print("- " * 20)
    print(color_delta_e([139, 109, 169], [0, 0, 255]))
    print(color_delta_e([94, 98, 78], [0, 0, 255]))

    print(color_delta_e([139, 109, 169], [0, 0, 0]))
    print(color_delta_e([94, 98, 78], [0, 0, 0]))
