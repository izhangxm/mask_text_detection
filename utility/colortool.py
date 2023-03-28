#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author tccw
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
import math
import random
from math import pow

import cv2
import numpy as np
from colormath import color_conversions
from colormath import color_objects
from colormath.color_diff import delta_e_cie1976
from skimage.color import rgb2lab

from .timetool import get_timer


def get_rgb_array(hex_str_list):
    res = []
    for ss in hex_str_list:
        if ss.startswith("#"):
            ss = ss[1:]
        rgb = [int(ss[0:2], 16), int(ss[2:4], 16), int(ss[4:6], 16)]
        res.append(rgb)
    return np.array(res, dtype=np.uint8)


def get_colorcode_map(n_gray=16):
    rgb = [[255, 182, 193], [255, 192, 203], [220, 20, 60], [255, 240, 245], [219, 112, 147], [255, 105, 180],
           [255, 20, 147], [199, 21, 133], [218, 112, 214], [216, 191, 216], [221, 160, 221], [238, 130, 238],
           [255, 0, 255], [255, 0, 255], [139, 0, 139], [128, 0, 128], [186, 85, 211], [148, 0, 211], [153, 50, 204],
           [75, 0, 130], [138, 43, 226], [147, 112, 219], [123, 104, 238], [106, 90, 205], [72, 61, 139],
           [230, 230, 250], [248, 248, 255], [0, 0, 255], [0, 0, 205], [25, 25, 112], [0, 0, 139], [0, 0, 128],
           [65, 105, 225], [100, 149, 237], [176, 196, 222], [119, 136, 153], [112, 128, 144], [30, 144, 255],
           [240, 248, 255], [70, 130, 180], [135, 206, 250], [135, 206, 235], [0, 191, 255], [173, 216, 230],
           [176, 224, 230], [95, 158, 160], [240, 255, 255], [224, 255, 255], [175, 238, 238], [0, 255, 255],
           [0, 255, 255], [0, 206, 209], [47, 79, 79], [0, 139, 139], [0, 128, 128], [72, 209, 204], [32, 178, 170],
           [64, 224, 208], [127, 255, 212], [102, 205, 170], [0, 250, 154], [245, 255, 250], [0, 255, 127],
           [60, 179, 113], [46, 139, 87], [240, 255, 240], [144, 238, 144], [152, 251, 152], [143, 188, 143],
           [50, 205, 50], [0, 255, 0], [34, 139, 34], [0, 128, 0], [0, 100, 0], [127, 255, 0], [124, 252, 0],
           [173, 255, 47], [85, 107, 47], [154, 205, 50], [107, 142, 35], [245, 245, 220], [250, 250, 210],
           [255, 255, 240], [255, 255, 224], [255, 255, 0], [128, 128, 0], [189, 183, 107], [255, 250, 205],
           [238, 232, 170], [240, 230, 140], [255, 215, 0], [255, 248, 220], [218, 165, 32], [184, 134, 11],
           [255, 250, 240], [253, 245, 230], [245, 222, 179], [255, 228, 181], [255, 165, 0], [255, 239, 213],
           [255, 235, 205], [255, 222, 173], [250, 235, 215], [210, 180, 140], [222, 184, 135], [255, 228, 196],
           [255, 140, 0], [250, 240, 230], [205, 133, 63], [255, 218, 185], [244, 164, 96], [210, 105, 30],
           [139, 69, 19], [255, 245, 238], [160, 82, 45], [255, 160, 122], [255, 127, 80], [255, 69, 0],
           [233, 150, 122], [255, 99, 71], [255, 228, 225], [250, 128, 114], [255, 250, 250], [240, 128, 128],
           [188, 143, 143], [205, 92, 92], [255, 0, 0], [165, 42, 42], [178, 34, 34], [139, 0, 0], [128, 0, 0],
           [255, 255, 255], [245, 245, 245], [220, 220, 220], [211, 211, 211], [192, 192, 192], [169, 169, 169],
           [128, 128, 128], [105, 105, 105], [0, 0, 0]]

    hex_rgb_list = ["#000000", "#F5F5F5", "#DCDCDC", "#D3D3D3", "#C0C0C0", "#A9A9A9", "#808080", "#696969",
                    "#000033", "#000066", "#000099", "#0000CC", "#0000FF", "#003300", "#003333", "#003366",
                    "#003399", "#0033CC", "#0033FF", "#006600", "#006633", "#006666", "#006699", "#0066CC", "#0066FF",
                    "#009900", "#009933", "#009966", "#009999", "#0099CC", "#0099FF", "#00CC00", "#00CC33", "#00CC66",
                    "#00CC99", "#00CCCC", "#00CCFF", "#00FF00", "#00FF33", "#00FF66", "#00FF99", "#00FFCC", "#00FFFF",
                    "#330000", "#330033", "#330066", "#330099", "#3300CC", "#3300FF", "#333300", "#333333", "#333366",
                    "#333399", "#3333CC", "#3333FF", "#336600", "#336633", "#336666", "#336699", "#3366CC", "#3366FF",
                    "#339900", "#339933", "#339966", "#339999", "#3399CC", "#3399FF", "#33CC00", "#33CC33", "#33CC66",
                    "#33CC99", "#33CCCC", "#33CCFF", "#33FF00", "#33FF33", "#33FF66", "#33FF99", "#33FFCC", "#33FFFF",
                    "#660000", "#660033", "#660066", "#660099", "#6600CC", "#6600FF", "#663300", "#663333", "#663366",
                    "#663399", "#6633CC", "#6633FF", "#666600", "#666633", "#666666", "#666699", "#6666CC", "#6666FF",
                    "#669900", "#669933", "#669966", "#669999", "#6699CC", "#6699FF", "#66CC00", "#66CC33", "#66CC66",
                    "#66CC99", "#66CCCC", "#66CCFF", "#66FF00", "#66FF33", "#66FF66", "#66FF99", "#66FFCC", "#66FFFF",
                    "#990000", "#990033", "#990066", "#990099", "#9900CC", "#9900FF", "#993300", "#993333", "#993366",
                    "#993399", "#9933CC", "#9933FF", "#996600", "#996633", "#996666", "#996699", "#9966CC", "#9966FF",
                    "#999900", "#999933", "#999966", "#999999", "#9999CC", "#9999FF", "#99CC00", "#99CC33", "#99CC66",
                    "#99CC99", "#99CCCC", "#99CCFF", "#99FF00", "#99FF33", "#99FF66", "#99FF99", "#99FFCC", "#99FFFF",
                    "#CC0000", "#CC0033", "#CC0066", "#CC0099", "#CC00CC", "#CC00FF", "#CC3300", "#CC3333", "#CC3366",
                    "#CC3399", "#CC33CC", "#CC33FF", "#CC6600", "#CC6633", "#CC6666", "#CC6699", "#CC66CC", "#CC66FF",
                    "#CC9900", "#CC9933", "#CC9966", "#CC9999", "#CC99CC", "#CC99FF", "#CCCC00", "#CCCC33", "#CCCC66",
                    "#CCCC99", "#CCCCCC", "#CCCCFF", "#CCFF00", "#CCFF33", "#CCFF66", "#CCFF99", "#CCFFCC", "#CCFFFF",
                    "#FF0000", "#FF0033", "#FF0066", "#FF0099", "#FF00CC", "#FF00FF", "#FF3300", "#FF3333", "#FF3366",
                    "#FF3399", "#FF33CC", "#FF33FF", "#FF6600", "#FF6633", "#FF6666", "#FF6699", "#FF66CC", "#FF66FF",
                    "#FF9900", "#FF9933", "#FF9966", "#FF9999", "#FF99CC", "#FF99FF", "#FFCC00", "#FFCC33", "#FFCC66",
                    "#FFCC99", "#FFCCCC", "#FFCCFF", "#FFFF00", "#FFFF33", "#FFFF66", "#FFFF99", "#FFFFCC", "#FFFFFF"]

    ant_design_hex = ["f5f5f5", "f0f0f0", "d9d9d9", "bfbfbf", "8c8c8c", "595959", "434343", "262626", "1f1f1f",
                      "141414", "fff1f0", "ffccc7", "ffa39e", "ff7875", "ff4d4f", "f5222d", "cf1322", "a8071a",
                      "820014", "5c0011", "fffbe6", "fff1b8", "ffe58f", "ffd666", "ffc53d", "faad14", "d48806",
                      "ad6800", "874d00", "613400", "feffe6", "ffffb8", "fffb8f", "fff566", "ffec3d", "fadb14",
                      "d4b106", "ad8b00", "876800", "614700", "f6ffed", "d9f7be", "b7eb8f", "95de64", "73d13d",
                      "52c41a", "389e0d", "237804", "135200", "092b00", "e6fffb", "b5f5ec", "87e8de", "5cdbd3",
                      "36cfc9", "13c2c2", "08979c", "006d75", "00474f", "002329", "e6f7ff", "bae7ff", "91d5ff",
                      "69c0ff", "40a9ff", "1890ff", "096dd9", "0050b3", "003a8c", "002766", "f9f0ff", "efdbff",
                      "d3adf7", "b37feb", "9254de", "722ed1", "531dab", "391085", "22075e", "120338", "fff0f6",
                      "ffd6e7", "ffadd2", "ff85c0", "f759ab", "eb2f96", "c41d7f", "9e1068", "780650", "520339"]
    # rgb = get_rgb_array(hex_rgb_list)
    #  work
    # rgb = [[255,0,0],[0,255,0],[0,0,255],
    #        [255, 255, 255], [245, 245, 245], [220, 220, 220], [211, 211, 211], [192, 192, 192], [169, 169, 169],
    #        [128, 128, 128], [105, 105, 105], [0, 0, 0]]
    #
    # rgb = [[255, 255, 255], [245, 245, 245], [220, 220, 220], [211, 211, 211], [192, 192, 192], [169, 169, 169],
    #        [128, 128, 128], [105, 105, 105], [0, 0, 0]]
    if n_gray is not None:
        rgb = []
        _margin = math.ceil(256/n_gray)
        _s = -_margin
        F = True
        while F:
            _s += _margin
            if _s > 255:
                _s = 255
                F = False
            rgb.append([_s, _s, _s])

    # for i in range(0, 256, math.ceil(256/n_gray)):
    #     rgb.append([i,i,i])
    rgb = np.array(rgb, dtype=np.uint8)
    bgr = rgb[:, ::-1]
    lab = rgb2lab(rgb)

    names = ["浅粉红", "粉红", "深红(猩红)", "淡紫红", "弱紫罗兰红", "热情的粉红", "深粉红", "中紫罗兰红", "暗紫色(兰花紫)", "蓟色", "洋李色(李子紫)", "紫罗兰",
             "洋红(玫瑰红)", "紫红(灯笼海棠)", "深洋红", "紫色", "中兰花紫", "暗紫罗兰", "暗兰花紫", "靛青/紫兰色", "蓝紫罗兰", "中紫色", "中暗蓝色(中板岩蓝)",
             "石蓝色(板岩蓝)", "暗灰蓝色(暗板岩蓝)", "淡紫色(熏衣草淡紫)", "幽灵白", "纯蓝", "中蓝色", "午夜蓝", "暗蓝色", "海军蓝", "皇家蓝/宝蓝", "矢车菊蓝", "亮钢蓝",
             "亮蓝灰(亮石板灰)", "灰石色(石板灰)", "闪兰色(道奇蓝)", "爱丽丝蓝", "钢蓝/铁青", "亮天蓝色", "天蓝色", "深天蓝", "亮蓝", "粉蓝色(火药青)", "军兰色(军服蓝)",
             "蔚蓝色", "淡青色", "弱绿宝石", "青色", "浅绿色(水色)", "暗绿宝石", "暗瓦灰色(暗石板灰)", "暗青色", "水鸭色", "中绿宝石", "浅海洋绿", "绿宝石", "宝石碧绿",
             "中宝石碧绿", "中春绿色", "薄荷奶油", "春绿色", "中海洋绿", "海洋绿", "蜜色(蜜瓜色)", "淡绿色", "弱绿色", "暗海洋绿", "闪光深绿", "闪光绿", "森林绿", "纯绿",
             "暗绿色", "黄绿色(查特酒绿)", "草绿色(草坪绿_", "绿黄色", "暗橄榄绿", "黄绿色", "橄榄褐色", "米色/灰棕色", "亮菊黄", "象牙色", "浅黄色", "纯黄", "橄榄",
             "暗黄褐色(深卡叽布)", "柠檬绸", "灰菊黄(苍麒麟色)", "黄褐色(卡叽布)", "金色", "玉米丝色", "金菊黄", "暗金菊黄", "花的白色", "老花色(旧蕾丝)", "浅黄色(小麦色)",
             "鹿皮色(鹿皮靴)", "橙色", "番木色(番木瓜)", "白杏色", "纳瓦白(土著白)", "古董白", "茶色", "硬木色", "陶坯黄", "深橙色", "亚麻布", "秘鲁色", "桃肉色",
             "沙棕色", "巧克力色", "重褐色(马鞍棕色)", "海贝壳", "黄土赭色", "浅鲑鱼肉色", "珊瑚", "橙红色", "深鲜肉/鲑鱼色", "番茄红", "浅玫瑰色(薄雾玫瑰)", "鲜肉/鲑鱼色",
             "雪白色", "淡珊瑚色", "玫瑰棕色", "印度红", "纯红", "棕色", "火砖色(耐火砖)", "深红色", "栗色", "纯白", "白烟", "淡灰色(庚斯博罗灰)", "浅灰色", "银灰色",
             "深灰色", "灰色", "暗淡的灰色", "纯黑"]
    names_cn = names
    names_en = ["LightPink", "Pink", "Crimson", "LavenderBlush", "PaleVioletRed", "HotPink", "DeepPink",
                "MediumVioletRed", "Orchid", "Thistle", "Plum", "Violet", "Magenta", "Fuchsia", "DarkMagenta", "Purple",
                "MediumOrchid", "DarkViolet", "DarkOrchid", "Indigo", "BlueViolet", "MediumPurple", "MediumSlateBlue",
                "SlateBlue", "DarkSlateBlue", "Lavender", "GhostWhite", "Blue", "MediumBlue", "MidnightBlue",
                "DarkBlue", "Navy", "RoyalBlue", "CornflowerBlue", "LightSteelBlue", "LightSlateGray", "SlateGray",
                "DodgerBlue", "AliceBlue", "SteelBlue", "LightSkyBlue", "SkyBlue", "DeepSkyBlue", "LightBlue",
                "PowderBlue", "CadetBlue", "Azure", "LightCyan", "PaleTurquoise", "Cyan", "Aqua", "DarkTurquoise",
                "DarkSlateGray", "DarkCyan", "Teal", "MediumTurquoise", "LightSeaGreen", "Turquoise", "Aquamarine",
                "MediumAquamarine", "MediumSpringGreen", "MintCream", "SpringGreen", "MediumSeaGreen", "SeaGreen",
                "Honeydew", "LightGreen", "PaleGreen", "DarkSeaGreen", "LimeGreen", "Lime", "ForestGreen", "Green",
                "DarkGreen", "Chartreuse", "LawnGreen", "GreenYellow", "DarkOliveGreen", "YellowGreen", "OliveDrab",
                "Beige", "LightGoldenrodYellow", "Ivory", "LightYellow", "Yellow", "Olive", "DarkKhaki", "LemonChiffon",
                "PaleGoldenrod", "Khaki", "Gold", "Cornsilk", "Goldenrod", "DarkGoldenrod", "FloralWhite", "OldLace",
                "Wheat", "Moccasin", "Orange", "PapayaWhip", "BlanchedAlmond", "NavajoWhite", "AntiqueWhite", "Tan",
                "BurlyWood", "Bisque", "DarkOrange", "Linen", "Peru", "PeachPuff", "SandyBrown", "Chocolate",
                "SaddleBrown", "Seashell", "Sienna", "LightSalmon", "Coral", "OrangeRed", "DarkSalmon", "Tomato",
                "MistyRose", "Salmon", "Snow", "LightCoral", "RosyBrown", "IndianRed", "Red", "Brown", "FireBrick",
                "DarkRed", "Maroon", "White", "WhiteSmoke", "Gainsboro", "LightGrey", "Silver", "DarkGray", "Gray",
                "DimGray", "Black"]

    return {"names": names, "names_cn": names_cn, "names_en": names_en, "bgr": bgr, "rgb": rgb, 'lab': lab}


def vis_cls_map(cls_map, n_gray=16):
    color_map = get_colorcode_map(n_gray=n_gray)
    mask = np.zeros([*cls_map.shape, 3], dtype=np.uint8)
    for idx, clr in enumerate(color_map['bgr']):
        ii = np.where(cls_map == idx)
        mask[ii] = clr
    return mask


def get_color_seg(img_bgr, use_lab=True, vis=True, n_gray=16, force_cpu=False):
    color_map = get_colorcode_map(n_gray=n_gray)
    img = img_bgr.copy()
    colors = color_map['bgr']
    timer = get_timer()

    if use_lab:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = rgb2lab(img)
        print(timer.cost())
        colors = color_map['lab']
    try:
        import torch
        dev = 'cpu' if force_cpu else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        timer.reset()
        colors = torch.tensor(colors)
        print("get colors tensor", timer.cost())
        colors = colors.to(dev)
        print("to(dev) colors", timer.cost())
        img = torch.tensor(img).to(dev)
        print("to(dev)", timer.cost())
        img_full = torch.unsqueeze(img, dim=2).repeat(1, 1, len(colors), 1)
        print("repeat", timer.cost())
        diff_color = img_full - colors
        print("diff_color", timer.cost())
        # dis_color = torch.sqrt(torch.sum(torch.square(diff_color), dim=3))
        dis_color = torch.sum(torch.square(diff_color), dim=3)
        print("sqrt", timer.cost())
        cls_map = dis_color.argmin(dim=2).cpu().numpy().astype(np.uint8)
        print("argmin", timer.cost())
    except ImportError:
        timer.reset()
        img_full = np.expand_dims(img, axis=2).repeat(len(colors), axis=2)
        print("repeat", timer.cost())
        diff_color = img_full - colors
        print("diff_color", timer.cost())
        dis_color = np.sqrt(np.sum(np.square(diff_color), axis=3))
        print("sqrt", timer.cost())
        cls_map = dis_color.argmin(axis=2).astype(np.uint8)
        print("argmin", timer.cost())

    mask = None
    if vis:
        mask = vis_cls_map(cls_map, n_gray)
    return mask, cls_map


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def remove_shadows(bgr_img):
    rgb_planes = cv2.split(bgr_img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    # image_tool.show_cvimg_in_sciview(result, result_norm)

    return result, result_norm


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


def color_delta_e(bgr1, bgr2):
    lab1 = bgr2Labobject(bgr1)
    lab2 = bgr2Labobject(bgr2)
    delta_e = delta_e_cie1976(lab1, lab2)

    return delta_e


def rand_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def t1():
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


COLOR_INFO = {}

if __name__ == '__main__':
    pass
