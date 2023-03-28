#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author tccw
# @date 2021/4/9
# @fileName fitTools.py
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
from scipy import optimize


def fitCircle(X, Y):
    """
    :param sample_points: which shape is [-1, 2]
    :type sample_points:
    :return:
    :rtype:
    """

    def loss_f(center, X, Y):
        x1, y1 = center
        r_s = np.sqrt((X - x1) ** 2 + (Y - y1) ** 2)
        res = r_s - r_s.mean()
        return res

    center_init = np.array([np.mean(X), np.mean(Y)])

    # noinspection PyTupleAssignmentBalance
    center, _ = optimize.leastsq(loss_f, center_init, args=(X, Y))
    r_s = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    r = r_s.mean()
    return center, r


def circle2PolyVals(center, r, X):
    c_x, c_y = center
    points = []
    for x in X:
        p2 = 1
        p1 = -2 * c_y
        p0 = c_y ** 2 + (x - c_x) ** 2 - r ** 2
        y1, y2 = np.roots([p2, p1, p0])
        if not isinstance(y1, complex):
            points.append([x, y1])
        if not isinstance(y2, complex):
            points.append([x, y2])
    return np.array(points, dtype=np.int32)


def circle_polt_vals(center, r, start_angle, end_angle, delta=0.5):
    c_x, c_y = center
    # 方式一
    # points = []
    # for ang in np.arange(start_angle, end_angle+delta, delta):
    #     theta = ang/360 * 2 * np.pi
    #     x = c_x + r * np.cos(theta)
    #     y = c_y + r * np.sin(theta)
    #     points.append([x, y])

    # 优化为numpy的方式
    angles = np.arange(start_angle, end_angle + delta, delta)
    theta = angles / 360 * 2 * np.pi
    X = c_x + r * np.cos(theta)
    Y = c_y + r * np.sin(theta)
    p_s = np.vstack([X, Y]).T.astype(np.int32)

    return p_s


def color_labelmap(label_map):
    rec_h, rec_w = label_map.shape[:2]
    color_img = np.zeros((rec_h, rec_w, 3), dtype=np.uint8)
    point_indexs = np.where(label_map == 1)
    scale_indexs = np.where(label_map == 2)
    color_img[point_indexs] = (0, 255, 0)
    color_img[scale_indexs] = (255, 0, 0)
    return color_img


def test_fit_tool():
    pass
    from CoreApp.utils import image_tool
    import cv2

    img_path = "/Users/Simon/WorkStation/PycharmProjects/MeterReader/exampleImages/02_standard_voltmeter/IMG_20210317_153353_BURST0095_labelmap.png"

    label_map = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    s_i = np.where(label_map == 2)

    scale_label_map = np.zeros(label_map.shape, dtype=np.uint8)
    scale_label_map[s_i] = 2

    s_X, s_Y = s_i[1], s_i[0]
    (c_x, c_y), c_r = fitCircle(s_X, s_Y)

    vis_img = color_labelmap(scale_label_map)

    cv2.circle(vis_img, (int(c_x), int(c_y)), int(c_r), (0, 255, 0))

    image_tool.show_cvimg_in_sciview(vis_img)


if __name__ == '__main__':
    test_fit_tool()

    # p_s = circle_polt_vals((0,0),10,0,90,1)
