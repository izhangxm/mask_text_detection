#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# @date 2021/12/22
# @fileName xx.py
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
import matplotlib.pyplot as plt
import numpy as np
import time
import glob
import cv2
import math
from matplotlib.font_manager import FontProperties


def func01():

    x = np.arange(0, 2*np.pi, 0.1)
    y = np.sin(x)

    fig, axes = plt.subplots(nrows=6)

    fig.show()

    # We need to draw the canvas before we start animating...
    fig.canvas.draw()

    styles = ['r-', 'g-', 'y-', 'm-', 'k-', 'c-']
    def plot(ax, style):
        return ax.plot(x, y, style, animated=True)[0]
    lines = [plot(ax, style) for ax, style in zip(axes, styles)]

    # Let's capture the background of the figure
    backgrounds = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axes]

    tstart = time.time()
    for i in range(1, 2000):
        items = enumerate(zip(lines, axes, backgrounds), start=1)
        for j, (line, ax, background) in items:
            fig.canvas.restore_region(background)
            line.set_ydata(np.sin(j*x + i/10.0))
            ax.draw_artist(line)

            fig.canvas.blit(ax.bbox)

    print('FPS:' , 2000/(time.time()-tstart))


def func02():


    img_path_list = glob.glob('datasets01/all_text_db_full_size/test/*.jpg')

    img_list = []
    for img_path in img_path_list:
        img_list.append(cv2.imread(img_path))

    imgs = img_list
    cols = 3
    font_size = 14
    cell_hw = [500, 500]
    max_width = None

    if len(imgs) < cols:
        cols = len(imgs)
    rows = math.ceil(len(imgs) / cols)

    cell_h, cell_w = cell_hw
    w = cell_w * cols
    h = cell_h * rows
    if max_width and w > max_width:
        h = int(max_width * h / w)
        w = int(max_width)
    dpi = 100

    font_size = int(font_size * 100 / dpi)
    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')
    font.set_size(font_size)

    title_font = font.copy()
    title_font.set_size(int(font_size * 1.6))

    figsize = (w / dpi, h / dpi)

    fig = plt.figure(constrained_layout=True, dpi=dpi, figsize=figsize)
    spec = fig.add_gridspec(ncols=cols, nrows=rows)


    print("")



def func03():

    img_path_list = glob.glob('datasets01/all_text_db_full_size/test/*.jpg')

    img_list = []
    for img_path in img_path_list:
        img_list.append(cv2.imread(img_path))

    fig, axes = plt.subplots(nrows=2, ncols=3)
    # We need to draw the canvas before we start animating...
    # fig.canvas.draw()

    axes = axes.reshape(-1)
    # Let's capture the background of the figure
    backgrounds = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axes.reshape(-1)]


    tstart = time.time()
    for i in range(1, 10):
        items = enumerate(zip(axes, backgrounds), start=0)
        for j, (ax, background) in items:
            fig.canvas.restore_region(background)
            ax.imshow(img_list[j], cmap='gray')
            fig.canvas.blit(ax.bbox)

    fig.show()
    print('FPS:' , 10/(time.time()-tstart))




if __name__ == '__main__':
    func03()

