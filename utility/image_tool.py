#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# @date 2021/3/31
# @fileName image_tool.py
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
from matplotlib import gridspec
import numpy as np
import cv2
import math
import matplotlib.font_manager as font_manager
from matplotlib.font_manager import FontProperties
import io

def abandoned_my_plt_render_v1(imgs: [list, np.ndarray], titles=None):
    """
    generate combined image use matplotlib
    :param imgs:
    :type imgs:
    :param titles:
    :type titles:
    :return:
    :rtype:
    """
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    cols = 3
    if len(imgs) < 3:
        cols = len(imgs)
    rows = math.ceil(len(imgs) / cols)

    w = 1 * cols
    h = rows

    # gs = gridspec.GridSpec(rows, 3)
    figsize = (w, h * rows)
    # fig = plt.figure(figsize=figsize, dpi=200)

    # fig, fig_axes = plt.subplots(ncols=cols, nrows=rows, constrained_layout=True,  dpi=200)
    fig, fig_axes = plt.subplots(ncols=cols, nrows=rows, figsize=figsize, dpi=200)

    if isinstance(fig_axes, np.ndarray):
        fig_axes = fig_axes.reshape(-1)
    else:
        fig_axes = [fig_axes]
    for index, axes in enumerate(fig_axes):
        if index < len(imgs):
            title = ""
            if titles != None and index <= len(titles) - 1:
                title = titles[index]
            mask = imgs[index]
            if len(mask.shape) == 3:
                mask = mask[:, :, ::-1]
            axes.imshow(mask, cmap='gray')
            axes.set_title(title)
        else:
            axes.axis('off')
    # for index, mask in enumerate(imgs):
    #     try:
    #         title = ""
    #         if titles != None and index <= len(titles) - 1:
    #             title = titles[index]
    #         if len(mask.shape) == 3:
    #             mask = mask[:, :, ::-1]
    #         # fig.add_subplot(rows, 3, int(index + 1)).imshow(mask, cmap='gray'), plt.title(title)
    #
    #         r = int(index/3)
    #         c = index % 3
    #         f1_axes[r, c].imshow(mask, cmap='gray')
    #         f1_axes[r, c].set_title(title)
    #
    #     except Exception as e:
    #         raise Exception(e)
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=1, wspace=1)
    # plt.subplots_adjust( hspace=0.5, wspace=0.35)
    # plt.subplots_adjust( hspace=1, wspace=1)
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=1, top=1, hspace=0.2, wspace=0.35)
    # fig.tight_layout()
    plt.show()

    vis_image = pltfig2cvimg(fig)
    plt.close('all')
    return vis_image


def get_plt_images_figure(*args, titles=None, cols=3, cell_hw=(500, 500), max_width=None, font_size=14):
    """
        generate combined image use matplotlib
        :param imgs:
        :type imgs:
        :param titles:
        :type titles:
        :param cols:
        :type cols:
        :param cell_hw:
        :type cell_hw:
        :param max_width:
        :type max_width:
        :param font_size:
        :type font_size:
        :return:
        :rtype:
        """
    imgs = []
    for arg in args:
        if isinstance(arg, list):
            imgs += arg
        if isinstance(arg, np.ndarray):
            imgs.append(arg)
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
    gs = fig.add_gridspec(ncols=cols, nrows=rows)
    axes = gs.subplots()
    if isinstance(axes, np.ndarray):
        axes = axes.reshape(-1).tolist()
    else:
        axes = [axes]
    backgrounds = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axes]
    for index in range(rows * cols):
        r = int(index / cols)
        c = index % cols
        ax = axes[index]
        bg = backgrounds[index]
        fig.canvas.restore_region(bg)
        if index < len(imgs):
            title = ""
            if titles != None and index <= len(titles) - 1:
                title = titles[index]
            mask = imgs[index]

            # --------------- ------------------------------------------
            # x_step = int(mask.shape[1] / 5)
            # y_setp = int(mask.shape[0] / 5)
            # plt.xticks(np.arange(0, mask.shape[1], x_step), fontproperties=font)
            # plt.yticks(np.arange(0, mask.shape[0], y_setp), fontproperties=font)

            # 以下两行在多线程读表时会报错，因此暂时注释之
            # plt.xticks(fontproperties=font)
            # plt.yticks(fontproperties=font)
            # --------------- ------------------------------------------

            if len(mask.shape) == 3:
                mask = mask[:, :, ::-1]
            ax.imshow(mask, cmap='gray')
            ax.set_title(title, fontproperties=title_font)
        else:
            ax.axis('off')
        fig.canvas.blit(ax.bbox)
    return fig


def make_img_border(image, dst_hw, border_size=10):
    top, bottom, left, right = (0, 0, 0, 0)
    h, w, _ = image.shape
    longest_edge = max(h, w)
    if h < longest_edge:
        pass

# TODO
def get_concat_cv_image(*args, cols=3, cell_hw=(500, 500), max_width=None):
    """
        generate combined image use matplotlib
        :param imgs:
        :type imgs:
        :param titles:
        :type titles:
        :param cols:
        :type cols:
        :param cell_hw:
        :type cell_hw:
        :param max_width:
        :type max_width:
        :param font_size:
        :type font_size:
        :return:
        :rtype:
        """
    imgs = []
    for arg in args:
        if isinstance(arg, list):
            imgs += arg
        if isinstance(arg, np.ndarray):
            imgs.append(arg)
    if len(imgs) < cols:
        cols = len(imgs)
    rows = math.ceil(len(imgs) / cols)

    cell_h, cell_w = cell_hw
    w = cell_w * cols
    h = cell_h * rows

    res_img = None
    for r in range(rows):
        row_img = None
        for c in range(cols):
            _img = imgs[r * cols] + c

            if row_img is None:
                row_img = _img
            else:
                pass

    return

def my_plt_render(imgs: [list, np.ndarray], titles=None, cols=3, cell_hw=(500, 500), max_width=None, font_size=14, mode='plt'):
    """
    generate combined image use matplotlib
    :param imgs:
    :type imgs:
    :param titles:
    :type titles:
    :param cols:
    :type cols:
    :param cell_hw:
    :type cell_hw:
    :param max_width:
    :type max_width:
    :param font_size:
    :type font_size:
    :return:
    :rtype:
    """
    fig = get_plt_images_figure(imgs, titles=titles, cols=cols, cell_hw=cell_hw, max_width=max_width, font_size=font_size)
    vis_image = pltfig2cvimg(fig)
    return vis_image

def pltfig2cvimg_v1(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)[:, :, :3][:, :, ::-1]
    return image

def pltfig2cvimg(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = data.reshape((int(h), int(w), -1))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_cvimg_in_sciview_v1(*args, names=None, cols=3, cell_hw=(500, 500), max_width=None, font_size=14, **kwargs):
    imgs = []
    for arg in args:
        if isinstance(arg, list):
            imgs += arg
        if isinstance(arg, np.ndarray):
            imgs.append(arg)
    if isinstance(names, str):
        names = [names]
    if len(imgs) > 1:
        vis_img = my_plt_render(imgs, names, cols=cols, cell_hw=cell_hw, max_width=max_width, font_size=font_size)
    else:
        vis_img = imgs[0]
    h, w = vis_img.shape[:2]
    if max_width and w > max_width:
        h = int(max_width * h / w)
        w = int(max_width)
    dpi = 100
    figsize = (w / dpi, h / dpi)
    fig, fig_axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True, figsize=figsize)

    if len(vis_img.shape) == 3:
        fig_axes.imshow(vis_img[:, :, ::-1], cmap='gray')
    else:
        fig_axes.imshow(vis_img, cmap='gray')
    fig_axes.axis('off')
    plt.show()
    plt.close('all')

def show_cvimg_in_sciview(*args, names=None, cols=3, cell_hw=(500, 500), max_width=None, font_size=14, **kwargs):
    imgs = []
    for arg in args:
        if isinstance(arg, list):
            imgs += arg
        if isinstance(arg, np.ndarray):
            imgs.append(arg)
    if isinstance(names, str):
        names = [names]
    if len(imgs) > 1:
        fig = get_plt_images_figure(*args, titles=names, cols=cols, cell_hw=cell_hw, max_width=max_width, font_size=font_size)
    else:
        vis_img = imgs[0]
        cell_hw = vis_img.shape[:2]
        fig = get_plt_images_figure(*args, titles=names, cols=1, cell_hw=cell_hw, max_width=max_width, font_size=font_size)
    fig.show()
    return fig

def show_ready_plt():
    plt.show()

def show_ready_plt_in_sciview():
    plt.show()
    plt.close('all')

def sector2line_v1(img: np.ndarray, c_x, c_y, radius, ring_stride, start_angle=0, end_angle=360):
    """
    convert sector to line rectangle
    :param img: a cv image
    :type img:
    :param c_x: X of rotation center in cv xoy coordinate system
    :type c_x:
    :param c_y:  y of rotation center in cv xoy coordinate system
    :type c_y:
    :param radius:
    :type radius:
    :param start_angle: radian angle begin in cv xoy coordinate system
    :type start_angle:
    :param end_angle: radian angle end in cv xoy coordinate system
    :type end_angle:
    :return: a rectangular cv img
    :rtype:
    """
    # convert
    if end_angle < start_angle:
        end_angle += 360

    start_angle = start_angle / 360 * 2 * math.pi

    end_angle = end_angle / 360 * 2 * math.pi

    angle_range = end_angle - start_angle
    radius = int(radius)
    rec_w = int(angle_range * radius)
    rec_h = int(ring_stride)
    h, w = img.shape[:2]
    if len(img.shape) > 2:
        rectangle_img = np.zeros((rec_h, rec_w, 3), dtype=np.uint8)
    else:
        rectangle_img = np.zeros((rec_h, rec_w), dtype=np.uint8)
    for row in range(rec_h):
        for col in range(rec_w):
            theta = angle_range / rec_w * col + start_angle
            theta = theta % (2 * math.pi)
            rho = radius - row
            x = min(int(c_x + rho * math.cos(theta) - 0.5), w - 1)
            y = min(int(c_y + rho * math.sin(theta) - 0.5), h - 1)
            x, y = max(x, 0), max(y, 0)
            rectangle_img[row, col] = img[y, x]
    return rectangle_img

def sector2line(img: np.ndarray, c_x, c_y, radius, ring_stride, start_angle=0, end_angle=360):
    """
    convert sector to line rectangle
    :param img: a cv image
    :type img:
    :param c_x: X of rotation center in cv xoy coordinate system
    :type c_x:
    :param c_y:  y of rotation center in cv xoy coordinate system
    :type c_y:
    :param radius:
    :type radius:
    :param start_angle: radian angle begin in cv xoy coordinate system
    :type start_angle:
    :param end_angle: radian angle end in cv xoy coordinate system
    :type end_angle:
    :return: a rectangular cv img
    :rtype:
    """
    # convert
    if end_angle < start_angle:
        end_angle += 360

    start_angle = start_angle / 360 * 2 * math.pi

    end_angle = end_angle / 360 * 2 * math.pi

    angle_range = end_angle - start_angle
    radius = int(radius)
    rec_w = int(angle_range * radius)
    rec_h = int(ring_stride)
    h, w = img.shape[:2]
    if len(img.shape) > 2:
        rectangle_img = np.zeros((rec_h, rec_w, 3), dtype=np.uint8)
    else:
        rectangle_img = np.zeros((rec_h, rec_w), dtype=np.uint8)

    # ========= 旧方式：1700ms 左右 ===================
    # for row in range(rec_h):
    #     for col in range(rec_w):
    #         theta = angle_range / rec_w * col + start_angle
    #         theta = theta % (2 * math.pi)
    #         rho = radius - row
    #         x = min(int(c_x + rho * math.cos(theta) - 0.5), w - 1)
    #         y = min(int(c_y + rho * math.sin(theta) - 0.5), h - 1)
    #         x, y = max(x, 0), max(y, 0)
    #         rectangle_img[row, col] = img[y, x]

    # ========== 新方法：缩减到1/10左右 ==========
    # 创建rectangle_img的顺序索引矩阵
    # 注意按照循环的顺序进行创建，所以会复杂些
    _t = np.ones((rec_w, rec_h)) * np.arange(0, rec_h)
    ROW = _t.T.reshape(-1).astype(np.int32)
    COL = np.mod(np.arange(0, rec_h * rec_w), rec_w).astype(np.int32)

    # 计算theta 和 rho
    THETA = COL * (angle_range / rec_w) + start_angle
    THETA = np.mod(THETA, 2 * np.pi)
    RHO = radius - ROW

    X = c_x + np.cos(THETA) * RHO - 0.5
    X = X.astype(np.int32)
    X = np.minimum(X, np.ones_like(X) * (w - 1))
    X = np.maximum(X, np.zeros_like(X))

    Y = c_y + np.sin(THETA) * RHO - 0.5
    Y = Y.astype(np.int32)
    Y = np.minimum(Y, np.ones_like(Y) * (h - 1))
    Y = np.maximum(Y, np.zeros_like(Y))

    rectangle_img[(ROW, COL)] = img[(Y, X)]

    return rectangle_img

# --------------------------------------------------------------------------------------------------------------------

def constrain_resize_hw(src_hw: [list, tuple], max_hw: [list, tuple]) -> [list, tuple]:
    h, w = src_hw
    if h > max_hw or w > max_hw:
        if h > w:
            new_h = max_hw
            new_w = int(w * new_h / h)
        else:
            new_w = max_hw
            new_h = int(h * new_w / w)
        return new_h, new_w
    else:
        return h, w


def resize_image(src_image: np.ndarray, dst_hw: [list, tuple], method=cv2.INTER_LINEAR) -> np.ndarray:
    h, w = dst_hw
    sr_h, sr_w = src_image.shape[:2]
    if sr_h == h and sr_w == w:
        return src_image
    resized_img = cv2.resize(src_image, (w, h), interpolation=method)
    return resized_img

def resize_text_info(src_image_hw: [list, tuple], text_res: [list, None], dst_hw: [list, tuple]) -> [list, None]:
    if text_res is None:
        return None
    if len(text_res) == 0:
        return []
    assert len(src_image_hw) == 2, "length of src_image_hw should be equal than 2"
    src_h, src_w = src_image_hw
    dst_h, dst_w = dst_hw
    ratio_h, ratio_w = dst_h / src_h, dst_w / src_w
    new_res = []
    text_res_shape = np.array(text_res).shape
    if len(text_res_shape) != 2:
        text_res = [text_res]

    for info in text_res:
        _tmp = []
        for j, lo in enumerate(info[:8]):
            if j % 2 == 0:
                _tmp.append(lo * ratio_w)
            else:
                _tmp.append(lo * ratio_h)
        _tmp.append(info[8])
        _tmp.append(info[9])
        new_res.append(_tmp)
    if len(text_res_shape) != 2:
        new_res = new_res[0]
    return new_res


def resize_detect_info(src_image_hw: [list, tuple], detect_res: [list, None], dst_hw: [list, tuple]) -> [list, None]:
    assert len(src_image_hw) == 2, "length of src_image_hw should be equal than 2"
    src_h, src_w = src_image_hw
    dst_h, dst_w = dst_hw
    ratio_h, ratio_w = dst_h / src_h, dst_w / src_w
    new_res = []
    # detect_res = [[751.0, 44.0, 1159.0, 713.0, 0.8219207525253296, 0.0], [111.0, 199.0, 1069.0, 715.0, 0.5671095848083496, 0.0],
    #  [425.0, 429.0, 516.0, 717.0, 0.5414489507675171, 27.0]]
    if detect_res is None:
        return None
    if len(detect_res) == 0:
        return []

    det_shape = np.array(detect_res).shape
    if len(det_shape) != 2:
        detect_res = [detect_res]

    for info in detect_res:
        _tmp = []
        for j, lo in enumerate(info[:4]):
            if j % 2 == 0:
                _tmp.append(lo * ratio_w)
            else:
                _tmp.append(lo * ratio_h)
        _tmp.append(info[4])
        _tmp.append(info[5])
        new_res.append(_tmp)
    if len(det_shape) != 2:
        new_res = new_res[0]
    return new_res


def resize_image_with_detect_info(src_image: np.ndarray, detect_res: list, dst_hw: [list, tuple],
                                  method=cv2.INTER_LINEAR):
    src_image_hw = src_image.shape[:2]
    resized_img = resize_image(src_image=src_image, dst_hw=dst_hw)
    resized_detect_res = resize_detect_info(src_image_hw=src_image_hw, detect_res=detect_res, dst_hw=dst_hw)
    return resized_img, resized_detect_res


def resize_image_with_text_info(src_image: np.ndarray, text_res: list, dst_hw: [list, tuple], method=cv2.INTER_LINEAR):
    src_image_hw = src_image.shape[:2]
    resized_img = resize_image(src_image=src_image, dst_hw=dst_hw)
    resized_text_res = resize_text_info(src_image_hw=src_image_hw, text_res=text_res, dst_hw=dst_hw)
    return resized_img, resized_text_res


def resize_image_with_detect_and_text_info(src_image: np.ndarray, detect_res: list, text_res: list,
                                           dst_hw: [list, tuple], method=cv2.INTER_LINEAR) -> [list, tuple]:
    resized_img = resize_image(src_image=src_image, dst_hw=dst_hw)
    resized_detect_info = resize_detect_info(src_image_hw=src_image.shape[:2], detect_res=detect_res,
                                             dst_hw=dst_hw)
    resized_text_info = resize_text_info(src_image_hw=src_image.shape[:2], text_res=text_res, dst_hw=dst_hw)
    return resized_img, resized_detect_info, resized_text_info


def constrain_resize_image(src_image: np.ndarray, max_hw=720, method=cv2.INTER_LINEAR):
    h, w = constrain_resize_hw(src_hw=src_image.shape[:2], max_hw=max_hw)
    resized_img = resize_image(src_image=src_image, dst_hw=(h, w), method=method)
    return resized_img


def constrain_resize_image_with_detect_info(src_image: np.ndarray, detect_res: list, max_hw=720,
                                            method=cv2.INTER_LINEAR):
    src_image_hw = src_image.shape[:2]
    new_h, new_w = constrain_resize_hw(src_hw=src_image_hw, max_hw=max_hw)
    resized_img = resize_image(src_image=src_image, dst_hw=[new_h, new_w])
    resized_detect_res = resize_detect_info(src_image_hw=src_image_hw, detect_res=detect_res,
                                            dst_hw=[new_h, new_w])
    return resized_img, resized_detect_res


def constrain_resize_image_with_text_info(src_image: np.ndarray, text_res: list, max_hw=720, method=cv2.INTER_LINEAR):
    src_image_hw = src_image.shape[:2]
    new_h, new_w = constrain_resize_hw(src_hw=src_image_hw, max_hw=max_hw)
    resized_img = resize_image(src_image=src_image, dst_hw=[new_h, new_w])
    resized_text_res = resize_text_info(src_image_hw=src_image_hw, text_res=text_res, dst_hw=[new_h, new_w])
    return resized_img, resized_text_res


def constrain_resize_detect_info(src_image_hw: [list, tuple], detect_res: list, max_hw=720):
    assert len(src_image_hw) == 2, "length of src_image_hw should be equal than 2"
    new_h, new_w = constrain_resize_hw(src_hw=src_image_hw, max_hw=max_hw)
    resized_detect_res = resize_detect_info(src_image_hw=src_image_hw, detect_res=detect_res,
                                            dst_hw=[new_h, new_w])
    return resized_detect_res


def constrain_resize_text_info(src_image_hw: [list, tuple], text_res: list, max_hw=720):
    assert len(src_image_hw) == 2, "length of src_image_hw should be equal than 2"
    new_h, new_w = constrain_resize_hw(src_hw=src_image_hw, max_hw=max_hw)
    resized_text_res = resize_text_info(src_image_hw=src_image_hw, text_res=text_res, dst_hw=[new_h, new_w])
    return resized_text_res


def constrain_resize_image_with_detect_and_text_info(src_image: np.ndarray, detect_res: list, text_res: list,
                                                     max_hw=720, method=cv2.INTER_LINEAR):
    hw = constrain_resize_hw(src_hw=src_image.shape[:2], max_hw=max_hw)
    return resize_image_with_detect_and_text_info(src_image=src_image, detect_res=detect_res,
                                                  text_res=text_res, dst_hw=hw)


def constrain_resize_image_to_approximate_template(src_image: np.ndarray, template_hw: list, method=cv2.INTER_LINEAR):
    """
    src_image: 原图中截取出来的待匹配的图像
    template_hw: 模板的高和宽
    函数作用：将待匹配的图像resize到接近模板分辨率大小
    注意：如果从整张原图中匹配目标图像，这个函数显然是不适用的。
    """
    src_h, src_w = src_image.shape[0], src_image.shape[1]
    t_h, t_w = template_hw[0], template_hw[1]
    if src_w / src_h < t_w / t_h:  # 判断以哪条边为基准进行缩放
        new_w = t_w
        new_h = int(src_h * (new_w / src_w))
    else:
        new_h = t_h
        new_w = int(src_w * (new_h / src_h))
    return cv2.resize(src_image, (new_w, new_h), interpolation=method)


def put_optimize_text(img, label="",  location=(0, 0), margin=(1, 1), color=(0, 0, 255), font_bold=1, font_size=1, bg_color=True, bg_padding=1, transparency=0.65):
    new_fc = font_size * 1.4 * (img.shape[1]) / (1000)

    standard_text_size, baseline = cv2.getTextSize(label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=font_bold)
    f_height = round(standard_text_size[1] * new_fc)
    f_width = round(standard_text_size[0] * new_fc)

    if f_height >= 16:
        font_bold = font_bold + 1

    new_loc = location[0] + margin[0] * f_height, location[1] + margin[1] * f_height + f_height
    bg_padding = bg_padding * f_height * 0.2
    c1_no_p = round(new_loc[0]), round(new_loc[1] - f_height)
    c2_no_p = round(new_loc[0] + f_width), round(new_loc[1])

    c1 = round(c1_no_p[0] - bg_padding), round(c1_no_p[1] - bg_padding)
    c2 = round(c2_no_p[0] + bg_padding), round(c2_no_p[1] + bg_padding)

    zeros_mask = np.zeros(img.shape, dtype=np.uint8)
    if isinstance(bg_color, list) or isinstance(bg_color, tuple) or ():
        zeros_mask = cv2.rectangle(zeros_mask, c1, c2, color=bg_color, thickness=-1)
    elif isinstance(bg_color, bool) and bg_color:
        bg_color = (255 - color[0], 255 - color[1], 255 - color[2])
        zeros_mask = cv2.rectangle(zeros_mask, c1, c2, color=bg_color, thickness=-1)

    zeros_mask = cv2.putText(zeros_mask, label, new_loc, cv2.FONT_HERSHEY_DUPLEX, fontScale=new_fc, color=color, thickness=font_bold, lineType=cv2.LINE_AA)

    gray = cv2.cvtColor(zeros_mask, cv2.COLOR_BGR2GRAY)
    a_i = np.where(gray != 0)

    result_area = img[a_i]*(1 - transparency) + zeros_mask[a_i] * transparency
    img[a_i] = result_area

    return img



# --------------------------------------------------------------------------------------------------------------------


def xxsector2line():
    circle_img_path = '../../exampleImages/other/orthogonal_scale_colormap.png'
    circle_img = cv2.imread(circle_img_path)
    c_x, c_y = 370, 370
    radius = 320
    line_img = sector2line(circle_img, c_x, c_y, radius, 100, 180, 270)
    show_cvimg_in_sciview(circle_img, line_img, cols=1)

    circle_img_path = '../../exampleImages/other/circle.jpg'
    circle_img = cv2.imread(circle_img_path)
    h, w = circle_img.shape[:2]
    c_x, c_y = w / 2, h / 2
    radius = w / 2 - 15
    line_img = sector2line(circle_img, c_x, c_y, radius, 110, 0, 360)
    show_cvimg_in_sciview(circle_img, line_img, cols=1)
    print("ok")


def cv_concat():
    import glob
    img_path_list = glob.glob('datasets01/all_text_db_full_size/train/**/*.jpg', recursive=True)
    img_path_list2 = glob.glob('datasets01/all_text_db_full_size/test/**/*.jpg', recursive=True)
    img_path_list += img_path_list2
    img_list = []
    for img_path in img_path_list:
        img_list.append(cv2.imread(img_path))
    import time

    start = time.time()
    fig = get_plt_images_figure(img_list)
    print(time.time() - start)

    start = time.time()
    cvimg = pltfig2cvimg(fig)
    print(time.time() - start)
    print("")


if __name__ == "__main__":
    cv_concat()
