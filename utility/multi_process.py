#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author Chenlei
# @date 6/9/21
# @fileName multi_process.py
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

import multiprocessing
import threading

from CoreApp.utils import image_tool
from CoreApp.model.base import MeterBase
from MeterReader.settings import METER_DETECT_ENGINE_MANAGER
from MeterReader.settings import TEXT_ENGINE_MANAGER


def detect_core(meter_image, ori_image, src_hw, dst_hw, is_vis_detect, is_return_detect, temp_res):
    try:
        # timer.reset()
        det_results, _ = METER_DETECT_ENGINE_MANAGER.detect_meters(ori_image=meter_image, is_visualize=False)
        det_results = image_tool.resize_detect_info(src_hw, det_results, dst_hw)
        temp_res["det_results"] = det_results
        if is_vis_detect:
            vis_d_t = METER_DETECT_ENGINE_MANAGER.visualize(ori_image=ori_image, det_results=det_results)
            temp_res["vis_d_t"] = vis_d_t
        if is_return_detect:
            temp_res["resp_data"]["data"]["det_results"] = det_results
        # print("detect_meters", timer.cost())
    except Exception as e:
        raise Exception('detection model is failed:' + str(e))


def text_core(meter_image, ori_image, src_hw, dst_hw, is_vis_detect, is_vis_text, is_return_text, temp_res):
    try:
        # timer.reset()
        text_results, _ = TEXT_ENGINE_MANAGER.analyze_text(ori_image=meter_image, is_visualize=False)
        text_results = image_tool.resize_text_info(src_hw, text_results, dst_hw)
        temp_res["text_results"] = text_results

        if is_vis_text:
            vis_base = temp_res["vis_d_t"] if is_vis_detect else ori_image
            temp_res["vis_d_t"] = TEXT_ENGINE_MANAGER.visualize(vis_base, text_results)
        if is_return_text:
            temp_res["resp_data"]["data"]["text_results"] = text_results
        # print("text_results", timer.cost())
    except Exception as e:
        raise Exception('text model is failed:' + str(e))


def read_meter_core(info, src_hw, dst_hw, is_vis_process, vis_images, meter_infos):
    ######
    # 经过简单的试探，增加读表的时间，多进程能带来明显的提升，而多线程提升不明显
    # for n in range(100000000):
    #     pass
    ######

    location_info = image_tool.resize_detect_info(src_hw, info['location_info'], dst_hw)
    _tmp = {'meter_type': info['meter_type'],
            'location_info': location_info,
            }
    meter_reader: MeterBase = info['reader']
    meter_info = meter_reader.read_meter()
    _tmp['value'] = meter_info['value']
    if is_vis_process:
        _vis = meter_reader.get_visualized_cvimg()  # TODO: visualizing image costs too much time
        vis_images.append(_vis)
    meter_infos.append(_tmp)


# 不支持多核，共享数据比较简单。
# 在有些任务上，速度提升不明显（有点鸡肋）
class MyThread(threading.Thread):
    def __init__(self, fun, args):
        threading.Thread.__init__(self)
        self.fun = fun
        self.args = args
        self.sub_result = None

    def run(self) -> None:
        self.fun(*self.args)

    def get_res(self):
        return self.sub_result


# 多进程，支持多核，对于执行时间较长的子任务，速度提升较明显，但共享数据较麻烦
class MyProcess(multiprocessing.Process):
    def __init__(self, fun, args):
        multiprocessing.Process.__init__(self)
        self.fun = fun
        self.args = args
        self.sub_result = None

    def run(self) -> None:
        self.fun(*self.args)

    def get_res(self):
        return self.sub_result


def test_fun(a: int, b: list):
    for n in range(100000000):
        pass
    print(f"a = {a}")
    print(f"变化前： {b}")
    total = 0
    for n in range(101):
        total += n
    b.append(total)
    print(f"变化后： {b}")
    return b


def test_fun2(a):
    a.value = [91023012]


def test_multi_thread():
    threads = []
    c = [-1]
    for i in range(8):
        t = MyThread(fun=test_fun, args=(i, c))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    for i, t in enumerate(threads):
        print("i = {}, sub_result = {}, c = {}".format(i, t.get_res(), c))


def test_multi_process():
    threads = []
    c = multiprocessing.Manager().list()  # c = [-1]
    for i in range(8):
        t = MyProcess(fun=test_fun, args=(i, c))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    for i, t in enumerate(threads):
        print("i = {}, sub_result = {}, c = {}".format(i, t.get_res(), c))
    print(type(c))
    c = list(c)
    print(type(c))


if __name__ == '__main__':
    # test_multi_thread()
    test_multi_process()
