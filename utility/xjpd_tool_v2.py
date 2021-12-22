#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# @date 2021/6/11
# @fileName thread_tool.py
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
from threading import Thread
import threading
import queue
from multiprocessing import Process, Event, Queue, Value, Manager
import functools
import time
import uuid
import warnings

__all__ = ['XJParallels', 'XJPPool']


class SubM(Process):
    def __init__(self, status,  worker, task_q, result_d):
        super().__init__()
        self.task_q = task_q
        self.result_d = result_d
        self.worker = worker
        self.status = status

    def run(self) -> None:
        while True:
            # 检查是否时被终止
            if self.status.value == 4:
                break
            task_info = None
            try:
                task_info = self.task_q.get(timeout=10)
                func_name = task_info['func_name']
                args, kwargs = task_info['params']
                f_event = task_info['f_event']
                task_id = task_info['task_id']
                worker_func = getattr(self.worker['worker'], func_name)
                result_data = worker_func(*args, **kwargs)
                self.result_d[task_id] = result_data
                f_event.set()

            except Exception as e:
                if task_info:
                    self.result_d[task_info['task_id']] = e
                    task_info['event'].set()
                else:
                    continue


class XJParallels(object):

    def __init__(self, worker, pd_num=1, sync=True):
        self.manager = Manager()
        self._xjp2021_worker = self.manager.dict({'worker':worker})
        self._xjp2021_sync_res = sync
        self._xjp2021_require_sync = False
        self._xjp2021_require_async = False

        self._xjp2021_status = Value('i', 1)  # 程序状态
        self._xjp2021_task_queen = self.manager.Queue(maxsize=-1)  # 任务队列

        self._xjp2021_result_dict = self.manager.dict()  # 任务结果存贮字典
        self._xjp2021_task_dict = self.manager.dict()  # 任务结果存贮字典

        self._xjp2021_subms = []
        for i in range(pd_num):
            sm = SubM(self._xjp2021_status,  self._xjp2021_worker, self._xjp2021_task_queen, self._xjp2021_result_dict)
            sm.daemon = True
            sm.start()
            self._xjp2021_subms.append(sm)

        if pd_num > 1:
            warnings.warn('Make sure that the methods you want to parallelize are thread safe')

    def __enter__(self):
        self._xjp2021_prev = self._xjp2021_sync_res
        if self._xjp2021_require_sync:
            self._xjp2021_sync_res = True

        if self._xjp2021_require_async:
            self._xjp2021_sync_res = False

    def __exit__(self, *args):
        self._xjp2021_sync_res = self._xjp2021_prev
        self._xjp2021_require_sync = False
        self._xjp2021_require_async = False
        return False

    def sync_result(self):
        self._xjp2021_require_sync = True
        return self

    def async_result(self):
        self._xjp2021_require_async = True
        return self

    def set_worker(self, worker):
        self._xjp2021_worker['worker'] = worker
        return self

    def _xjp2021_worker_func_decorator(self, attr_name):
        _worker_attr = getattr(self._xjp2021_worker['worker'], attr_name)
        if not callable(_worker_attr):
            return _worker_attr

        if self._xjp2021_sync_res:
            @functools.wraps(_worker_attr)
            def decorate_sync(*args, **kwargs):
                return _worker_attr(*args, **kwargs)
            return decorate_sync
        else:
            @functools.wraps(_worker_attr)
            def decorate_async(*args, **kwargs):
                f_evet = self.manager.Event()
                task_id = str(uuid.uuid4())
                task_info = {'func_name': attr_name, 'params': (args, kwargs), 'f_event': f_evet, 'task_id': task_id}
                self._xjp2021_task_dict[task_id] = task_info
                self._xjp2021_task_queen.put(task_info)
                return task_id, f_evet, self
            return decorate_async

    def get_result(self, task_id, timeout=None):
        if task_id not in self._xjp2021_task_dict:
            raise Exception('task_id not exist')
        task_info = self._xjp2021_task_dict.pop(task_id)
        f_event = task_info['f_event']
        f_event.wait(timeout)
        result_data = self._xjp2021_result_dict.pop(task_id, None)
        return result_data

    def XJPMagic(self, attr_name):
        return self._xjp2021_worker_func_decorator(attr_name)

    def __del__(self):
        self._xjp2021_status = 4

    # def __setattr__(self, key, value):
    #     pass
    #
    # def __getattribute__(self, item):
    #     public_item = ['XJPMagic', 'get_result', 'async_result']
    #     if item in public_item:
    #         return object.__getattribute__(self, item)

    def __getattr__(self, item):
        # print(item)
        if hasattr(self._xjp2021_worker['worker'], item):
            return self._xjp2021_worker_func_decorator(item)
        # raise AttributeError(f'{self._xjp2021_worker.__class__.__name__} has no attr{item}')


class XJPPool(object):
    def __init__(self, xjp_num, sync=False):
        self.ava_queen = queue.Queue()
        self.xjp_num = xjp_num
        self.sync = sync
        self.pop_xpg_eve = threading.Event()
        factor_t = Thread(target=self._xjp_factor, daemon=True)
        factor_t.start()

    def get_XJP(self, worker=None):
        if worker is None:
            xjp = self.ava_queen.get()
            self.pop_xpg_eve.set()
            return xjp

        if not isinstance(worker, list):
            worker = [worker]

        res = []
        for wk in worker:
            xjp: XJParallels = self.ava_queen.get()
            self.pop_xpg_eve.set()
            xjp.set_worker(wk)
            res.append(xjp)

        if len(worker) == 1:
            return res[0]
        return res

    def _xjp_factor(self):
        self.pop_xpg_eve.set()
        while True:
            self.pop_xpg_eve.wait()
            _l = self.xjp_num - self.ava_queen.qsize()
            for _ in range(_l):
                self.ava_queen.put(XJParallels(None, sync=self.sync))
            self.pop_xpg_eve.clear()


class TInstance(object):
    cls_attr = 'TInstance ATTR'

    def __init__(self):
        self.name = 'TInstance'
        self.status = 'd'

    def say_hello(self, ddd=12):
        time.sleep(1)
        print("TInstance: Hi~ ", ddd)
        return ddd

    def muu_add(self, a, b):
        return a + b

    def muu_sleep(self, name, t):
        time.sleep(t)
        print(name)
        return name


def test01():
    worker = TInstance()
    xjp_woker = XJParallels(worker, 1)
    print(xjp_woker._xjp2021_sync_res)
    print(xjp_woker.name)

    with xjp_woker.sync_result():
        xjp_woker.muu_sleep('muu_sleep sync 1', 1)
        xjp_woker.muu_sleep('muu_sleep sync 2', 2)

    with xjp_woker.async_result():
        tid1, _, _ = xjp_woker.muu_sleep('muu_sleep async 3', 3)
        tid2, _, _ = xjp_woker.muu_sleep('muu_sleep async 1', 1)

        # print()
        xjp_woker.get_result(tid2)

        task_id, f_event, _ = xjp_woker.say_hello('hahah')

    print('sync', xjp_woker.get_result(task_id, timeout=1))

    with xjp_woker.sync_result():
        print('muu_add', xjp_woker.muu_add(1, 4))
        print('say_hello', xjp_woker.say_hello(1233))

    print('ok')


class DDThread(Thread):
    def run(self) -> None:
        print('hello')


def test02():
    xx = DDThread()
    xx.start()
    time.sleep(1)
    xx.start()


def test03():
    xpj_pool = XJPPool(8)
    worker = TInstance()
    xjp = xpj_pool.get_XJP(worker)
    xjp2 = XJParallels(worker)
    xjp.say_hello('xjp')
    xjp2.say_hello('xjp2')

    with xjp.async_result():
        tid, _, _ = xjp.say_hello('你好')
        print(xjp.get_result(tid))


if __name__ == '__main__':
    test01()
    test03()

    print("ok")
