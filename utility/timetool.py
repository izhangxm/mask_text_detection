#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author tccw
# @date 2021/5/30
# @fileName timetool.py
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

import time


class _MyTimer(object):

    def __init__(self):
        self.now = self._get_now()

    def _get_now(self):
        return int(time.time() * 1000)

    def reset(self):
        self.now = self._get_now()

    def cost(self):
        _n = self._get_now()
        _c = _n - self.now
        self.now = _n
        return _c


def get_timer():
    _t = _MyTimer()
    return _t


if __name__ == '__main__':
    timer = get_timer()
    time.sleep(0.1)
    print(timer.cost())
    timer.reset()
    time.sleep(0.2)
    print(timer.cost())
    timer.reset()
    time.sleep(0.3)
    print(timer.cost())
