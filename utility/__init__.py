#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author tccw
# @date 2021/3/28
# @fileName __init__.py
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
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

class XJExceptionContext(object):
    def __init__(self, block_name, suppress_expt=False, prt_cost=False):
        self.block_name = block_name
        self.suppress_expt = suppress_expt
        self.prt_cost = prt_cost

    def __enter__(self):
        self.start = int(time.time() * 1000)
        return self.suppress_expt

    def __exit__(self, exc_type, exc_val, exc_tb):
        _now = int(time.time() * 1000)
        if exc_type is not None:
            _msg = f"{self.block_name} Error: {exc_val.args[0]}"
            exc_val.args = tuple([_msg] + list(exc_val.args[1:]))
        if self.prt_cost:
            print(f"{self.block_name} cost:{_now - self.start}ms")
        return self.suppress_expt is True
