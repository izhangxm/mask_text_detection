#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author tccw
# @date 2021/11/5
# @fileName KMeanCuda.py
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

import numpy as np
import torch
from tqdm import tqdm


class KMeansTorch:
    def __init__(self, n_clusters=3, max_iter=None, verbose=True, device=None):

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        self.device = device

        self.n_cluster = n_clusters
        self.n_clusters = n_clusters
        self.labels_ = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(self.device)
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.cluster_centers_ = None

    def fit(self, x):
        # 随机选择初始中心点，想更快的收敛速度可以借鉴sklearn中的kmeans++初始化方法
        x = torch.from_numpy(np.array(x)).to(self.device)

        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        init_points = x[init_row]
        self.centers = init_points
        while True:
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            self.count += 1

        self._representative_sample()

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)

        for i, sample in enumerate(x):
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            labels[i] = torch.argmin(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        self.labels_ = labels
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            mask = self.labels_ == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers

    def _representative_sample(self):
        # 查找距离中心点最近的样本，作为聚类的代表样本，更加直观
        self.representative_samples = torch.argmin(self.dists, (0))
        self.cluster_centers_ = torch.argmin(self.dists, (0))


class KMeansNumpy:
    def __init__(self, n_clusters=3, max_iter=None, verbose=True, device=None):

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        self.device = device

        self.n_cluster = n_clusters
        self.n_clusters = n_clusters
        self.labels_ = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(self.device)
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.cluster_centers_ = None

    def fit(self, x):
        # 随机选择初始中心点，想更快的收敛速度可以借鉴sklearn中的kmeans++初始化方法
        x = torch.from_numpy(np.array(x)).to(self.device)

        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        init_points = x[init_row]
        self.centers = init_points
        while True:
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            self.count += 1

        self._representative_sample()

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)

        for i, sample in enumerate(x):
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            labels[i] = torch.argmin(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        self.labels_ = labels
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            mask = self.labels_ == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers

    def _representative_sample(self):
        # 查找距离中心点最近的样本，作为聚类的代表样本，更加直观
        self.representative_samples = torch.argmin(self.dists, (0))
        self.cluster_centers_ = torch.argmin(self.dists, (0))


def time_clock(matrix, device):
    a = time.time()
    k = KMEANS(n_clusters=3, max_iter=10, verbose=False, device=device)
    k.fit(matrix)
    b = time.time()
    return (b - a) / k.count


def choose_device(cuda=False):
    if cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    device = choose_device(False)
    cpu_speeds = []
    for i in tqdm([1, 1, 1, 1, 8000, 20000]):
        matrix = torch.rand((6, i)).to(device)
        speed = time_clock(matrix, device)
        cpu_speeds.append(speed)
    l1, = plt.plot([20, 100, 500, 2000, 8000, 20000], cpu_speeds, color='r', label='CPU')

    device = choose_device(True)

    gpu_speeds = []
    for i in tqdm([20, 100, 500, 2000, 8000, 20000]):
        matrix = torch.rand((10000, i)).to(device)
        speed = time_clock(matrix, device)
        gpu_speeds.append(speed)
    l2, = plt.plot([20, 100, 500, 2000, 8000, 20000], gpu_speeds, color='g', label="GPU")

    plt.xlabel("num_features")
    plt.ylabel("speed(s/iter)")
    plt.title("Speed with cuda")
    plt.legend(handles=[l1, l2], labels=['CPU', 'GPU'], loc='best')

    plt.show("dd")
