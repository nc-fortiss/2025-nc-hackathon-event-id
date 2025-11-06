# Copyright (C) 2025 fortiss GmbH
# SPDX-License-Identifier:  BSD-3-Clause

import os
import glob
import zipfile
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# import lava.lib.dl.slayer as slayer
import sys
curPath = os.path.abspath(__file__)
rootPath = os.path.split(curPath)[0]
dataPath = os.path.join(rootPath, '../../data/dvs_gesture_bs2')
lavadlPath = os.path.join(rootPath, '../../nc-libs/lava-dl/src')
sys.path.insert(0, lavadlPath)
import lava.lib.dl.slayer as slayer



def augment(event):
    # same as in: https://arxiv.org/pdf/2008.01151.pdf
    x_shift = 8
    y_shift = 8
    theta = 10
    xjitter = np.random.randint(2 * x_shift) - x_shift
    yjitter = np.random.randint(2 * y_shift) - y_shift
    ajitter = (np.random.rand() - 0.5) * theta / 180 * 3.141592654
    sin_theta = np.sin(ajitter)
    cos_theta = np.cos(ajitter)
    event.x = event.x * cos_theta - event.y * sin_theta + xjitter
    event.y = event.x * sin_theta + event.y * cos_theta + yjitter
    return event


def shift(event):
    # same as in: https://arxiv.org/pdf/2008.01151.pdf
    x_shift = 16
    y_shift = 16
    xjitter = np.random.randint(2 * x_shift) - x_shift
    yjitter = np.random.randint(2 * y_shift) - y_shift
    event.x = event.x + xjitter
    event.y = event.y + yjitter
    return event


def rotate(event):
    # same as in: https://arxiv.org/pdf/2008.01151.pdf
    theta = 20
    ajitter = (np.random.rand() - 0.5) * theta / 180 * 3.141592654
    sin_theta = np.sin(ajitter)
    cos_theta = np.cos(ajitter)
    event.x = event.x * cos_theta - event.y * sin_theta
    event.y = event.x * sin_theta + event.y * cos_theta
    return event


def scale(event):
    # same as in: https://arxiv.org/pdf/2008.01151.pdf
    max_scale = 0.3
    e_scale = 1 + (np.random.rand() - 0.5) * max_scale
    event.x = event.x * e_scale
    event.y = event.y * e_scale
    return event


def downsample_events(event, factor):
    event.x = event.x // factor
    event.y = event.y // factor
    return event


class DVSGestureDataset(Dataset):
    """DVS Gesture dataset class

    Parameters
    ----------
    path : str, optional
        path of dataset root, by default '/home/datasets/dvs_gesture_bs2'
    train : bool, optional
        train/test flag, by default True
    sampling_time : int, optional
        sampling time of event data, by default 1
    sample_length : int, optional
        length of sample data, by default 300
    transform : None or lambda or fx-ptr, optional
        transformation method. None means no transform. By default None.
    random_shift: bool, optional
        shift input sequence randomly in time. By default True.
    data_format: str, optional
        data format of the input data, either 'bs2' or 'npy'. By default 'bs2'.
    ds_factor: int, optional
        factor to downsample event input. By default 1.
    """

    def __init__(
            self, path=dataPath,
            train=True, classes=11,
            sampling_time=1, sample_length=1450,
            transform=None, random_shift=True, data_format='bs2', ds_factor=1, lava=False,
    ):
        super(DVSGestureDataset, self).__init__()
        self.path = path
        if train:
            dataParams = np.loadtxt(self.path + '/train_' + str(classes) + '.txt').astype('int')
        else:
            dataParams = np.loadtxt(self.path + '/test_' + str(classes) + '.txt').astype('int')

        self.samples = dataParams[:, 0]
        self.labels = dataParams[:, 1]
        self.sampling_time = sampling_time
        self.num_time_bins = int(sample_length / sampling_time)
        self.transform = transform
        self.random_shift = random_shift
        self.data_format = data_format
        self.ds_factor = ds_factor
        self.lava = lava

    def __getitem__(self, i):
        label = self.labels[i]  # // 2 # for train 6 classes
        # dataset in .bs2-format
        if self.data_format == 'bs2':
            filename = self.path + '/' + str(self.samples[i]) + '.bs2'
            event = slayer.io.read_2d_spikes(filename)
        # dataset in .npy-format
        elif self.data_format == 'npy':
            filename = self.path + '/' + str(self.samples[i]) + '.npy'
            event = slayer.io.read_np_spikes(filename, time_unit=1e-3)
        else:
            print('No correct data format!!! -> Only bs2 and npy valid')

        if self.transform is not None:
            # event = self.transform(event)
            event = shift(event)
            event = rotate(event)
            event = scale(event)

        # downsample event input
        event = downsample_events(event, self.ds_factor)
        h_inp = int(128 / self.ds_factor)
        w_inp = int(128 / self.ds_factor)

        if self.random_shift:
            spike = event.fill_tensor(np.zeros((2, h_inp, w_inp, self.num_time_bins)),
                                      sampling_time=self.sampling_time,
                                      random_shift=True)
        else:
            spike = event.fill_tensor(np.zeros((2, h_inp, w_inp, self.num_time_bins)),
                                      sampling_time=self.sampling_time)
        spike = self.sampling_time * spike

        if self.lava:
            # convert to WHC format
            spike = np.moveaxis(spike, 0, 2)
            spike = np.moveaxis(spike, 0, 1)
            spike = spike.astype(np.int32)
            return spike, label.astype(np.int32)
        else:
            spike = torch.tensor(spike, dtype=torch.float)

        return spike, label

    def __len__(self):
        return len(self.samples)







