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
from operator import itemgetter
import sys
sys.path.append("../")
from config import Config  


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
    x_shift = 20
    y_shift = 10
    xjitter = np.random.randint(2 * x_shift) - x_shift
    yjitter = np.random.randint(2 * y_shift) - y_shift
    event[:, 0] = event[:, 0] + xjitter
    event[:, 1] = event[:, 1] + yjitter
    return event


def rotate(event):
    # same as in: https://arxiv.org/pdf/2008.01151.pdf
    theta = 5
    ajitter = (np.random.rand() - 0.5) * theta / 180 * 3.141592654
    sin_theta = np.sin(ajitter)
    cos_theta = np.cos(ajitter)
    event[:, 0] = event[:, 0] * cos_theta - event[:, 1] * sin_theta
    event[:, 1] = event[:, 0] * sin_theta + event[:, 1] * cos_theta
    return event


def scale(event):
    # same as in: https://arxiv.org/pdf/2008.01151.pdf
    max_scale = 0.3
    e_scale = 1 + (np.random.rand() - 0.5) * max_scale
    event[:, 0] = event[:, 0] * e_scale
    event[:, 1] = event[:, 1] * e_scale
    return event


def downsample_events(event, factor):
    print(event)
    print(event[:, 1])
    event[:, 1] = event[:, 1] // factor
    event[:, 2] = event[:, 2] // factor
    return event

def del_outside(event, x=128, y=128):
    mask_x = np.logical_and(event[:,0]>=0, event[:,0]<x)
    mask_y = np.logical_and(event[:,1]>=0, event[:,1]<y)
    mask = np.logical_and(mask_x, mask_y)
    return event[mask]


class DVSGaitDataset(Dataset):
    """DVS Gait dataset class

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
    #data_format: str, optional
        data format of the input data, either 'bs2' or 'npy'. By default 'bs2'.
    ds_factor: int, optional
        factor to downsample event input. By default 1.
    """

    def __init__(
            self, path=dataPath,
            train=True, classes=20,
            sampling_time=1, sample_length=1450,
            transform=None, random_shift=True, ds_factor=1, lava=False, day=True
    ):
        super(DVSGaitDataset, self).__init__()
        self.path = path

        hdf = h5py.File(path, 'r')
        if train and day:
            train_keys = [key for key in hdf.keys() if "train" in key]
            train_person = [int(key.replace("train (1)_","").replace(".txt", "").split("_")[0]) for key in train_keys]
            train_num_sample = [int(key.replace("train (1)_","").replace(".txt", "").split("_")[1]) for key in train_keys]
            data_train = itemgetter(*train_keys)(hdf)
            self.labels = train_person
            self.data = data_train
        elif day:
            test_keys = [key for key in hdf.keys() if "test" in key]
            test_person = [int(key.replace("test (1)_","").replace(".txt", "").split("_")[0]) for key in test_keys]
            test_num_sample = [int(key.replace("test (1)_","").replace(".txt", "").split("_")[1]) for key in test_keys]
            data_test = itemgetter(*test_keys)(hdf)
            self.labels = test_person
            self.data = data_test
        elif not day:
            assert False
        self.sampling_time = sampling_time
        self.num_time_bins = int(sample_length / sampling_time)
        self.transform = transform
        self.random_shift = random_shift
        self.ds_factor = ds_factor
        self.lava = lava

    def __getitem__(self, i):
        label = self.labels[i]      
        
        #event = slayer.io.Event(x, y, p, t)
        event = np.array(self.data[i][:,:]) # x,y,t,p
        event = event[:, [1, 2, 0, 3]]  # x,y,t,p
        if self.transform == "all":
            # event = self.transform(event)
            event = shift(event)
            event = rotate(event)
            event = scale(event)
            event = del_outside(event)
        elif self.transform is not None:
            event = self.transform(event)
            event = del_outside(event)

        # downsample event input
        #event = downsample_events(event, self.ds_factor)
        #h_inp = int(128 / self.ds_factor)
        #w_inp = int(128 / self.ds_factor)


        reprs = np.zeros(
                (
                    2,
                    self.num_time_bins,
                    128 // self.ds_factor,
                    128 // self.ds_factor,
                )
            )
        #event[:, 2] = event[:, 2] / self.sampling_time
        t = event[:, 2]
        event[:, 2] = (t - t.min()) / (t.max() - t.min() + 1e-9) * (self.num_time_bins - 1)

        #event[:, 3] = (event[:, 3] + 1.0) / 2.0

        np.add.at(
            reprs,
            (
                np.floor(event[:, 3]).astype(np.int32),
                np.clip(
                    np.floor(event[:, 2]).astype(np.int32),
                    0,
                    self.num_time_bins - 1,
                ),
                np.floor(event[:, 1]).astype(np.int32),
                np.floor(event[:, 0]).astype(np.int32),
            ),
            1,
        )
        reprs = torch.tensor(reprs, dtype=torch.float)
        reprs = reprs.permute(0, 2, 3, 1)

        return reprs, label


    def __len__(self):
        return len(self.data)