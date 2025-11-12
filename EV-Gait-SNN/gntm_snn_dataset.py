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
import os
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
    event[:, 0] = event[:, 0] // factor
    event[:, 1] = event[:, 1] // factor
    return event

def del_outside(event, x=128, y=128):
    mask_x = np.logical_and(event[:,0]>=0, event[:,0]<x)
    mask_y = np.logical_and(event[:,1]>=0, event[:,1]<y)
    mask = np.logical_and(mask_x, mask_y)
    return event[mask]

def parse_input(path):
    hdf = h5py.File(path, 'r')
    events = hdf['events']
    # This list will hold all the event data
    all_events_list = []
    try:
        with h5py.File(path, 'r') as f:
            # 1. Access the 'events' group
            if 'events' not in f:
                print("Error: 'events' group not found in the file.")
            else:
                events_group = f['events']
                
                # 2. Get all dataset keys
                event_keys = list(events_group.keys())
                
                # 3. Sort the keys NUMERICALLY, not alphabetically
                # This is crucial so that '2' comes before '10'
                try:
                    sorted_keys = sorted(event_keys, key=int)
                except ValueError:
                    sorted_keys = event_keys # Fallback to default sort if they aren't numbers

                # 4. Loop through sorted keys and append data
                for key in sorted_keys:
                    # Read the data, which is a structured array, e.g., [(...)(...)]
                    data = events_group[key][()]
                    
                    # Check if the dataset is not empty
                    if data.size > 0:
                        # .tolist() converts the structured array into a
                        # simple Python list of lists, e.g., [[...],[...]]
                        # .extend() adds these lists to our master list
                        all_events_list.extend(data.tolist())

    except FileNotFoundError:
        print(f"Error: File not found at path: {path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    events_all = np.array(all_events_list)
    
    events_all[:, 0] = events_all[:, 0] - 321
    events_all[:, 1] = events_all[:, 1] - 21

    events_all = downsample_events(events_all, 5)
    events_all = del_outside(events_all, x=128, y=128)

    return events_all


class GNTMGaitDataset(Dataset):
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
            transform=None, random_shift=True, ds_factor=1, lava=False,
    ):
        super(GNTMGaitDataset, self).__init__()
        self.path = path
        print(path)

        def list_files_in_directory(directory):
            return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        def list_dirs_in_directory(directory):
            return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

        persons = list_dirs_in_directory(path)
        print(persons)
        self.data = []
        self.labels = []
        for person_idx, person in enumerate(persons):
            samples = list_files_in_directory(os.path.join(path, person))
            for sample_idx, sample in enumerate(samples):
                print(sample_idx, person)
                self.labels.append(person_idx)
                self.data.append(parse_input(os.path.join(path, person, sample)))
                
        np.random.seed(0)
        train_mask = np.random.rand(len(self.labels)) < 0.7
        if train:
            self.data = [self.data[i] for i in range(len(self.data)) if train_mask[i]]
            self.labels = [self.labels[i] for i in range(len(self.labels)) if train_mask[i]]
        else:
            self.data = [self.data[i] for i in range(len(self.data)) if not train_mask[i]]
            self.labels = [self.labels[i] for i in range(len(self.labels)) if not train_mask[i]]
        
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
        event = event[:, [0, 1, 3, 2]]  # x,y,t,p
        if self.transform == "all":
            # event = self.transform(event)
            event = shift(event)
            event = rotate(event)
            event = scale(event)
            event = del_outside(event)
        elif self.transform is not None:
            event = self.transform(event)
            event = del_outside(event)


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