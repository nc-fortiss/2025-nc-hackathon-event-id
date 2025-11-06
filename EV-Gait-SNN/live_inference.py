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

import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from config import Config

from dvs_gesture_datset import DVSGestureDataset, augment
from models_snn import PLIFSNN, LoihiCuBaSNN

from utils.assistant import Assistant
from utils.estimate_ops import detect_activations_connections, activation_sparsity, number_neuron_updates, SynapticOperations
from utils.loss import calc_ce_loss


def load_events(path, down_input=5, events=None, shrinking_factor=1.0, resolution=(640,640), sequence_length=1, bins_per_seq=50, sampling_time=2, binary=True):
    try:
        spikes = np.load(path)
    except ValueError as e:
        # print("wrong reshape : ", e)
        spikes = None
    except EOFError as e:
        print("\n!!!                        EOF exception triggered !!!\n")
        spikes = None
    input_resolution = int(resolution[0] / down_input), int(resolution[1] / down_input)
    erate = 0

    if events is None:
        events = np.zeros((sequence_length, 2, input_resolution[1], input_resolution[0], bins_per_seq))
    else:
        events.fill(0)

    ds_time = bins_per_seq * sampling_time

    if spikes is not None and len(spikes['t']):

        # Optimization in the TONUS set up
        shift_x, shift_y = (1 - shrinking_factor) / 2, 1 - shrinking_factor
        ev_seq = np.clip((spikes['t'] - spikes['t'][0]) // ds_time, 0, sequence_length - 1)
        events_x = np.round((640 - spikes['x'] * shrinking_factor) / down_input - input_resolution[0] * shift_x).astype(
            np.uint8)
        events_y = np.round(spikes['y'] * shrinking_factor / down_input + input_resolution[1] * shift_y).astype(np.uint8)
        cropped_indexes = (events_x >= 0) & (events_x < input_resolution[0]) & (events_y >= 0) & (
                events_y < input_resolution[1])
        events_t = np.clip(((spikes['t'] - spikes['t'][0]) / ds_time) // sampling_time, 0, bins_per_seq - 1).astype(
            np.uint32)
        events_p = spikes['p']
        events_t, events_p, events_x, events_y, ev_seq = events_t[cropped_indexes], events_p[cropped_indexes], \
            events_x[cropped_indexes], events_y[cropped_indexes], ev_seq[cropped_indexes]
        # latest_events = np.zeros((2, input_resolution[1], input_resolution[0], bins_per_seq))
        if binary:
            events[ev_seq, events_p, events_y, events_x, events_t] = 1
        else:
            np.add.at(events, (ev_seq, events_p, events_y, events_x, events_t), 1)

        # np.add.at(events, (ev_seq,events_p, events_y, events_x, events_t), 1)
        erate = len(spikes['x'])
    else: 
        print("\n\n                     Loading None spikes !!! \n\n")


    return events, erate


if __name__ == '__main__':
    # live inference event-based gait recognition