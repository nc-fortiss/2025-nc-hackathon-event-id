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

lavadlPath = os.path.join(rootPath, '../nc-libs/lava-dl/src')
sys.path.insert(0, lavadlPath)
import lava.lib.dl.slayer as slayer

sjPath = os.path.join(rootPath, '../nc-libs/spikingjelly')
sys.path.insert(0, sjPath)
from spikingjelly.activation_based import surrogate, neuron, layer


from gait_snn_dataset import DVSGaitDataset, augment
from models_snn import PLIFSNN, LoihiCuBaSNN

from utils.assistant import Assistant
#from utils.estimate_ops import detect_activations_connections, activation_sparsity, number_neuron_updates, SynapticOperations
from utils.loss import calc_ce_loss
import time

def ann_classifier(y):
    pred = torch.argmax(y, dim=1)
    return pred

def load_events(path, down_input=5, events=None, shrinking_factor=1.0, resolution=(640,640), sequence_length=1, bins_per_seq=5, sampling_time=20, binary=True):
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


    if spikes is not None:
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
    # device = torch.device('cpu')
    device = torch.device('cuda')

    # model hyperparams
    inp_features = 2
    channels = 8
    feat_neur = 512
    classes = 2
    delay = False
    dropout = 0.2
    quantize = False 

    # one GPU
    
    net = PLIFSNN(inp_features, channels, feat_neur, classes, delay, dropout, quantize, device).to(device)
    net.blocks[-2] = layer.Linear(feat_neur, classes, bias=True).to(device)
    net.blocks[-1] = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(),detach_reset=True, v_reset=0.0, decay_input=True, init_tau=2.0).to(device)
    net.load_state_dict(torch.load("./trained_models/finetuned" + '/network.pt', map_location=device))
    net.eval()


    # parameters count
    total_params = sum(p.numel() for p in net.parameters())
    print(total_params)

    print('Network on device!')

    extr_loss = None  
    lam = None 

    classifier = slayer.classifier.Rate.predict
    # alternative for continuous output
    #classifier = mean_classifier

    stats = slayer.utils.LearningStats()
    
    events_path = "../../dataevents.npy"
    events, target = load_events(events_path, down_input=5, events=None, shrinking_factor=1.0, sequence_length=1, # richtige werte
                                 bins_per_seq=5, sampling_time=20, binary=False)
    clp_predictions_path = "./predictions.npy"
    empty_array = np.zeros(classes, dtype=float)
    np.save(clp_predictions_path, empty_array)       # initially over-write the predictions file with an empty array
    
    
    # live loop
    threshold = 0.4
    while True:
        print("- - - - - - - - - - -")

        start = time.time()

        outputs = []

        for event_iteration in range(40): # anpassbar
            events, _ = load_events(events_path, down_input=5, events=events, shrinking_factor=1.0, sequence_length=1,
                                        bins_per_seq=5, sampling_time=20, binary=False)
            data = torch.from_numpy(events).type('torch.FloatTensor')

            data = data.to(device)

            with torch.no_grad():
                output, rates = net(data)
                output = output.mean(dim=-1)
                outputs.append(output)

            time.sleep(0.09)

        stacked_data = torch.stack(outputs)
        pred = stacked_data.mean(dim=0).squeeze(0)        # compute mean across all tensors in list
        y_pred = torch.argmax(pred, dim=0)

        if pred[y_pred.item()] > threshold:
            np.save(clp_predictions_path, pred.numpy())
        else:
            if y_pred == 1:
                print("Simon detected!")
            else:
                print("Paul detected!")
            print(f"confidence: {pred}")     

        end = time.time()
        print(f"Iteration Time: {end-start}s")
