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

from gait_snn_dataset import DVSGaitDataset, augment
from models_snn import PLIFSNN, LoihiCuBaSNN

from utils.assistant import Assistant
from utils.estimate_ops import detect_activations_connections, activation_sparsity, number_neuron_updates, SynapticOperations
from utils.loss import calc_ce_loss


data = np.load('/Users/simon/Downloads/dataevents.npy')
print(data.shape)
print(data[0])
print(data[0][0]) 

exit()

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
    trained_folder = 'trained_models/Trained_plif_snn'
    os.makedirs(trained_folder, exist_ok=True)

    # device = torch.device('cpu')
    device = torch.device('cuda')

    # model hyperparams
    inp_features = 2
    channels = 8
    feat_neur = 512
    classes = 50
    delay = False
    dropout = 0.1
    quantize = False 

    # one GPU
    net = PLIFSNN(inp_features, channels, feat_neur, classes, delay, dropout, quantize, device).to(device)
    print(net)

    # parameters count
    total_params = sum(p.numel() for p in net.parameters())
    print(total_params)

    # two GPUs in parallel
    # print(torch.cuda.device_count())
    # net = AllConvPLIFSNN(inp_features, channels, feat_neur, classes, delay, dropout, quantize, device).to(device)
    # net.forward(torch.rand(1, 2, 100, 100, 1).to(device))
    # net = nn.DataParallel(net, device_ids=[0, 1])
    # net.to(device)
    print('Network on device!')

    # optimizer = Nadam(net.parameters(), lr=0.003, amsgrad=True)
    lr = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # steps = [10, 100, 200, 500]

    # data hyperparams
    sampling_time = 2
    sample_length = 1000
    ds_factor = 1 
    # data_directory = os.path.join(rootPath, '../data/dvsgesture)

    training_set = DVSGestureDataset(path=data_directory,
                                     train=True, transform=augment, ds_factor=1)
    testing_set = DVSGestureDataset(path=data_directory,
                                    train=False, random_shift=False, ds_factor=1)
    
    
    batch_size = 8

    train_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=testing_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # select loss function
    # supervised
    error = slayer.loss.SpikeRate(
        true_rate=0.5, false_rate=0.03, reduction='sum').to(device)

    # alternative for continuous output
    #error = calc_ce_loss
    

    extr_loss = None  
    lam = None 

    classifier = slayer.classifier.Rate.predict
    # alternative for continuous output
    classifier = mean_classifier

    stats = slayer.utils.LearningStats()
    assistant = Assistant(
        net, error, optimizer, stats,
        classifier=classifier, count_log=True, 
        lam=lam, extr_loss=extr_loss
    )
    
    events_path = "./events.npy"
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

        collect_features_list = []

        for event_iteration in range(10): # anpassbar
            events, _ = load_events(events_path, down_input=5, events=events, shrinking_factor=1.0, sequence_length=1,
                                        bins_per_seq=50, sampling_time=2, binary=True)
            data = torch.from_numpy(events).type('torch.FloatTensor')
            data = data.to(device)#.unsqueeze(0)

            with torch.no_grad():
                spks, cnts, features = net(data)
                collect_features_list.append(features)

            # TODO: adjust parameter
            time.sleep(0.08)


        stacked_data = torch.stack(collect_features_list)
        features = stacked_data.mean(dim=0)        # compute mean across all tensors in list


        y_pred = torch.argmax(pred, dim=0)
        print("Prediction: ", pred)
        if pred[y_pred.item()] > threshold:
            print("Predicted Class: ", y_pred)
            np.save(clp_predictions_path, pred.numpy())
        else:
            print("No known class detected")
        
        if show: 
            inp_event = slayer.io.tensor_to_event(data[0].cpu().data.numpy().reshape(2, 128, 128, -1))
            inp_event.t = inp_event.t * sampling_time
            #print("GT: ", label)
            fig = plt.figure()
            timer = fig.canvas.new_timer(
                interval=1500)  # creating a timer object and setting an interval of 2000 milliseconds
            timer.add_callback(close_event)
            timer.start()
            inp_anim = inp_event.show(fig=fig, frame_rate=20, repeat=False)
            #inp_anim.show(animation.PillowWriter(fps=20), dpi=300)
            #plt.show()
       

        end = time.time()
        print(f"Iteration Time: {end-start}s")
