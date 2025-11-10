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

# lavadlPath = os.path.join(rootPath, '../nc-libs/lava-dl/src')
sys.path.insert(0, lavadlPath)
import lava.lib.dl.slayer as slayer

from gait_snn_dataset import DVSGaitDataset, augment
from models_snn import PLIFSNN, LoihiCuBaSNN

from utils.assistant import Assistant
#from utils.estimate_ops import detect_activations_connections, activation_sparsity, number_neuron_updates, SynapticOperations
from utils.loss import calc_ce_loss

def ann_classifier(y):
    pred = torch.argmax(y, dim=1)
    return pred


if __name__ == '__main__':
    trained_folder = 'trained_models/Trained_plif_snn_2025-11-10T12.20.03.478367/'
    os.makedirs(trained_folder, exist_ok=True)

    device = torch.device('cpu')
    # device = torch.device('cuda')

    # model hyperparams
    inp_features = 2
    channels = 8
    feat_neur = 512
    classes = 20
    delay = False
    dropout = 0.2
    quantize = False 
    ce_loss = False

    # one GPU
    net = PLIFSNN(inp_features, channels, feat_neur, classes, delay, dropout, quantize, ce_loss, device).to(device)
    print(net)

    # parameters count
    total_params = sum(p.numel() for p in net.parameters())
    print(total_params)
    

    lr = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # steps = [10, 100, 200, 500]

    # data hyperparams
    sampling_time = 20000
    sample_length = 4000000
    ds_factor = 1 
    # data_directory = os.path.join(rootPath, '../data/dvs_gesture_bs2')
    data_directory = Config.events_file
    #data_directory = '/mnt/nas02nc/datasets/DVS_Gesture/dvs_gesture_bs2'

    training_set = DVSGaitDataset(path=data_directory, sampling_time=sampling_time, sample_length=sample_length,
                                     train=True, random_shift=False, ds_factor=ds_factor, transform="all")
    testing_set = DVSGaitDataset(path=data_directory, sampling_time=sampling_time, sample_length=sample_length,
                                    train=False, random_shift=False, ds_factor=ds_factor, transform=None)
    
    
    batch_size = 2

    train_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=testing_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # select loss function
    # supervised
    error = slayer.loss.SpikeRate(
        true_rate=0.5, false_rate=0.03, reduction='sum').to(device)

    # alternative for continuous output
    #error = calc_ce_loss
    
    # optional additional losses
    extr_loss = None  
    lam = None 

    classifier = slayer.classifier.Rate.predict
    # alternative for continuous output
    #classifier = mean_classifier

    stats = slayer.utils.LearningStats()
    assistant = Assistant(
        net, error, optimizer, stats,
        classifier=classifier, count_log=True, 
        lam=lam, extr_loss=extr_loss
    )

    show = False
    num = 5
    if show:
        t_loader = iter(train_loader)
        for i in range(num):
            data_batch, targets = next(t_loader)
            for v, ev in enumerate(data_batch):
                #ev = ev.permute(0, 2, 1, 3)
                inp_event = slayer.io.tensor_to_event(ev.cpu().data.numpy().reshape(2, 100, 100, -1))
                inp_event.t = inp_event.t * sampling_time
                # for gif
                inp_anim = inp_event.anim(frame_rate=50, repeat=True)
                from matplotlib import animation
                inp_anim.save(f'gifs/dvs_tr_{i}_class_{targets[0].item()}.gif', animation.PillowWriter(fps=50), dpi=300)
                # for 3d plot
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                evs_x = inp_event.x
                evs_y = inp_event.y
                evs_t = inp_event.t
                evs_p = inp_event.c
                ax.scatter(evs_t, evs_x, evs_y, c=evs_p, cmap='RdYlBu', s=1) # 'RdYlGn', 'plasma'
                ax.set_xlabel('Time (ms)')
                #ax.set_ylabel('X')
                #ax.set_zlabel('Y')
                plt.savefig(f'gifs/dvs_tr_{i}_class_{targets[0].item()}.png', bbox_inches='tight')
                #plt.show()
                print('plotting')
                
                
    network_path = "D:/Mirror/LMU/Semester 1/NeuroTUM/Munich Neuromorphic Hackathon/Simi/dev_tree/2025-nc-hackathon-event-id/EV-Gait-SNN/trained_models/Trained_plif_snn_2025-11-10T12.20.03.478367/network.pt"
    spike, lab = training_set[0]
    spike = spike.to(device)
    output = net(spike.unsqueeze(0))
    # net.load_state_dict(torch.load(trained_folder + '/network.pt', map_location=device))
    net.load_state_dict(torch.load(network_path, map_location=device))
    print(net)
    
    # evaluate
    # finetune_folder = trained_folder + '/finetuned_lavadl'
    # trained_folder = finetune_folder
    # os.makedirs(finetune_folder, exist_ok=True)
    # net.load_state_dict(torch.load(trained_folder + '/network.pt', map_location=device))
    net.load_state_dict(torch.load(network_path, map_location=device))
    for i, (inp, label) in enumerate(train_loader):  # testing loop
        output, count = assistant.valid(inp, label)
    for i, (inp, label) in enumerate(test_loader):  # testing loop
        output, count = assistant.test(inp, label)
    print(stats.validation.accuracy, stats.testing.accuracy)
    stats.validation.reset() #update()
    stats.testing.reset() #update()
    # export
    # net.export_hdf5(trained_folder + '/network.net', input_dims=[2, int(128//ds_factor), int(128//ds_factor)])
