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
import argparse
import datetime

import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from config import Config

lavadlPath = os.path.join(rootPath, '../nc-libs/lava-dl/src')
sys.path.insert(0, lavadlPath)
import lava.lib.dl.slayer as slayer

from gait_snn_dataset import DVSGaitDataset, augment
from models_snn import PLIFSNN, LoihiCuBaSNN

from utils.assistant import Assistant
#from utils.estimate_ops import detect_activations_connections, activation_sparsity, number_neuron_updates, SynapticOperations
from utils.loss import calc_ce_loss, mean_classifier

def ann_classifier(y):
    pred = torch.argmax(y, dim=1)
    return pred


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Train Gait SNN')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use for training')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training and testing')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--sampling_time', type=int, default=20_000, help='Sampling time for DVS data')
    parser.add_argument('--sample_length', type=int, default=4_000_000, help='Sample length for DVS data')
    parser.add_argument('--ce_loss', type=bool, help='Use cross-entropy loss')
    args = parser.parse_args()

    timestamp = datetime.datetime.now().isoformat()
    trained_folder = f'trained_models/Trained_plif_snn_{timestamp}'.replace(':', '.')
    os.makedirs(trained_folder, exist_ok=True)

    device = torch.device(args.device)

    # Print all arguments
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # model hyperparams
    inp_features = 2
    channels = 8
    feat_neur = 512
    classes = 20
    delay = False
    dropout = 0.2
    quantize = False 
    # one GPU
    net = PLIFSNN(inp_features, channels, feat_neur, classes, delay, dropout, quantize, args.ce_loss, device).to(device)
    print(net)

    # parameters count
    total_params = sum(p.numel() for p in net.parameters())
    print(total_params)

    lr = args.learning_rate
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # data hyperparams
    sampling_time = args.sampling_time
    sample_length = args.sample_length
    ds_factor = 1 
    data_directory = os.path.join(Config.events_file)

    training_set = DVSGaitDataset(path=data_directory, sampling_time=sampling_time, sample_length=sample_length,
                                     train=True, random_shift=False, ds_factor=ds_factor, transform="all")
    testing_set = DVSGaitDataset(path=data_directory, sampling_time=sampling_time, sample_length=sample_length,
                                    train=False, random_shift=False, ds_factor=ds_factor, transform=None)
    
    batch_size = args.batch_size

    train_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=testing_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # select loss function
    if args.ce_loss:
        error = calc_ce_loss
        classifier = mean_classifier
    else:
        error = slayer.loss.SpikeRate(true_rate=0.5, false_rate=0.03, reduction='sum').to(device)
        classifier = slayer.classifier.Rate.predict
    extr_loss = None  
    lam = None 


    stats = slayer.utils.LearningStats()
    assistant = Assistant(
        net, error, optimizer, stats,
        classifier=classifier, count_log=True, 
        lam=lam, extr_loss=extr_loss
    )
    # torch.autograd.set_detect_anomaly(True)
    show = False
    num = 5
    if show:
        t_loader = iter(train_loader)
        for i in range(num):
            data_batch, targets = next(t_loader)
            for v, ev in enumerate(data_batch):
                #ev = ev.permute(0, 2, 1, 3)
                inp_event = slayer.io.tensor_to_event(ev.cpu().data.numpy().reshape(2, 128, 128, -1))
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

    train = True 
    epochs = args.epochs
    if train:
        for epoch in range(epochs):
            # if epoch in steps:
            # assistant.reduce_lr(factor=10 / 3)
            for i, (inp, label) in enumerate(train_loader):  # training loop
                #print(label)
                #output, count = assistant.train(inp, label)
                output = assistant.train(inp, label)
                #print(output[0].shape)
                stats.print(epoch, iter=i, dataloader=train_loader)

            for i, (inp, label) in enumerate(test_loader):  # testing loop
                #output, count = assistant.test_simclr(inp, label)
                output = assistant.test(inp, label)
                # stats.print(epoch, iter=i, dataloader=test_loader)

            stats.update()
            stats.save(trained_folder + '/')
            stats.plot(path=trained_folder + '/')
            print("Epoch " + str(epoch) + " finished")
            net.grad_flow(trained_folder + '/')

            if stats.testing.best_accuracy and classifier:
                torch.save(net.state_dict(), trained_folder + '/network.pt')
            elif stats.testing.best_loss:
                torch.save(net.state_dict(), trained_folder + '/network.pt')

            if stats.training.best_accuracy and classifier:
                torch.save(net.state_dict(), trained_folder + '/network_tr.pt')
            elif stats.training.best_loss:
                torch.save(net.state_dict(), trained_folder + '/network_tr.pt')
    else:
        spike, lab = training_set[0]
        spike = spike.to(device)
        output = net(spike.unsqueeze(0))

    net.load_state_dict(torch.load(trained_folder + '/network.pt', map_location=device))
    print(net)
    

    finetune = False 
    if finetune:
        finetune_folder = trained_folder + '/finetuned_lavadl'
        trained_folder = finetune_folder
        os.makedirs(finetune_folder, exist_ok=True)
        # fuse bn
        net.fuse_bn()
        # convert to lava-dl
        net.to_lava_dl(quantize=True)
        net.device = device
        #print(net)
        # new assistant
        stats = slayer.utils.LearningStats()
        assistant = slayer.utils.Assistant(
            net, error, optimizer, stats,
            classifier=classifier, count_log=True
        )
        assistant.device = device
        # save params
        torch.save(net.state_dict(), finetune_folder + '/network.pt')

        epochs = 0
        for epoch in range(epochs):
            # if epoch in steps:
            # assistant.reduce_lr(factor=10 / 3)
            for i, (inp, label) in enumerate(train_loader):  # training loop
                output, count = assistant.train(inp, label)
                # stats.print(epoch, iter=i, dataloader=train_loader)
                #print(i)

            for i, (inp, label) in enumerate(test_loader):  # testing loop
                output, count = assistant.test(inp, label)
                # stats.print(epoch, iter=i, dataloader=test_loader)

            stats.update()
            stats.save(finetune_folder + '/')
            stats.plot(path=finetune_folder + '/')
            print("Epoch " + str(epoch) + " finished")
            net.grad_flow(finetune_folder + '/')

            if stats.testing.best_accuracy and classifier:
                torch.save(net.state_dict(), finetune_folder + '/network.pt')
                # net.export_hdf5(trained_folder + '/network.net', add_input_layer=True, input_dims=[2, 128, 128])
            elif stats.testing.best_loss:
                torch.save(net.state_dict(), finetune_folder + '/network.pt')

            if stats.training.best_accuracy and classifier:
                torch.save(net.state_dict(), finetune_folder + '/network_tr.pt')
            elif stats.training.best_loss:
                torch.save(net.state_dict(), finetune_folder + '/network_tr.pt')

        # evaluate
        net.load_state_dict(torch.load(trained_folder + '/network.pt', map_location=device))
        for i, (inp, label) in enumerate(train_loader):  # testing loop
            output, count = assistant.valid(inp, label)
        for i, (inp, label) in enumerate(test_loader):  # testing loop
            output, count = assistant.test(inp, label)
        print(stats.validation.accuracy, stats.testing.accuracy)
        stats.validation.reset() #update()
        stats.testing.reset() #update()

        # export
        net.export_hdf5(finetune_folder + '/network.net', input_dims=[2, int(128//ds_factor), int(128//ds_factor)])

    
