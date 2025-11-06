# Copyright (C) 2025 fortiss GmbH
# SPDX-License-Identifier:  BSD-3-Clause

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import logging
from typing import Dict, Tuple
import sys
import os
from pynput import keyboard
import time
import threading
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from config import Config
lavaPath = os.path.join(rootPath, '../nc-libs/lava/src')
sys.path.insert(0, lavaPath)

from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc import io
from lava.proc.io.encoder import Compression
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import RefPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort, RefPort
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU, Loihi2NeuroCore

from lava.proc.cyclic_buffer.process import CyclicBuffer
from lava.proc.lif.process import LIF

lavadlPath = os.path.join(rootPath, '../nc-libs/lava-dl/src')
sys.path.insert(0, lavadlPath)
from lava.lib.dl import netx

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models_snn import AllConvPLIFSNN


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

    # check if Loihi2 is available
    from lava.utils.system import Loihi2
    loihi2_is_available = Loihi2.is_loihi2_available
    if loihi2_is_available:
        from lava.proc import embedded_io as eio
        from lava.utils.profiler import Profiler
        print(f'Running on {Loihi2.partition}')
        from lava.proc.lif.ncmodels import NcL2ModelLif
    else:
        print("Loihi2 compiler is not available in this system. "
              "This tutorial will execute on CPU backend.")

    # data hyperparams
    sampling_time = 2
    sample_length = 1000
    ds_factor = 5
    events_path = "./events.npy"
    events, target = load_events(events_path, down_input=6.4, events=None, shrinking_factor=1.0, sequence_length=1,
                                 bins_per_seq=50, sampling_time=2, binary=False)
    predictions_path = "./predictions.npy"
    sample = torch.from_numpy(events).type('torch.FloatTensor')[0]
    empty_array = np.zeros(10, dtype=float)
    np.save(clp_predictions_path, empty_array)  # initially over-write the predictions file with an empty array
    
    # net hyperparams
    inp_features = 2
    channels = 8
    feat_neur = 512
    classes = 50
    delay = False
    dropout = 0.05
    quantize = True #False
    input_dim = [sample.shape[0], sample.shape[1], sample.shape[2]]
    print(input_dim)

    # instantiate network
    #trained_folder = 'trained_models/Pretrained_thu_eact_50_sj_scnnL_noDecInp_ds6_ce_plif_Q_25actions_myCnt_aug_2ms'
    trained_folder = 'trained_models/Train_plif_snn'
    device = torch.device('cpu')
    

    net_torch = AllConvPLIFSNN(inp_features, channels, feat_neur, classes, delay, dropout, quantize, device).to(device)
    output = net_torch(sample.unsqueeze(0).to(device))
    net_torch.load_state_dict(torch.load(trained_folder + '/network.pt', map_location=torch.device('cpu')))


    convert = True #False
    if convert:
        # convert spikingjelly
        # fuse bn
        net_torch.fuse_bn()
        # convert to lava-dl
        net_torch.to_lava_dl(quantize=True)
        # rates
        spikes, _, rates_lavadl = net_torch.get_features(sample.unsqueeze(0).to(device))
        rates_lavadl = rates_lavadl[0].detach().numpy()

    
    save = True #False #True
    if save:
        output, count = net_torch(sample.unsqueeze(0))
        net_torch.export_hdf5_fe(trained_folder + '/network.net', add_input_layer=False, input_dims=input_dim)

    # import trained network
    net = netx.hdf5.Network(net_config=trained_folder + '/network.net', input_shape=(input_dim[1], input_dim[2], input_dim[0]),
                            input_message_bits=16)
                            #reset_interval=steps_per_sample, reset_offset=2)
    print(net.inp.shape)
    print(net)

    # run parameters
    num_steps = 100000000

    # interactive
    data_injector = io.injector.Injector(shape=net.inp.shape, buffer_size=50, channel_config=None)
    input_data = np.moveaxis(events[0].copy(), 0, 2)
    input_data = np.moveaxis(input_data, 0, 1)
    input_data = np.round(input_data).astype(np.int32)
    receiver = io.extractor.Extractor(shape=net.out.shape, buffer_size=50, channel_config=None)

    # customize run config
    loihi2sim_exception_map = {
        io.sink.RingBuffer: io.sink.PyReceiveModelFloat,}
    loihi2hw_exception_map = {
        #DeltaEncoder: PyDeltaEncoderModelSparse,
        #MyEncoder: NxEncoderModel,
        LIF: NcL2ModelLif,
        io.sink.RingBuffer: io.sink.PyReceiveModelFloat,
        }
    #run_config = CustomRunConfig(select_tag='fixed_pt')
    if loihi2_is_available:
        # connect with embedded processor adapters
        in_adapter = eio.spike.PyToN3ConvAdapter(shape=net.inp.shape, num_message_bits=16, compression=Compression.DENSE)
        data_injector.out_port.connect(in_adapter.inp)
        in_adapter.out.connect(net.inp)
        # output
        out_adapter = eio.spike.NxToPyAdapter(shape=net.out.shape, num_message_bits=0)
        net.out.connect(out_adapter.inp)
        out_adapter.out.connect(receiver.in_port)
        run_config = Loihi2HwCfg(exception_proc_model_map=loihi2hw_exception_map)
    else:
        data_injector.out_port.connect(net.in_layer.inp)
        out_logger = io.sink.RingBuffer(shape=net.out.shape, buffer=num_steps)
        net.out.connect(receiver.in_port)
        run_config = Loihi2SimCfg(select_tag='fixed_pt', exception_proc_model_map=loihi2sim_exception_map)

        # net.out.connect(out_logger.a_in) #(net.out_layer.neuron.v)
        # dataloader.ground_truth.connect(out_proc.label_in)

    net._log_config.level = logging.INFO


    threshold = 0.3

    # run network
    run_condition = RunSteps(num_steps=num_steps, blocking=False)
    net.run(condition=run_condition, run_cfg=run_config)

    while(True):
        print("- - - - - - - - - - -")
        start = time.time()

        collect_features_list = []

        for event_iteration in range(10):
            time.sleep(0.03)
            events, _ = load_events(events_path, down_input=6.4, events=events, shrinking_factor=1.0, sequence_length=1,
                                    bins_per_seq=50, sampling_time=2, binary=False)
            input_data = np.moveaxis(events[0].copy(), 0, 2)
            input_data = np.moveaxis(input_data, 0, 1)
            input_data = np.round(input_data).astype(np.int32)
            for t in range(input_data.shape[-1]):
                data_injector.send(input_data[..., t])  # .reshape(net.inp.shape))
                #print(f"Send Complete num_steps={t}, Max Value={np.max(input_data[..., t])}")
                # Interpret the data as 24 bit signed value
                collect_features_list.append(torch.from_numpy(receiver.receive()).float().unsqueeze(0))
                # sleep 2ms
                #time.sleep(0.002)

        stacked_data = torch.stack(collect_features_list)
        #print(stacked_data.shape)
        features = stacked_data.mean(dim=0)
        #print(features.shape)

        stacked_data = torch.stack(collect_features_list)
        features = stacked_data.mean(dim=0)        # compute mean across all tensors in list


        y_pred = torch.argmax(pred, dim=0)
        print("Prediction: ", pred)
        if pred[y_pred.item()] > threshold:
            print("Predicted Class: ", y_pred)
            np.save(clp_predictions_path, pred.numpy())
        else:
            print("No known class detected")


    # stop execution
    net.wait()
    net.stop()



    




