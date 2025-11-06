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

from lava.proc.lif.process import LIF

lavadlPath = os.path.join(rootPath, '../nc-libs/lava-dl/src')
sys.path.insert(0, lavadlPath)
from lava.lib.dl import netx

from dvs_gesture_dataset import DVSGestureDataset, augment
from models_snn import PLIFSNN, LoihiCuBaSNN


if __name__ == '__main__':

    # check if Loihi2 is available
    from lava.utils.system import Loihi2
    loihi2_is_available = Loihi2.is_loihi2_available
    if loihi2_is_available:
        from lava.proc import embedded_io as eio
        from lava.utils.profiler import Profiler
        print(f'Running on {Loihi2.partition}')
        from lava.proc.lif.ncmodels import NcL2ModelLif
        from lava.proc.cyclic_buffer.process import CyclicBuffer
    else:
        print("Loihi2 compiler is not available in this system. "
              "This tutorial will execute on CPU backend.")

    # data hyperparams
    sampling_time = 2
    sample_length = 1000
    ds_factor = 1
    data_directory = os.path.join(rootPath, '../data/dvs_gesture_bs2')
    #data_directory = '/mnt/nas02nc/datasets/DVS_Gesture/dvs_gesture_bs2'
    training_set = DVSGestureDataset(path=data_directory, sampling_time=sampling_time, sample_length=sample_length,
                                     train=True, random_shift=False, ds_factor=ds_factor)
    testing_set = DVSGestureDataset(path=data_directory, sampling_time=sampling_time, sample_length=sample_length,
                                    train=False, random_shift=False, ds_factor=ds_factor)
    sample, gt = testing_set[0]
    print(sample.shape)
    
    # net hyperparams
    inp_features = 2
    channels = 8
    feat_neur = 512
    classes = 11
    delay = False
    dropout = 0.05
    quantize = True #False
    input_dim = [sample.shape[0], sample.shape[1], sample.shape[2]]
    print(input_dim)

    # instantiate network
    trained_folder = 'trained_models/Trained_cuba_snn'
    os.makedirs(trained_folder, exist_ok=True)
    device = torch.device('cpu')
    

    net_torch = PLIFSNN(inp_features, channels, feat_neur, classes, delay, dropout, quantize, device).to(device)
    output = net_torch(sample.unsqueeze(0).to(device))
    #net_torch.load_state_dict(torch.load(trained_folder + '/network.pt', map_location=torch.device('cpu')))


    convert = True
    if convert:
        # convert spikingjelly
        # fuse bn
        net_torch.fuse_bn()
        # convert to lava-dl
        net_torch.to_lava_dl(quantize=True)
        # rates
        out = net_torch(sample.unsqueeze(0).to(device))
        out_lavadl = out[0].detach().numpy()

    
    save = True #False 
    if save:
        output, count = net_torch(sample.unsqueeze(0))
        net_torch.export_hdf5(trained_folder + '/network.net', add_input_layer=False, input_dims=input_dim)

    # import trained network
    net = netx.hdf5.Network(net_config=trained_folder + '/network.net', input_shape=(input_dim[1], input_dim[2], input_dim[0]),
                            input_message_bits=0)
                            #reset_interval=steps_per_sample, reset_offset=1)
    print(net.inp.shape)
    print(net)

    # run parameters
    num_samples = 1  # len(testing_set)
    steps_per_sample = int(sample_length / sampling_time)
    num_steps = num_samples * steps_per_sample + 1

    training_set.lava = True
    testing_set.lava = True
    dataloader = io.dataloader.SpikeDataloader(
        dataset=training_set,
        interval=steps_per_sample)
    sample, gt = testing_set[0]

    if loihi2_is_available:
        # connect with embedded processor adapters
        in_adapter = eio.spike.PyToN3ConvAdapter(shape=net.inp.shape, num_message_bits=0, compression=Compression.DENSE)
        dataloader.s_out.connect(in_adapter.inp)
        in_adapter.out.connect(net.inp)
        # cyclic buffer
        #input_buffer = CyclicBuffer(first_frame=sample[..., 0], replay_frames=sample[...,
        #                                                                      301:311])  # num_ca_count = 1 + (1 + self.num_buffer - 3 + 7) // 8 => 11, 19, 27, 35, .. 51, ..., 91
        #input_buffer.s_out.connect(net.inp)
        # output
        out_adapter = eio.spike.NxToPyAdapter(shape=net.out.shape, num_message_bits=0)
        net.out.connect(out_adapter.inp)
        out_logger = io.sink.RingBuffer(shape=net.out.shape, buffer=num_steps)
        out_adapter.out.connect(out_logger.a_in)
        # customize run config
        loihi2hw_exception_map = {
            LIF: NcL2ModelLif,
            io.sink.RingBuffer: io.sink.PyReceiveModelFloat,
        }
        run_config = Loihi2HwCfg(exception_proc_model_map=loihi2hw_exception_map)
        # profiling
        #profiler = Profiler.init(run_config)
        #profiler.energy_probe(num_steps=num_steps)
        # profiler.activity_probe()
    else:
        dataloader.s_out.connect(net.inp)
        out_logger = io.sink.RingBuffer(shape=net.out.shape, buffer=num_steps)
        net.out.connect(out_logger.a_in)
        # customize run config
        loihi2sim_exception_map = {
            io.sink.RingBuffer: io.sink.PyReceiveModelFloat,}
        run_config = Loihi2SimCfg(select_tag='fixed_pt', exception_proc_model_map=loihi2sim_exception_map)

    net._log_config.level = logging.INFO

    # run network
    run_condition = RunSteps(num_steps=num_steps)
    net.run(condition=run_condition, run_cfg=run_config)

    # get spikes
    loihi_spikes = out_logger.data.get()
    print(loihi_spikes.shape)

    # stop execution
    net.stop()

    if loihi2_is_available:
        # profiling
        print("----------------------")
        # print(f"Execution times: {profiler.execution_time}")
        # print(f"Total execution time: {np.round(np.sum(profiler.execution_time), 6)} s")
        # total_time = np.sum(profiler.execution_time)
        # per_scan_time = total_time / num_samples
        # print(f"Per scan execution time: {np.round(per_scan_time, 6)} s")
        # print("----------------------")
        # print(f"Total power: {np.round(profiler.power, 6)} W")
        # print(f"Static power: {np.round(profiler.static_power, 6)} W")
        # print(f"Dynamic power: {np.round(profiler.dynamic_power, 6)} W")
        # print(f"Total energy: {np.round(profiler.energy, 6)} J")
        # print(f"Static energy: {np.round(profiler.static_energy, 6)} J")
        # dynamic_energy = profiler.energy - profiler.static_energy
        # print(f"Dynamic energy: {np.round(profiler.dynamic_energy, 6)} J")
        # per_scan_total_energy = profiler.energy / num_samples
        # print(f"Per scan total energy: {np.round(per_scan_total_energy, 6)} J")
        # per_scan_static_energy = profiler.static_energy / num_samples
        # print(f"Per scan static energy: {np.round(per_scan_static_energy, 6)} J")
        # per_scan_dynamic_energy = dynamic_energy / num_samples
        # print(f"Per scan dynamic energy: {np.round(per_scan_dynamic_energy, 6)} J")
        #
        # edp = per_scan_total_energy * per_scan_time
        # print(f"Energy Delay Product: {np.round(edp, 6)} Js")



        




