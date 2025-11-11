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

# import lava.lib.dl.slayer as slayer
import sys
curPath = os.path.abspath(__file__)
rootPath = os.path.split(curPath)[0]
lavadlPath = os.path.join(rootPath, '../../nc-libs/lava-dl/src')
sys.path.insert(0, lavadlPath)
import lava.lib.dl.slayer as slayer
from lava.lib.dl.slayer.utils import quantize as slayer_quantize

sjPath = os.path.join(rootPath, '../../nc-libs/spikingjelly')
sys.path.insert(0, sjPath)
from spikingjelly.activation_based import surrogate, neuron, functional, layer
from spikingjelly.activation_based.lava_exchange import to_lava_blocks, conv2d_to_lava_synapse_conv, to_lava_neuron, quantize_8bit, linear_to_lava_synapse_dense, to_lava_block_conv, to_lava_block_dense
from spikingjelly.activation_based.lava_exchange import BatchNorm2d as LoihiBatchNorm2d

def quantize_8bit(weight, descale=False):
    w_scale = 1 << 6
    if descale is False:
        return slayer_quantize(
            weight, step=2 / w_scale
        ).clamp(-256 / w_scale, 255 / w_scale)
    else:
        return slayer_quantize(
            weight, step=2 / w_scale
        ).clamp(-256 / w_scale, 255 / w_scale) * w_scale
        
def fuse_bn(module):
    module_output = module
    if isinstance(module, (nn.Sequential,)):
        print("[nn.Sequential]\tfusing BN and dropout")
        idx = 0
        for idx in range(len(module) - 1):
            if not isinstance(module[idx], nn.Conv2d) or not isinstance(
                module[idx + 1], nn.BatchNorm2d
            ):
                continue
            conv = module[idx]
            bn = module[idx + 1]
            channels = bn.weight.shape[0]
            invstd = 1 / torch.sqrt(bn.running_var + bn.eps)
            conv.weight.data = (
                conv.weight
                * bn.weight[:, None, None, None]
                * invstd[:, None, None, None]
            )
            if conv.bias is None:
                conv.bias = nn.Parameter(torch.zeros(conv.out_channels).to(conv.weight.device))
            conv.bias.data = (
                conv.bias - bn.running_mean
            ) * bn.weight * invstd + bn.bias
            module[idx + 1] = nn.Identity()
        for name, child in module.named_children():
            module_output.add_module(name, fuse_bn(child))
        del module

    elif isinstance(module, (nn.ModuleList,)):
        print("[nn.ModuleList]\tfusing BN and dropout")
        idx = 0
        for idx in range(len(module) - 1):
            if not isinstance(module[idx], nn.Conv2d) or not isinstance(
                module[idx + 1], nn.BatchNorm2d
            ):
                continue
            conv = module[idx]
            bn = module[idx + 1]
            channels = bn.weight.shape[0]
            invstd = 1 / torch.sqrt(bn.running_var + bn.eps)
            conv.weight.data = (
                conv.weight
                * bn.weight[:, None, None, None]
                * invstd[:, None, None, None]
            )
            if conv.bias is None:
                conv.bias = nn.Parameter(torch.zeros(conv.out_channels).to(conv.weight.device))
            conv.bias.data = (
                conv.bias - bn.running_mean
            ) * bn.weight * invstd + bn.bias
            module[idx + 1] = nn.Identity()
        module_output = module
        del module

    return module_output

class TemporalMean(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        print(x.shape)
        x = x.mean(dim=0)
        print(x.shape)
        return x
        return x.mean(dim=-1)
    
    
class PLIFSNN(torch.nn.Module):
    def __init__(self, inp_features=2, channels=8, feat_neur=512, classes=12, delay=False, dropout=0.05, quantize=False, ce_loss=False, device=None):
        super(PLIFSNN, self).__init__()

        self.v_reset = 0.0  # 0.0 # None
        self.bias = False  # False
        self.decay_inp = not quantize #False #True
        self.init_tau = 2.0  

        if quantize:
            self.quantize = quantize_8bit
        else:
            self.quantize = lambda x: x

        self.blocks = torch.nn.ModuleList([
            layer.Conv2d(inp_features, channels, kernel_size=3, padding=1, stride=2, bias=self.bias),
            layer.BatchNorm2d(channels, momentum=0.01, eps=1e-3),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=self.v_reset,
                                     decay_input=self.decay_inp, init_tau=self.init_tau),
            layer.Dropout(p=dropout),
            layer.Conv2d(channels, 2*channels, kernel_size=3, padding=1, stride=2, bias=self.bias),
            layer.BatchNorm2d(2*channels, momentum=0.01, eps=1e-3),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=self.v_reset,
                                     decay_input=self.decay_inp, init_tau=self.init_tau),
            layer.Dropout(p=dropout),
            layer.Conv2d(2*channels, 4*channels, kernel_size=3, padding=1, stride=2, bias=self.bias),
            layer.BatchNorm2d(4*channels, momentum=0.01, eps=1e-3),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=self.v_reset,
                                     decay_input=self.decay_inp, init_tau=self.init_tau),
            layer.Dropout(p=dropout),
            layer.Conv2d(4*channels, 8 * channels, kernel_size=3, padding=1, stride=2, bias=self.bias),
            layer.BatchNorm2d(8 * channels, momentum=0.01, eps=1e-3),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=self.v_reset,
                                     decay_input=self.decay_inp, init_tau=self.init_tau),
            layer.Dropout(p=dropout),
            layer.Flatten(),
            layer.Linear((8 * 8 * 8 * channels), feat_neur, bias=self.bias), #layer.Linear((12 * 12 * 8 * channels), feat_neur, bias=self.bias),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=self.v_reset,
                                     decay_input=self.decay_inp, init_tau=self.init_tau),
            layer.Linear(feat_neur, classes, bias=self.bias),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(),detach_reset=True, v_reset=self.v_reset, decay_input=self.decay_inp, init_tau=self.init_tau) if not ce_loss else nn.Identity(),
        ])

        for m in self.modules():
            if isinstance(m, (layer.Conv2d, layer.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,
                            (layer.BatchNorm2d, layer.BatchNorm1d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.step_mode = 'm'
        functional.set_step_mode(self, step_mode=self.step_mode)

        if isinstance(self.init_tau, float) and device==torch.device('cuda'):
            use_cupy = True  # False
        else:
            use_cupy = False

        if use_cupy and self.step_mode == 'm':
            functional.set_backend(self, backend='cupy')

        self.device = device
        self.lava_dl = False

    def set_step_mode(self, mode='m'):
        self.step_mode = mode
        functional.set_step_mode(self, step_mode=self.step_mode)
        if mode == 'm':
            functional.set_backend(self, backend='cupy')
        else:
            functional.set_backend(self, backend='torch')

    def forward(self, spike):
        if not self.lava_dl:
            functional.reset_net(self)
            # print(spike.size())
            spike = spike.permute(4, 0, 1, 2, 3).contiguous()
        if self.step_mode == 'm':
            count = []
            for block in self.blocks:
                spike = block(spike)
                count.append(torch.mean(spike).item())
        else:
            print('Not implemented!')

        if not self.lava_dl:
            spike = spike.permute(1, 2, 0).contiguous()
        return spike, count


    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = []
        for m in self.modules():
            if isinstance(m, (layer.Conv2d, layer.Linear, nn.Conv3d)):
                if m.weight.grad is None:
                    grad.append(0)
                else:
                    grad.append(torch.norm(m.weight.grad).item() / torch.numel(m.weight.grad))

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad
    
    def fuse_bn(self):
        """
        Fuse Conv2D and BatchNorm2D
        """
        self.blocks = fuse_bn(self.blocks)
        print(self.blocks)

    def to_lava_dl(self, quantize=True):
        '''
        Convert to lava-dl
        '''
        device = self.blocks[0].weight.device
        bias = None
        #self.fuse_bn()
        l_type = None
        syn = None
        blocks = torch.nn.ModuleList([])
        for i, b in enumerate(self.blocks):
            if isinstance(b, layer.Conv2d):
                bias = b.bias
                b.bias = None
                syn = b
                l_type = 'conv'
            elif isinstance(b, layer.Linear):
                bias = None #b.bias
                b.bias = None
                syn = b
                l_type = 'linear'
            elif isinstance(b, neuron.ParametricLIFNode):
                b.decay_input = False
                b.tau = float(1 / b.w.sigmoid())
                neur = b
                if l_type == 'conv':
                    b = to_lava_block_conv(syn, neur, quantize_to_8bit=quantize)
                elif l_type == 'linear':
                    b = to_lava_block_dense(syn, neur, quantize_to_8bit=quantize)
                b.neuron.bias = bias
                b.to(device)
                blocks.append(b)
            elif isinstance(b, layer.Flatten):
                b = slayer.block.cuba.Flatten()
                b.to(device)
                blocks.append(b)

        self.blocks = blocks
        print(self.blocks)

        self.lava_dl = True
        self.step_mode = 'm' 

    def export_hdf5(self, filename, add_input_layer=False, input_dims=[2, 128, 128]):
        # network export to hdf5 format
        with h5py.File(filename, 'w') as h:
            layer = h.create_group('layer')
            if add_input_layer:
                input_layer = layer.create_group(f'{0}')
                input_layer.create_dataset('shape', data=np.array(input_dims))
                input_layer.create_dataset('type', (1,), 'S10', ['input'.encode('ascii', 'ignore')])
            for i, b in enumerate(self.blocks):
                if add_input_layer:
                    b.export_hdf5(layer.create_group(f'{i + 1}'))
                else:
                    b.export_hdf5(layer.create_group(f'{i}'))   

        
class LoihiCuBaSNN(torch.nn.Module):
    def __init__(self, inp_features=2, channels=8, feat_neur=512, classes=12, delay=False, dropout=0.05, quantize=False, device=None):
        super(LoihiCuBaSNN, self).__init__()

        neuron_params = {
            'threshold': 1.25,
            'current_decay': 0.25,
            'voltage_decay': 0.03,
            'tau_grad': 0.03,
            'scale_grad': 3,
            'requires_grad': True,
        }
        neuron_params_drop = {**neuron_params, 'dropout': slayer.neuron.Dropout(p=dropout), }
        
        if quantize:
            self.quantize = quantize_8bit
            self.delay_shift = True
        else:
            self.quantize = lambda x: x
            self.delay_shift = False

        self.blocks = torch.nn.ModuleList([
            slayer.block.cuba.Conv(neuron_params=neuron_params_drop, in_features=inp_features, out_features=channels,
                                   kernel_size=3, padding=1, stride=2, 
                                   delay=delay, delay_shift=self.delay_shift, weight_norm=True, pre_hook_fx=self.quantize),
            slayer.block.cuba.Conv(neuron_params=neuron_params_drop, in_features=channels, out_features=2 * channels,
                                   kernel_size=3, padding=1, stride=2, 
                                   delay=delay, delay_shift=self.delay_shift, weight_norm=True, pre_hook_fx=self.quantize),
            slayer.block.cuba.Conv(neuron_params=neuron_params_drop, in_features=2 * channels,
                                   out_features=4 * channels, kernel_size=3, padding=1, stride=2, 
                                   delay=delay, delay_shift=self.delay_shift, weight_norm=True, pre_hook_fx=self.quantize),
            slayer.block.cuba.Conv(neuron_params=neuron_params_drop, in_features=4 * channels,
                                   out_features=8 * channels, kernel_size=3, padding=1, stride=2, 
                                   delay=delay, delay_shift=self.delay_shift, weight_norm=True, pre_hook_fx=self.quantize),
            slayer.block.cuba.Flatten(),
            slayer.block.cuba.Dense(neuron_params=neuron_params_drop, in_neurons=(8 * 8 * 8 * channels),
                                    out_neurons=feat_neur, weight_scale=10,
                                    delay=delay, delay_shift=self.delay_shift, weight_norm=True, pre_hook_fx=self.quantize),
            slayer.block.cuba.Dense(neuron_params=neuron_params, in_neurons=feat_neur,
                                    out_neurons=classes, weight_scale=10,
                                    delay=delay, delay_shift=self.delay_shift, weight_norm=True, pre_hook_fx=self.quantize),
        ])

    def forward(self, spike):
        # print(spike.size())
        count = []
        for block in self.blocks:
            spike = block(spike)
            count.append(torch.mean(spike).item())
        return spike, torch.FloatTensor(count).reshape(
            (1, -1)
        ).to(spike.device)

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [
            b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')
        ]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename, add_input_layer=False, input_dims=[2, 128, 128]):
        # network export to hdf5 format
        with h5py.File(filename, 'w') as h:
            layer = h.create_group('layer')
            if add_input_layer:
                input_layer = layer.create_group(f'{0}')
                input_layer.create_dataset('shape', data=np.array(input_dims))
                input_layer.create_dataset('type', (1,), 'S10', ['input'.encode('ascii', 'ignore')])
            for i, b in enumerate(self.blocks):
                if add_input_layer:
                    b.export_hdf5(layer.create_group(f'{i + 1}'))
                else:
                    b.export_hdf5(layer.create_group(f'{i}'))   
      
        
        
        
