# Copyright (C) 2025 fortiss GmbH
# SPDX-License-Identifier:  BSD-3-Clause

# benchmarking via pytorch hooks
# reference: https://github.com/NeuroBench/neurobench/tree/main

import torch
import numpy as np
import copy

import sys
#sys.path.insert(0, '/home/neumeier/Documents/corinne/libs/spikingjelly')
from spikingjelly.activation_based import neuron
from models.adlif import AdLIF
from models.egru import EGRU
from models.minegru import MinEGRU

class ActivationHook():
    """ Hook class for an activation layer in a NeuroBenchModel.

    Output of the activation layer in each forward pass will be stored.
    """

    def __init__(self, layer, connection_layer=None, prev_act_layer_hook=None):
        """ Initializes the class.

        A forward hook is registered for the activation layer.

        Args:
            layer: The activation layer which is a PyTorch nn.Module.
        """
        self.activation_outputs = []
        self.activation_inputs = []
        if layer is not None:
            self.hook = layer.register_forward_hook(self.hook_fn)
            self.hook_pre = layer.register_forward_pre_hook(self.pre_hook_fn)
        else:
            self.hook = None
            self.hook_pre = None

        self.layer = layer  # the activation layer

        # Check if the layer is a spiking layer (SpikingNeuron is the superclass of all snnTorch spiking layers)
        # Check if the layer is a spiking layer (BaseNode is the superclass of all spikingjelly spiking layers)
        self.spiking = isinstance(layer, neuron.BaseNode) or isinstance(layer, AdLIF) or isinstance(layer, EGRU)  or isinstance(layer, MinEGRU)

    def pre_hook_fn(self, layer, input):
        """Hook function that will be called before each forward pass of
        the activation layer.

        Each input of the activation layer will be stored.

        Args:
            layer: The registered layer
            input: Input of the registered layer
        """
        self.activation_inputs.append(input)

    def hook_fn(self, layer, input, output):
        """Hook function that will be called after each forward pass of
        the activation layer.

        Each output of the activation layer will be stored.

        Args:
            layer: The registered layer
            input: Input of the registered layer
            output: Output of the registered layer
        """
        if self.spiking:
            #self.activation_outputs.append(output[0])
            self.activation_outputs.append(output)

        else:
            self.activation_outputs.append(output)

    def empty_hook(self):
        """Deletes the contents of the hooks, but keeps the hook registered.
        """
        self.activation_outputs = []
        self.activation_inputs = []

    def reset(self):
        """ Resets the stored activation outputs and inputs
        """
        self.activation_outputs = []
        self.activation_inputs = []

    def close(self):
        """ Remove the registered hook.
        """
        if self.hook:
            self.hook.remove()
        if self.hook_pre:
            self.hook_pre.remove()


class LayerHook():
    def __init__(self, layer) -> None:
        self.layer = layer
        self.inputs = []
        if layer is not None:
            self.hook = layer.register_forward_pre_hook(self.hook_fn)
        else:
            self.hook = None

    def hook_fn(self, module, input):
        self.inputs.append(input)

    def register_hook(self):
        self.hook = self.layer.register_forward_pre_hook(self.hook_fn)

    def reset(self):
        self.inputs = []

    def close(self):
        if self.hook:
            self.hook.remove()


def detect_activations_connections(model):
    """Register hooks or other operations that should be called before running a benchmark.
    """
    for hook in model.activation_hooks:
        hook.reset()
        hook.close()
    for hook in model.connection_hooks:
        hook.reset()
        hook.close()
    model.activation_hooks = []
    model.connection_hooks = []

    #supported_layers = model.supported_layers

    # recurrent_supported_layers = (torch.nn.RNNBase)
    # recurr_cell_supported_layers = (torch.nn.RNNCellBase)

    #act_layers = model.activation_layers()
    # Registered activation hooks
    #for layer in act_layers:
    #    model.activation_hooks.append(ActivationHook(layer))

    #con_layers = model.connection_layers()
    #for flat_layer in con_layers:
    #    if isinstance(flat_layer, supported_layers):
    #        model.connection_hooks.append(LayerHook(flat_layer))

    supported_acts = (neuron.BaseNode, torch.nn.ReLU, AdLIF, EGRU, MinEGRU)

    supported_cons = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)

    for m in model.modules():
        if isinstance(m, supported_acts):
            model.activation_hooks.append(ActivationHook(m))
        elif isinstance(m, supported_cons):
            model.connection_hooks.append(LayerHook(m))


def activation_sparsity(model):
    """ Sparsity of model activations.

    Calculated as the number of zero activations over the total number
    of activations, over all layers, timesteps, samples in data.

    Args:
        model: A NeuroBenchModel.
        preds: A tensor of model predictions.
        data: A tuple of data and labels.
    Returns:
        float: Activation sparsity.
    """
    # TODO: for a spiking model, based on number of spikes over all timesteps over all samples from all layers
    #       Standard FF ANN depends on activation function, ReLU can introduce sparsity.
    total_spike_num, total_neuro_num = 0, 0
    for hook in model.activation_hooks:
        for spikes in hook.activation_outputs:  # do we need a function rather than a member
            spike_num, neuro_num = torch.count_nonzero(spikes).item(), torch.numel(spikes)
            total_spike_num += spike_num
            total_neuro_num += neuro_num

    sparsity = (total_neuro_num - total_spike_num) / total_neuro_num if total_neuro_num != 0 else 0.0
    return sparsity


def number_neuron_updates(model):
    """ Number of times each neuron type is updated.

    Args:
        model: A NeuroBenchModel.
        preds: A tensor of model predictions.
        data: A tuple of data and labels.
    Returns:
        dict: key is neuron type, value is number of updates.
    """
    # check_shape(preds, data[1])
    neuron_updates = 0

    update_dict = {}
    for hook in model.activation_hooks:
        for spikes_batch in hook.activation_inputs:
            for spikes in spikes_batch:
                nr_updates = torch.count_nonzero(spikes)
                if str(type(hook.layer)) not in update_dict:
                    update_dict[str(type(hook.layer))] = 0
                update_dict[str(type(hook.layer))] += int(nr_updates)
    # print formatting
    #print('Number of updates for:')
    for key in update_dict:
        #print(key, ':', update_dict[key])
        neuron_updates = neuron_updates + update_dict[key]
    return neuron_updates, update_dict


class SynapticOperations():
    """ Number of synaptic operations

    MACs for ANN
    ACs for SNN
    """

    def __init__(self):
        self.MAC = 0
        self.AC = 0
        self.total_synops = 0
        self.total_samples = 0

    def reset(self):
        self.MAC = 0
        self.AC = 0
        self.total_synops = 0
        self.total_samples = 0

    def __call__(self, model):
        """ Multiply-accumulates (MACs) of the model forward.

        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
            inputs: A tensor of model inputs.
        Returns:
            float: Multiply-accumulates.
        """
        for hook in model.connection_hooks:
            inputs = hook.inputs  # copy of the inputs, delete hooks after
            for spikes in inputs:
                # spikes is batch, features, see snntorchmodel wrappper
                # for single_in in spikes:
                if len(spikes) == 1:
                    spikes = spikes[0]
                hook.hook.remove()
                operations, spiking = single_layer_MACs(spikes, hook.layer)
                total_ops, _ = single_layer_MACs(spikes, hook.layer, total=True)
                self.total_synops += total_ops
                if spiking:
                    self.AC += operations
                else:
                    self.MAC += operations
                hook.register_hook()
        # ops_per_sample = ops / data[0].size(0)
        # assume batch_size = 1
        self.total_samples += 1 #data[0].size(0)
        syn_dict = self.compute()
        synops = np.array([syn_dict['Dense'], syn_dict['Effective_MACs'], syn_dict['Effective_ACs']])
        return synops, syn_dict

    def compute(self):
        if self.total_samples == 0:
            return {'Effective_MACs': 0, 'Effective_ACs': 0, 'Dense': 0}
        ac = self.AC / self.total_samples
        mac = self.MAC / self.total_samples
        total_synops = self.total_synops / self.total_samples
        return {'Effective_MACs': mac, 'Effective_ACs': ac, 'Dense': total_synops}


def check_shape(preds, labels):
    """ Checks that the shape of the predictions and labels are the same.
    """
    if preds.shape != labels.shape:
        raise ValueError("preds and labels must have the same shape")

def make_binary_copy(layer, all_ones=False):
    """ Makes a binary copy of the layer. All non 0 entries are made 1.
        If all_ones is True, then all entries are made 1.
    """
    layer_copy = copy.deepcopy(layer)

    stateless_layers = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d)
    # recurrent_layers = (torch.nn.RNNBase)
    recurrent_cells = (torch.nn.RNNCellBase)

    if isinstance(layer, stateless_layers):
        weights = layer_copy.weight.data
        weights[weights != 0] = int(1)
        if all_ones:
            weights[weights == 0] = int(1)

        if layer.bias is not None:
            biases = layer_copy.bias.data
            biases[biases != 0] = int(1)
            if all_ones:
                biases[biases == 0] = int(1)
            layer_copy.bias.data = biases

        layer_copy.weight.data = weights


    elif isinstance(layer, recurrent_cells):
        attribute_names = ['weight_ih', 'weight_hh']
        if layer.bias:
            attribute_names += ['bias_ih', 'bias_hh']
        # if layer.proj_size > 0: # it is lstm
        # 	attribute_names += ['weight_hr']

        for attr in attribute_names:
            with torch.no_grad():
                attr_val = getattr(layer_copy, attr)
                attr_val[attr_val != 0] = int(1)
                if all_ones:
                    attr_val[attr_val == 0] = int(1)
                setattr(layer_copy, attr, attr_val)

    return layer_copy

def single_layer_MACs(inputs, layer, total=False):
    """ Computes the MACs for a single layer.
        returns effective operations if total=False, else total operations (including zero operations)
        Supported layers: Linear, Conv1d, Conv2d, Conv3d, RNNCellBase, LSTMCell, GRUCell
    """
    macs = 0

    # copy input
    inputs, spiking, in_states = binary_inputs(inputs, all_ones=total)

    stateless_layers = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d)

    if isinstance(layer, stateless_layers):
        # then multiply the binary layer with the diagonal matrix to get the MACs
        layer_bin = make_binary_copy(layer, all_ones=total)

        # bias is not considered as a synaptic operation
        # in the future you can change this parameter to include bias
        bias = False
        if layer_bin.bias is not None and not bias:
            # suppress the bias to zero
            layer_bin.bias.data = torch.zeros_like(layer_bin.bias.data)

        nr_updates = layer_bin(
            inputs)  # this returns the number of MACs for every output neuron: if spiking neurons only AC
        macs = nr_updates.sum()

    else:
        print("No other layers supported right now!")

    return int(macs), spiking

def binary_inputs(inputs, all_ones=False):
    """ Returns a copy of the inputs with binary elements, all ones if all_ones is True"""
    in_states = True  # assume that input is tuple of inputs and states. If not, then set to False
    spiking = False

    with torch.no_grad():
        # TODO: should change this code block so that all inputs get cloned
        if isinstance(inputs, tuple):

            # input is first element, rest is hidden states
            test_ins = inputs[0]

            # NOTE: this only checks first input as everything else can be seen as hidden states in rnn block
            if len(test_ins[(test_ins != 0) & (test_ins != 1) & (test_ins != -1)]) == 0:
                spiking = True
            if not all_ones:
                inputs = cylce_tuple(inputs)
            else:
                inputs = cylce_tuple_ones(inputs)
        else:
            # clone tensor since it may be used as input to other layers
            inputs = inputs.detach().clone()
            in_states = False
            if len(inputs[(inputs != 0) & (inputs != 1) & (inputs != -1)]) == 0:
                spiking = True

            inputs[inputs != 0] = 1
            if all_ones:
                inputs[inputs == 0] = 1
    return inputs, spiking, in_states

def cylce_tuple(tup):
    """ Returns a copy of the tuple with binary elements
    """
    tup_copy = []
    for t in tup:
        if isinstance(t, tuple):
            tup_copy.append(cylce_tuple(t))
        elif t is not None:
            t = t.detach().clone()
            t[t != 0] = 1
            tup_copy.append(t)
    return tuple(tup_copy)


def cylce_tuple_ones(tup):
    """ Returns a copy of the tuple with ones elements
    """
    tup_copy = []
    for t in tup:
        if isinstance(t, tuple):
            tup_copy.append(cylce_tuple(t))
        elif t is not None:
            t = t.detach().clone()
            t[t != 0] = 1
            t[t == 0] = 1
            tup_copy.append(t)
    return tuple(tup_copy)