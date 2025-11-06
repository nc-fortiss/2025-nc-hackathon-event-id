import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import h5py

import lava.lib.dl.slayer as slayer
from lava.lib.dl.slayer.utils import quantize
from lava.lib.dl.slayer.neuron.base import Neuron
from lava.lib.dl.slayer.block.base import AbstractRecurrent, AbstractDense
from lava.lib.dl.slayer.synapse.layer import Dense, Conv
from lava.lib.dl.slayer.axon.delay import Delay, delay

def test_sigmoid():
    # test sigmoid
    import numpy as np
    x = np.linspace(-7, 7, 100)
    x_t = torch.tensor(x)
    x_sigmoid = x_t.sigmoid()
    x_loihi_sigmoid = loihi_sigmoid.forward(x_t)
    # plot
    fig, ax = plt.subplots()
    ax.plot(x, x_t.sigmoid().numpy(), 'x-', color='blue', label='sigmoid')
    ax.plot(x, x_loihi_sigmoid.numpy(), 'o-', color='red', label='loihi_sigmoid')
    ax.legend()
    plt.show()


class loihi_sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(x_input):
        # piecewise linear, according to: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9700744
        # To-Do: make it compatible with tensor -> torch.where
        #ctx.save_for_backward(x_input)
        x = x_input
        #if abs(x) >= 5:
        #    y = 1
        y0 = torch.where(x.abs() >= 5, 1, 0)
        #elif abs(x) >= 2.375:
        #    y = 0.03125 * x + 0.84375
        y1 = torch.where(((x.abs() < 5).bool() & (x.abs() >= 2.375).bool()), 0.03125 * x.abs() + 0.84375, 0)
        #elif abs(x) >= 1.0:
        #    y = 0.125 * x + 0.625
        y2 = torch.where(((x.abs() < 2.375).bool() & (x.abs() >= 1.0).bool()), 0.125 * x.abs() + 0.625, 0)
        #else:
        #    y = 0.25 * x + 0.5
        y3 = torch.where(x.abs() < 1.0, 0.25 * x.abs() + 0.5, 0)
        y = y0 + y1 + y2 + y3
        # mirror for negative x
        #if x < 0:
        #    y = 1 - y
        y = torch.where(x < 0, 1 - y, y)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        x_input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input


class SpikeFunction(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, inp, dampening_factor, pseudo_derivative_support):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(inp, dampening_factor, pseudo_derivative_support)
        return torch.heaviside(inp, inp)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        inp, dampening_factor, pseudo_derivative_support = ctx.saved_tensors
        dE_dz = grad_output

        dz_du = dampening_factor * torch.maximum(1 - pseudo_derivative_support * torch.abs(
            inp), torch.Tensor((0,)).to(grad_output.device))

        dE_dv = dE_dz * dz_du
        return dE_dv, None, None


class LavaGRUNeuron(Neuron):
    def __init__(
            self, threshold,
            binary=True, reset='hard', scale=1 << 6,
            norm=None, dropout=None, bias=False,
            shared_param=True, persistent_state=False, requires_grad=False
    ):
        """ """
        super(LavaGRUNeuron, self).__init__(
            threshold=threshold,
            w_scale=scale,
            s_scale=scale * (1 << 6),
            norm=norm,
            dropout=dropout,
            persistent_state=persistent_state,
            shared_param=shared_param,
            requires_grad=requires_grad
        )
        self.binary = binary
        self.bias = bias
        self.reset = reset
        #self.activation = activation
        self.shape = None

    @property
    def scale(self):
        """Scale difference between slayer representation and hardware
        representation of the variable states."""
        return self.w_scale

    @property
    def device_params(self):
        """Dictionary of device parameters."""
        return {
            'type': 'EGRU',
            #'activation': self.activation.__name__,
            'vThMant': self.v_th_mant,
            'binary': self.binary
        }

    def forward(self, x):
        pass

class LavaGRU(AbstractRecurrent):
    def __init__(self, *args, **kwargs):
        super(LavaGRU, self).__init__(*args, **kwargs)
        if self.neuron_params is not None:
            self.neuron = LavaGRUNeuron(**self.neuron_params)
        self.synapse_params['out_neurons'] = 3 * self.synapse_params['out_neurons']
        self.input_synapse = Dense(**self.synapse_params)
        self.recurrent_params['out_neurons'] = 3 * self.recurrent_params['out_neurons']
        self.recurrent_synapse = Dense(**self.recurrent_params)
        self.input_synapse.pre_hook_fx = self.neuron.quantize_8bit
        self.recurrent_synapse.pre_hook_fx = self.neuron.quantize_8bit

        #self.input_bias = kwargs['input_bias'] if 'input_bias' in kwargs.keys() else False
        #self.input_bias = self.neuron_params['bias']
        if self.neuron.bias:
            self.register_parameter(
                'inp_bias',
                torch.nn.Parameter(
                    torch.FloatTensor([self.num_neurons]),
                    requires_grad=True
                )
            )  # learnable bias
        #self.rec_bias = kwargs['rec_bias'] if 'rec_bias' in kwargs.keys() else False
        #self.rec_bias = self.neuron_params['bias']
        if self.neuron.bias:
            self.register_parameter(
                'rec_bias',
                torch.nn.Parameter(
                    torch.FloatTensor([self.num_neurons]),
                    requires_grad=True
                )
            )  # learnable bias
        del self.synapse_params
        del self.recurrent_params
        del self.neuron_params
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.7]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)

    def forward(self, x):
        """Forward computation method. The input can be either of ``NCT`` or
        ``NCHWT`` format.
        """

        h = []
        o = []
        y = []

        if self.neuron.bias:
            Wx = self.input_synapse(x) + self.inp_bias
        else:
            Wx = self.input_synapse(x)

        spike = torch.zeros((Wx.shape[0], Wx.shape[1] // 3)).to(x.device)
        hidden = torch.zeros((Wx.shape[0], Wx.shape[1] // 3)).to(x.device)

        #if Wx.shape[0] == self.spike_state.shape[0]:
        #    spike = spike + self.spike_state

        for t in range(x.shape[-1]):
            if self.neuron.bias:
                Rh = self.recurrent_synapse(spike.unsqueeze(-1)).squeeze(-1) + self.rec_bias
            else:
                Rh = self.recurrent_synapse(spike.unsqueeze(-1)).squeeze(-1)
            vx = torch.chunk(Wx[...,t], 3, 1)
            vh = torch.chunk(Rh, 3, 1)

            z = torch.sigmoid(vx[0] + vh[0])
            r = torch.sigmoid(vx[1] + vh[1])
            g = torch.tanh(vx[2] + r * vh[2])

            cur_h = (z * hidden + (1 - z) * g)
            #print(t, cur_h.max())

            if self.neuron.threshold != -1:
                event = SpikeFunction.apply(
                    cur_h - self.neuron.threshold, self.dampening_factor, self.pseudo_derivative_support)
                o.append(event)
                if self.neuron.reset == 'hard':
                    # hard reset
                    hidden = cur_h - event * cur_h
                elif self.neuron.reset == 'soft':
                    # soft reset
                    hidden = cur_h - event * self.neuron.threshold
                else:
                    # no reset
                    hidden = cur_h
                h.append(hidden)
                if self.neuron.binary:
                    spike = event
                else:
                    spike = event * cur_h
                y.append(spike)
            else:
                o.append(cur_h)
                hidden = cur_h
                h.append(hidden)
                #y = torch.nn.functional.relu(hidden)
                y.append(hidden)
                #print('No threshold')
                #sys.stdout.flush()
        y = torch.stack(y, dim=-1)
        h = torch.stack(h, dim=-1)
        o = torch.stack(o, dim=-1)

        #self.spike_state = spike.clone().detach().reshape((Wx.shape[0], Wx.shape[1] // 3))

        if self.neuron.binary:
            x = o
        else:
            x = y

        if self.neuron.drop:
            x = self.neuron.drop(x)
        
        if self.shape is None:
            self.neuron.shape = x.shape[1:-1]

        if self.count_log is True:
            return x, torch.mean(x > 0)
        else:
            return x

    @property
    def shape(self):
        """Shape of the block.
        """
        return self.neuron.shape

    def export_hdf5(self, handle):
        def weight(s):
            return s.pre_hook_fx(
                s.weight, descale=True
            ).reshape(s.weight.shape[:2]).cpu().data.numpy()

        def delay(d):
            return torch.floor(d.delay).flatten().cpu().data.numpy()

        handle.create_dataset(
            'type', (1, ), 'S10', ['dense_egru'.encode('ascii', 'ignore')]
        )

        handle.create_dataset('shape', data=np.array(self.neuron.shape))
        handle.create_dataset(
            'inFeatures', data=self.input_synapse.in_channels)
        handle.create_dataset(
            'outFeatures', data=self.input_synapse.out_channels)

        if self.input_synapse.weight_norm_enabled:
            self.input_synapse.disable_weight_norm()

        if hasattr(self.input_synapse, 'imag'):   # complex synapse
            handle.create_dataset(
                'weight/real',
                data=weight(self.input_synapse.real)
            )
            handle.create_dataset(
                'weight/imag',
                data=weight(self.input_synapse.imag)
            )
            raise NotImplementedError(f'Complex recurrent not implemented.')
        else:
            handle.create_dataset('weight', data=weight(self.input_synapse))
            handle.create_dataset(
                'weight_rec', data=weight(self.recurrent_synapse))

        # bias
        has_norm = False
        if hasattr(self.neuron, 'norm'):
            if self.neuron.norm is not None:
                has_norm = True
        if has_norm is True:
            handle.create_dataset(
                'bias',
                data=self.neuron.norm.bias.cpu().data.numpy().flatten()
            )
        if self.neuron.bias:
            handle.create_dataset(
                'bias',
                data=self.inp_bias.cpu().data.numpy().flatten()
            )
            handle.create_dataset(
                'bias_rec',
                data=self.rec_bias.cpu().data.numpy().flatten()
            )

        # delay
        if self.delay is not None:
            self.delay.clamp()  # clamp the delay value
            handle.create_dataset('delay', data=delay(self.delay))

        # neuron
        for key, value in self.neuron.device_params.items():
            handle.create_dataset(f'neuron/{key}', data=value)
        if has_norm is True:
            if hasattr(self.neuron.norm, 'weight_exp'):
                handle.create_dataset(
                    'neuron/weight_exp',
                    data=self.neuron.norm.weight_exp
                )

class SimpleLavaGRU(AbstractRecurrent):
    def __init__(self, *args, **kwargs):
        super(SimpleLavaGRU, self).__init__(*args, **kwargs)
        if self.neuron_params is not None:
            self.neuron = LavaGRUNeuron(**self.neuron_params)
        self.synapse_params['out_neurons'] = 2 * self.synapse_params['out_neurons']
        self.input_synapse = Dense(**self.synapse_params)
        self.recurrent_params['out_neurons'] = 2 * self.recurrent_params['out_neurons']
        self.recurrent_synapse = Dense(**self.recurrent_params)
        #self.input_synapse.pre_hook_fx = self.neuron.quantize_8bit
        #self.recurrent_synapse.pre_hook_fx = self.neuron.quantize_8bit

        #self.input_bias = kwargs['input_bias'] if 'input_bias' in kwargs.keys() else False
        #self.input_bias = self.neuron_params['bias']
        if self.neuron.bias:
            self.register_parameter(
                'inp_bias',
                torch.nn.Parameter(
                    torch.FloatTensor([self.num_neurons]),
                    requires_grad=True
                )
            )  # learnable bias
        #self.rec_bias = kwargs['rec_bias'] if 'rec_bias' in kwargs.keys() else False
        #self.rec_bias = self.neuron_params['bias']
        if self.neuron.bias:
            self.register_parameter(
                'rec_bias',
                torch.nn.Parameter(
                    torch.FloatTensor([self.num_neurons]),
                    requires_grad=True
                )
            )  # learnable bias
        del self.synapse_params
        del self.recurrent_params
        del self.neuron_params
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.7]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)

    def forward(self, x):
        """Forward computation method. The input can be either of ``NCT`` or
        ``NCHWT`` format.
        """

        h = []
        o = []
        y = []

        if self.neuron.bias:
            Wx = self.input_synapse(x) + self.inp_bias
        else:
            Wx = self.input_synapse(x)

        spike = torch.zeros((Wx.shape[0], Wx.shape[1] // 2)).to(x.device)
        hidden = torch.zeros((Wx.shape[0], Wx.shape[1] // 2)).to(x.device)

        #if Wx.shape[0] == self.spike_state.shape[0]:
        #    spike = spike + self.spike_state

        for t in range(x.shape[-1]):
            if self.neuron.bias:
                Rh = self.recurrent_synapse(spike.unsqueeze(-1)).squeeze(-1) + self.rec_bias
            else:
                Rh = self.recurrent_synapse(spike.unsqueeze(-1)).squeeze(-1)
            vx = torch.chunk(Wx[...,t], 2, 1)
            vh = torch.chunk(Rh, 2, 1)

            if self.delay_shift is True:
                z = torch.nn.functional.hardsigmoid(vx[0] + vh[0])
                g = torch.nn.functional.hardtanh(vx[1] + vh[1])
            else:
                z = torch.sigmoid(vx[0] + vh[0])
                g = torch.tanh(vx[1] + vh[1])
            
            cur_h = (z * hidden + (1 - z) * g)

            if self.neuron.threshold != -1:
                event = SpikeFunction.apply(
                    cur_h - self.neuron.threshold, self.dampening_factor, self.pseudo_derivative_support)
                o.append(event)
                if self.neuron.reset == 'hard':
                    # hard reset
                    hidden = cur_h - event * cur_h
                elif self.neuron.reset == 'soft':
                    # soft reset
                    hidden = cur_h - event * self.neuron.threshold
                else:
                    # no reset
                    hidden = cur_h
                h.append(hidden)
                if self.neuron.binary:
                    spike = event
                else:
                    spike = event * cur_h
                y.append(spike)
            else:
                o.append(cur_h)
                hidden = cur_h
                h.append(hidden)
                #y = torch.nn.functional.relu(hidden)
                y.append(hidden)
                #print('No threshold')
                #sys.stdout.flush()

        y = torch.stack(y, dim=-1)
        h = torch.stack(h, dim=-1)
        o = torch.stack(o, dim=-1)

        #self.spike_state = spike.clone().detach().reshape((Wx.shape[0], Wx.shape[1] // 2))

        if self.neuron.binary:
            x = o
        else:
            x = y

        if self.neuron.drop:
            x = self.neuron.drop(x)

        if self.shape is None:
            self.neuron.shape = x.shape[1:-1]

        if self.count_log is True:
            return x, torch.mean(x > 0)
        else:
            return x

    @property
    def shape(self):
        """Shape of the block.
        """
        return self.neuron.shape

    def export_hdf5(self, handle):
        def weight(s):
            return s.pre_hook_fx(
                s.weight, descale=True
            ).reshape(s.weight.shape[:2]).cpu().data.numpy()

        def delay(d):
            return torch.floor(d.delay).flatten().cpu().data.numpy()

        handle.create_dataset(
            'type', (1, ), 'S10', ['dense_egru'.encode('ascii', 'ignore')]
        )

        handle.create_dataset('shape', data=np.array(self.neuron.shape))
        handle.create_dataset(
            'inFeatures', data=self.input_synapse.in_channels)
        handle.create_dataset(
            'outFeatures', data=self.input_synapse.out_channels)

        if self.input_synapse.weight_norm_enabled:
            self.input_synapse.disable_weight_norm()

        if hasattr(self.input_synapse, 'imag'):   # complex synapse
            handle.create_dataset(
                'weight/real',
                data=weight(self.input_synapse.real)
            )
            handle.create_dataset(
                'weight/imag',
                data=weight(self.input_synapse.imag)
            )
            raise NotImplementedError(f'Complex recurrent not implemented.')
        else:
            handle.create_dataset('weight', data=weight(self.input_synapse))
            handle.create_dataset(
                'weight_rec', data=weight(self.recurrent_synapse))

        # bias
        has_norm = False
        if hasattr(self.neuron, 'norm'):
            if self.neuron.norm is not None:
                has_norm = True
        if has_norm is True:
            handle.create_dataset(
                'bias',
                data=self.neuron.norm.bias.cpu().data.numpy().flatten()
            )
        if self.neuron.bias:
            handle.create_dataset(
                'bias',
                data=self.inp_bias.cpu().data.numpy().flatten()
            )
            handle.create_dataset(
                'bias_rec',
                data=self.rec_bias.cpu().data.numpy().flatten()
            )

        # delay
        if self.delay is not None:
            self.delay.clamp()  # clamp the delay value
            handle.create_dataset('delay', data=delay(self.delay))

        # neuron
        for key, value in self.neuron.device_params.items():
            handle.create_dataset(f'neuron/{key}', data=value)
        if has_norm is True:
            if hasattr(self.neuron.norm, 'weight_exp'):
                handle.create_dataset(
                    'neuron/weight_exp',
                    data=self.neuron.norm.weight_exp
                )

class LavaMinGRU(AbstractDense):
    def __init__(self, *args, **kwargs):
        super(LavaMinGRU, self).__init__(*args, **kwargs)
        if self.neuron_params is not None:
            self.neuron = LavaGRUNeuron(**self.neuron_params)
        self.synapse_params['out_neurons'] = 2 * self.synapse_params['out_neurons']
        self.input_synapse = Dense(**self.synapse_params)
        #self.input_synapse.pre_hook_fx = self.neuron.quantize_8bit

        print('Reset:', self.neuron.reset)

        #self.input_bias = kwargs['input_bias'] if 'input_bias' in kwargs.keys() else False
        #self.input_bias = self.neuron_params['bias']

        if self.neuron.shared_param:
            self.register_parameter(
                'threshold',
                torch.nn.Parameter(
                    torch.FloatTensor([self.neuron.threshold]),
                    requires_grad=self.neuron.requires_grad
                )
            )  
        else:
            self.register_parameter(
                'threshold',
                torch.nn.Parameter(
                    torch.FloatTensor(torch.ones(self.synapse_params['out_neurons'])* self.neuron.threshold),
                    requires_grad=self.neuron.requires_grad
                )
            )  

        if self.neuron.bias:
            self.register_parameter(
                'inp_bias',
                torch.nn.Parameter(
                    torch.FloatTensor([self.num_neurons]),
                    requires_grad=True
                )
            )  # learnable bias
        del self.synapse_params
        del self.neuron_params
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.7]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)

    def forward(self, x):
        """Forward computation method. The input can be either of ``NCT`` or
        ``NCHWT`` format.
        """

        h = []
        o = []
        y = []
        z_out = []

        if self.neuron.bias:
            Wx = self.input_synapse(x) + self.inp_bias
        else:
            Wx = self.input_synapse(x)

        if self.delay_shift is True:
            Wx = delay(Wx, 1)

        spike = torch.zeros((Wx.shape[0], Wx.shape[1] // 2)).to(x.device)
        hidden = torch.zeros((Wx.shape[0], Wx.shape[1] // 2)).to(x.device)

        for t in range(x.shape[-1]):
            vx = torch.chunk(Wx[...,t], 2, 1)

            if self.delay_shift is True:
                z = torch.nn.functional.hardsigmoid(vx[0])
            else:
                z = torch.sigmoid(vx[0])
            #z = vx[0]
            g = vx[1]

            cur_h = (z * hidden + (1 - z) * g)

            if self.neuron.threshold != -1:
                event = SpikeFunction.apply(
                    cur_h - self.neuron.threshold, self.dampening_factor, self.pseudo_derivative_support)
                o.append(event)
                if self.neuron.reset == 'hard':
                    # hard reset
                    hidden = cur_h - event * cur_h
                elif self.neuron.reset == 'soft':
                    # soft reset
                    hidden = cur_h - event * self.neuron.threshold
                else:
                    # no reset
                    hidden = cur_h
                h.append(hidden)
                if self.neuron.binary:
                    spike = event
                else:
                    spike = event * cur_h
                y.append(spike)
                z_out.append(cur_h)
            else:
                o.append(cur_h)
                hidden = cur_h
                h.append(hidden)
                #y = torch.nn.functional.relu(hidden)
                y.append(hidden)
                z_out.append(cur_h)
                #print('No threshold')
                #sys.stdout.flush()


        y = torch.stack(y, dim=-1)
        h = torch.stack(h, dim=-1)
        o = torch.stack(o, dim=-1)
        z_out = torch.stack(z_out, dim=-1)

        if self.neuron.binary:
            x = o
        else:
            x = y

        if self.neuron.drop:
            x = self.neuron.drop(x)
        
        if self.shape is None:
            self.neuron.shape = x.shape[1:-1]

        if self.count_log is True:
            return x, torch.mean(x > 0)
        else:
            return x #z_out

    @property
    def shape(self):
        """Shape of the block.
        """
        return self.neuron.shape

    def export_hdf5(self, handle):
        def weight(s):
            return s.pre_hook_fx(
                s.weight, descale=True
            ).reshape(s.weight.shape[:2]).cpu().data.numpy()

        def delay(d):
            return torch.floor(d.delay).flatten().cpu().data.numpy()

        handle.create_dataset(
            'type', (1, ), 'S10', ['dense_egu'.encode('ascii', 'ignore')]
        )

        handle.create_dataset('shape', data=np.array(self.neuron.shape))
        handle.create_dataset(
            'inFeatures', data=self.input_synapse.in_channels)
        handle.create_dataset(
            'outFeatures', data=self.input_synapse.out_channels)

        if self.input_synapse.weight_norm_enabled:
            self.input_synapse.disable_weight_norm()

        if hasattr(self.input_synapse, 'imag'):   # complex synapse
            handle.create_dataset(
                'weight/real',
                data=weight(self.input_synapse.real)
            )
            handle.create_dataset(
                'weight/imag',
                data=weight(self.input_synapse.imag)
            )
            raise NotImplementedError(f'Complex recurrent not implemented.')
        else:
            #print(self.input_synapse.weight.shape)
            #print(self.input_synapse.weight)
            #wz, wh = torch.chunk(self.input_synapse.weight.data, 2, 0)
            #wz = wz / 6
            #self.input_synapse.weight.data = torch.cat((wz, wh), dim=0)
            #print(self.input_synapse.weight)
            handle.create_dataset('weight', data=weight(self.input_synapse))

        # bias
        has_norm = False
        if hasattr(self.neuron, 'norm'):
            if self.neuron.norm is not None:
                has_norm = True
        if has_norm is True:
            handle.create_dataset(
                'bias',
                data=self.neuron.norm.bias.cpu().data.numpy().flatten()
            )
        if self.neuron.bias:
            handle.create_dataset(
                'bias',
                data=self.inp_bias.cpu().data.numpy().flatten()
            )

        # delay
        if self.delay is not None:
            self.delay.clamp()  # clamp the delay value
            handle.create_dataset('delay', data=delay(self.delay))

        # correct threshold
        self.neuron.threshold = self.threshold
        
        # neuron
        for key, value in self.neuron.device_params.items():
            handle.create_dataset(f'neuron/{key}', data=value)
        if has_norm is True:
            if hasattr(self.neuron.norm, 'weight_exp'):
                handle.create_dataset(
                    'neuron/weight_exp',
                    data=self.neuron.norm.weight_exp
                )

class LavaConvMinGRU(AbstractDense):
    def __init__(self, *args, **kwargs):
        super(LavaConvMinGRU, self).__init__(*args, **kwargs)
        if self.neuron_params is not None:
            self.neuron = LavaGRUNeuron(**self.neuron_params)
        self.synapse_params['out_neurons'] = 2 * self.synapse_params['out_neurons']
        self.input_synapse = Conv(**self.synapse_params)
        #self.input_synapse.pre_hook_fx = self.neuron.quantize_8bit
        print('Reset:', self.neuron.reset)

        #self.input_bias = kwargs['input_bias'] if 'input_bias' in kwargs.keys() else False
        #self.input_bias = self.neuron_params['bias']
        if self.neuron.bias:
            self.register_parameter(
                'inp_bias',
                torch.nn.Parameter(
                    torch.FloatTensor([self.num_neurons]),
                    requires_grad=True
                )
            )  # learnable bias
        del self.synapse_params
        del self.neuron_params
        self.dampening_factor = nn.Parameter(
            torch.Tensor([0.7]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)

    def forward(self, x):
        """Forward computation method. The input can be either of ``NCT`` or
        ``NCHWT`` format.
        """

        h = []
        o = []
        y = []
        z_out = []

        if self.neuron.bias:
            Wx = self.input_synapse(x) + self.inp_bias
        else:
            Wx = self.input_synapse(x)

        if self.delay_shift is True:
            Wx = delay(Wx, 1)

        spike = torch.zeros((Wx.shape[0], Wx.shape[1]// 2, Wx.shape[2], Wx.shape[3])).to(x.device)
        hidden = torch.zeros((Wx.shape[0], Wx.shape[1]// 2, Wx.shape[2], Wx.shape[3])).to(x.device)

        for t in range(x.shape[-1]):
            vx = torch.chunk(Wx[...,t], 2, 1)

            if self.delay_shift is True:
                z = torch.nn.functional.hardsigmoid(vx[0])
            else:
                z = torch.sigmoid(vx[0])
            #z = vx[0]
            g = vx[1]

            cur_h = (z * hidden + (1 - z) * g)

            if self.neuron.threshold != -1:
                event = SpikeFunction.apply(
                    cur_h - self.neuron.threshold, self.dampening_factor, self.pseudo_derivative_support)
                o.append(event)
                if self.neuron.reset == 'hard':
                    # hard reset
                    hidden = cur_h - event * cur_h
                elif self.neuron.reset == 'soft':
                    # soft reset
                    hidden = cur_h - event * self.neuron.threshold
                else:
                    # no reset
                    hidden = cur_h
                h.append(hidden)
                if self.neuron.binary:
                    spike = event
                else:
                    spike = event * cur_h
                y.append(spike)
                z_out.append(cur_h)
            else:
                o.append(cur_h)
                hidden = cur_h
                h.append(hidden)
                #y = torch.nn.functional.relu(hidden)
                y.append(hidden)
                z_out.append(cur_h)
                #print('No threshold')
                #sys.stdout.flush()


        y = torch.stack(y, dim=-1)
        h = torch.stack(h, dim=-1)
        o = torch.stack(o, dim=-1)
        z_out = torch.stack(z_out, dim=-1)

        if self.neuron.binary:
            x = o
        else:
            x = y

        if self.shape is None:
            self.neuron.shape = x.shape[1:-1]

        if self.count_log is True:
            return x, torch.mean(x > 0)
        else:
            return x #z_out

    @property
    def shape(self):
        """Shape of the block.
        """
        return self.neuron.shape

    def export_hdf5(self, handle):
        def weight(s):
            return s.pre_hook_fx(
                s.weight, descale=True
            ).reshape(s.weight.shape[:2]).cpu().data.numpy()

        def delay(d):
            return torch.floor(d.delay).flatten().cpu().data.numpy()

        handle.create_dataset(
            'type', (1, ), 'S10', ['dense_egu'.encode('ascii', 'ignore')]
        )

        handle.create_dataset('shape', data=np.array(self.neuron.shape))
        handle.create_dataset(
            'inFeatures', data=self.input_synapse.in_channels)
        handle.create_dataset(
            'outFeatures', data=self.input_synapse.out_channels)

        if self.input_synapse.weight_norm_enabled:
            self.input_synapse.disable_weight_norm()

        if hasattr(self.input_synapse, 'imag'):   # complex synapse
            handle.create_dataset(
                'weight/real',
                data=weight(self.input_synapse.real)
            )
            handle.create_dataset(
                'weight/imag',
                data=weight(self.input_synapse.imag)
            )
            raise NotImplementedError(f'Complex recurrent not implemented.')
        else:
            #print(self.input_synapse.weight.shape)
            #print(self.input_synapse.weight)
            #wz, wh = torch.chunk(self.input_synapse.weight.data, 2, 0)
            #wz = wz / 6
            #self.input_synapse.weight.data = torch.cat((wz, wh), dim=0)
            #print(self.input_synapse.weight)
            handle.create_dataset('weight', data=weight(self.input_synapse))

        # bias
        has_norm = False
        if hasattr(self.neuron, 'norm'):
            if self.neuron.norm is not None:
                has_norm = True
        if has_norm is True:
            handle.create_dataset(
                'bias',
                data=self.neuron.norm.bias.cpu().data.numpy().flatten()
            )
        if self.neuron.bias:
            handle.create_dataset(
                'bias',
                data=self.inp_bias.cpu().data.numpy().flatten()
            )

        # delay
        if self.delay is not None:
            self.delay.clamp()  # clamp the delay value
            handle.create_dataset('delay', data=delay(self.delay))

        # neuron
        for key, value in self.neuron.device_params.items():
            handle.create_dataset(f'neuron/{key}', data=value)
        if has_norm is True:
            if hasattr(self.neuron.norm, 'weight_exp'):
                handle.create_dataset(
                    'neuron/weight_exp',
                    data=self.neuron.norm.weight_exp
                )