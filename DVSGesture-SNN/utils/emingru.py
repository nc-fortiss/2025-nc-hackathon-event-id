"""Event Min Gated Recurrent Unit"""
import math
import sys
#import evnn_pytorch_lib as LIB
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .egru import BaseRNN, _validate_state, SpikeFunction

from spikingjelly.activation_based import surrogate, neuron, functional, layer


# @torch.jit.script
def MinEGRUScript(
        training: bool,
        zoneout_prob: float,
        dampening_factor: float,
        pseudo_derivative_support: float,
        input,
        h0,
        kernel,
        synapse,
        bias,
        #recurrent_bias,
        thr,
        zoneout_mask,
        benchmark_mode=False,):
    """
    Perform Min EGRU computation using Pytorch primitives.

    :param training: bool,
    :type training: bool
    :param zoneout_prob: the probability of zoneout
    :type zoneout_prob: float
    :param dampening_factor: This is the dampening factor for the spike function
    :type dampening_factor: float
    :param pseudo_derivative_support: float,
    :type pseudo_derivative_support: float
    :param input: the input to the RNN, of shape (time_steps, batch_size, input_size)
    :param h0: initial hidden state
    :param kernel: the input weight matrix
    :param recurrent_kernel: the recurrent weight matrix
    :param bias: bias vector
    :param recurrent_bias: bias for recurrent kernel
    :param thr: threshold
    :param zoneout_mask: a mask that is used to randomly set some of the hidden units to zero
    :return: The output of the EGRU cell, the hidden state, the output of the spike function, and the
    trace values.
    """

    time_steps = input.shape[0]
    batch_size = input.shape[1]
    hidden_size = h0.shape[1]

    h = [torch.zeros_like(h0)]
    o = [torch.zeros_like(h0)]
    y = [h0]
    Wx = input @ kernel + bias
    for t in range(time_steps):
        vx = torch.chunk(Wx[t], 2, 1)

        z = torch.sigmoid(vx[0])
        g = vx[1]

        cur_h = (z * h[t] + (1 - z) * g)
        if zoneout_prob:
            if training:
                cur_h = (cur_h - h[t]) * zoneout_mask[t] + h[t]
            else:
                cur_h = zoneout_prob * h[t] + (1 - zoneout_prob) * h[t]

        event = SpikeFunction.apply(
            cur_h - thr, dampening_factor, pseudo_derivative_support)
        o.append(event)
        h.append(cur_h - event * thr)
        y.append(event * cur_h)

    y = torch.stack(y)
    h = torch.stack(h)
    o = torch.stack(o)

    if benchmark_mode:
        Wx = synapse(input.flatten(0, 1))

    tr_vals = torch.zeros_like(y)
    alpha = 0.9
    for t in range(1, time_steps + 1):
        tr_vals[t] = alpha * tr_vals[t - 1] + (1 - alpha) * y[t]

    return y, h, o, tr_vals


class MinEGRU(BaseRNN):
    """
    Minimal Event based Gated Recurrent Unit layer.

    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 batch_first=False,
                 binary=False,
                 dropout=0.0,
                 zoneout=0.0,
                 dampening_factor=0.7,
                 pseudo_derivative_support=1.0,
                 thr_mean=0.3,
                 weight_initialization_gain=1.0,
                 return_state_sequence=False,
                 grad_clip=None,
                 use_custom_cuda=True):
        """
        Initialize the parameters of the GRU layer.

        Arguments:
          input_size: int, the feature dimension of the input.
          hidden_size: int, the feature dimension of the output.
          batch_first: (optional) bool, if `True`, then the input and output
            tensors are provided as `(batch, seq, feature)`.
          dropout: (optional) float, sets the dropout rate for DropConnect
            regularization on the recurrent matrix.
          zoneout: (optional) float, sets the zoneout rate for Zoneout
            regularization.
          return_state_sequence: (optional) bool, if `True`, the forward pass will
            return the entire state sequence instead of just the final state. Note
            that if the input is a padded sequence, the returned state will also
            be a padded sequence.
          grad_clip: (optional) float, sets the gradient clipping value.
          use_custom_cuda (optional) bool, if `True`, the cuda code is used else
            pytorch implementation is used.

        """
        super().__init__(input_size, hidden_size, batch_first, zoneout, return_state_sequence)
        self.use_custom_cuda = False #use_custom_cuda

        if grad_clip:
            self.grad_clip_norm(enable=True, norm=grad_clip)
        else:
            self.grad_clip_norm(False)

        if dropout < 0 or dropout > 1:
            raise ValueError('GRU: dropout must be in [0.0, 1.0]')
        if zoneout < 0 or zoneout > 1:
            raise ValueError('GRU: zoneout must be in [0.0, 1.0]')

        self.dropout = dropout
        self.alpha = torch.tensor(0.9)
        self.binary = binary

        self.weight_initialization_gain = weight_initialization_gain

        self.kernel = nn.Parameter(torch.empty(input_size, hidden_size * 2))
        #self.recurrent_kernel = nn.Parameter(
        #    torch.empty(hidden_size, hidden_size * 3))
        self.bias = nn.Parameter(torch.empty(hidden_size * 2))
        #self.recurrent_bias = nn.Parameter(torch.empty(hidden_size * 3))
        self.reset_parameters()

        self.dampening_factor = nn.Parameter(
            torch.Tensor([dampening_factor]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([pseudo_derivative_support]), requires_grad=False)

        # initialize thresholds according to the beta distribution with mean 'thr_mean'
        assert 0 < thr_mean < 1, f"thr_mean must be between 0 and 1, but {thr_mean} was given"
        beta = 3
        alpha = beta * thr_mean / (1 - thr_mean)
        distribution = torch.distributions.beta.Beta(alpha, beta)
        self.thr = nn.Parameter(distribution.sample(torch.Size([self.hidden_size])))
        self.benchmark_mode = False
        self.synapse = None


    def reset_parameters(self):
        """Resets this layer's parameters to their initial values."""
        for k, v in self.named_parameters():
            if k in ['kernel', 'recurrent_kernel', 'bias', 'recurrent_bias']:
                if v.data.ndimension() >= 2:
                    nn.init.xavier_normal_(v, gain=self.weight_initialization_gain)
                else:
                    nn.init.zeros_(v)

    def grad_clip_norm(self, enable=True, norm=2.0):
        self._enable_grad_clip = enable
        self._max_norm = nn.Parameter(torch.Tensor(
            [norm if enable else -1.0]), requires_grad=False)

    def to_benchmark_mode(self):
        input_size = self.kernel.shape[0]
        hidden_size = int(self.kernel.shape[1] / 2)
        self.synapse = nn.Linear(input_size, 2*hidden_size, bias=False).to(self.kernel.device)
        self.synapse.weight.data = self.kernel.data.permute(1, 0).contiguous()
        self.benchmark_mode = True

    def forward(self, input, state=None, lengths=None):
        """
        Runs a forward pass of the Minimal EGRU layer.

        Arguments:
          input: Tensor, a batch of input sequences to pass through the GRU.
            Dimensions (seq_len, batch_size, input_size) if `batch_first` is
            `False`, otherwise (batch_size, seq_len, input_size).
          lengths: (optional) Tensor, list of sequence lengths for each batch
            element. Dimension (batch_size). This argument may be omitted if
            all batch elements are unpadded and have the same sequence length.

        Returns:
          output: Tensor, the output of the EGRU layer. Dimensions
            (seq_len, batch_size, hidden_size) if `batch_first` is `False` (default)
            or (batch_size, seq_len, hidden_size) if `batch_first` is `True`. Note
            that if `lengths` was specified, the `output` tensor will not be
            masked. It's the caller's responsibility to either not use the invalid
            entries or to mask them out before using them.
          h: the hidden state for all sequences. Dimensions
            (seq_len, batch_size, hidden_size).
          o: the output gate for all sequences (values: 0 or 1).
          trace: smoothed output values, can be beneficial for training.
        """
        input = self._permute(input)
        state_shape = [1, input.shape[1], self.hidden_size]
        h0 = self._get_state(input, state, state_shape)

        # restrict thresholds to be between 0 and 1
        self.thr.data.clamp_(min=0.0, max=1.0)

        # run forward pass
        y, h, o, trace = self._impl(
            input, h0[0], self.thr, self._get_zoneout_mask(input))

        # prepare outputs
        output = self._permute(y[1:])
        h = self._permute(h[1:])
        o = self._permute(o[1:])
        trace = self._permute(trace[1:])
        if self.binary:
            return o
        else:
            return output #, (h, o, trace)

    def _impl(self, input, state, thr, zoneout_mask):

        return MinEGRUScript(
            self.training,
            self.zoneout,
            self.dampening_factor,
            self.pseudo_derivative_support,
            input.contiguous(),
            state.contiguous(),
            self.kernel.contiguous(),
            self.synapse,
            self.bias.contiguous(),
            # self.recurrent_bias.contiguous(),
            thr.contiguous(),
            zoneout_mask.contiguous(),
            self.benchmark_mode)

# @torch.jit.script
def ConvMinEGRUScript(
        training: bool,
        zoneout_prob: float,
        dampening_factor: float,
        pseudo_derivative_support: float,
        input,
        h0,
        conv,
        bn,
        thr,
        zoneout_mask):
    """
    Perform Min EGRU computation using Pytorch primitives.

    :param training: bool,
    :type training: bool
    :param zoneout_prob: the probability of zoneout
    :type zoneout_prob: float
    :param dampening_factor: This is the dampening factor for the spike function
    :type dampening_factor: float
    :param pseudo_derivative_support: float,
    :type pseudo_derivative_support: float
    :param input: the input to the RNN, of shape (time_steps, batch_size, input_size)
    :param h0: initial hidden state
    :param kernel: the input weight matrix
    :param recurrent_kernel: the recurrent weight matrix
    :param bias: bias vector
    :param recurrent_bias: bias for recurrent kernel
    :param thr: threshold
    :param zoneout_mask: a mask that is used to randomly set some of the hidden units to zero
    :return: The output of the EGRU cell, the hidden state, the output of the spike function, and the
    trace values.
    """

    time_steps = input.shape[0]
    batch_size = input.shape[1]
    hidden_size = h0.shape[1]

    h = [torch.zeros_like(h0)]
    o = [torch.zeros_like(h0)]
    y = [h0]
    Wx = conv(input)
    if bn:
        Wx = bn(Wx)

    for t in range(time_steps):
        vx = torch.chunk(Wx[t], 2, 1)

        z = torch.sigmoid(vx[0])
        g = vx[1]

        cur_h = (z * h[t] + (1 - z) * g)
        # permute
        cur_h = cur_h.permute(0, 2, 3, 1)
        if zoneout_prob:
            if training:
                cur_h = (cur_h - h[t]) * zoneout_mask[t] + h[t]
            else:
                cur_h = zoneout_prob * h[t] + (1 - zoneout_prob) * h[t]

        event = SpikeFunction.apply(
            cur_h - thr, dampening_factor, pseudo_derivative_support)
        # permute back

        o.append(event.permute(0, 3, 1, 2))
        h.append((cur_h - event * thr).permute(0, 3, 1, 2))
        y.append((event * cur_h).permute(0, 3, 1, 2))

    y = torch.stack(y)
    h = torch.stack(h)
    o = torch.stack(o)

    tr_vals = torch.zeros_like(y)
    alpha = 0.9
    #for t in range(1, time_steps + 1):
    #    tr_vals[t] = alpha * tr_vals[t - 1] + (1 - alpha) * y[t]

    return y, h, o, tr_vals

class ConvMinEGRU(BaseRNN):
    """
    Minimal Event based Gated Recurrent Unit layer.

    """

    def __init__(self,
                 input_channels,
                 hidden_channels,
                 padding=1,
                 stride=1,
                 kernel_size=3,
                 bn=True,
                 batch_first=False,
                 binary=False,
                 dropout=0.0,
                 zoneout=0.0,
                 dampening_factor=0.7,
                 pseudo_derivative_support=1.0,
                 thr_mean=0.3,
                 weight_initialization_gain=1.0,
                 return_state_sequence=True,
                 grad_clip=None,
                 use_custom_cuda=True):
        """
        Initialize the parameters of the GRU layer.

        Arguments:
          input_size: int, the feature dimension of the input.
          hidden_size: int, the feature dimension of the output.
          batch_first: (optional) bool, if `True`, then the input and output
            tensors are provided as `(batch, seq, feature)`.
          dropout: (optional) float, sets the dropout rate for DropConnect
            regularization on the recurrent matrix.
          zoneout: (optional) float, sets the zoneout rate for Zoneout
            regularization.
          return_state_sequence: (optional) bool, if `True`, the forward pass will
            return the entire state sequence instead of just the final state. Note
            that if the input is a padded sequence, the returned state will also
            be a padded sequence.
          grad_clip: (optional) float, sets the gradient clipping value.
          use_custom_cuda (optional) bool, if `True`, the cuda code is used else
            pytorch implementation is used.

        """
        super().__init__(input_channels, hidden_channels, batch_first, zoneout, return_state_sequence)
        self.use_custom_cuda = False #use_custom_cuda

        if grad_clip:
            self.grad_clip_norm(enable=True, norm=grad_clip)
        else:
            self.grad_clip_norm(False)

        if dropout < 0 or dropout > 1:
            raise ValueError('GRU: dropout must be in [0.0, 1.0]')
        if zoneout < 0 or zoneout > 1:
            raise ValueError('GRU: zoneout must be in [0.0, 1.0]')

        self.dropout = dropout
        self.alpha = torch.tensor(0.9)
        self.binary = binary
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size

        self.weight_initialization_gain = weight_initialization_gain

        self.conv = layer.Conv2d(input_channels, 2*hidden_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=not bn)
        functional.set_step_mode(self.conv, step_mode='m')
        #self.recurrent_kernel = nn.Parameter(
        #    torch.empty(hidden_size, hidden_size * 3))
        if bn:
            self.bn = layer.BatchNorm2d(2*hidden_channels, momentum=0.01, eps=1e-3)
            functional.set_step_mode(self.bn, step_mode='m')
        #self.bias = nn.Parameter(torch.empty(hidden_size * 2))
        #self.recurrent_bias = nn.Parameter(torch.empty(hidden_size * 3))
        self.reset_parameters()

        self.dampening_factor = nn.Parameter(
            torch.Tensor([dampening_factor]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([pseudo_derivative_support]), requires_grad=False)

        # initialize thresholds according to the beta distribution with mean 'thr_mean'
        assert 0 < thr_mean < 1, f"thr_mean must be between 0 and 1, but {thr_mean} was given"
        beta = 3
        alpha = beta * thr_mean / (1 - thr_mean)
        distribution = torch.distributions.beta.Beta(alpha, beta)
        self.thr = nn.Parameter(distribution.sample(torch.Size([self.hidden_size])))

    def _permute(self, x):
        if self.batch_first:
            return x.permute(1, 0, 2, 3, 4)
        return x

    def reset_parameters(self):
        """Resets this layer's parameters to their initial values."""
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,
                                (layer.BatchNorm2d, layer.BatchNorm1d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def grad_clip_norm(self, enable=True, norm=2.0):
        self._enable_grad_clip = enable
        self._max_norm = nn.Parameter(torch.Tensor(
            [norm if enable else -1.0]), requires_grad=False)


    def forward(self, input, state=None, lengths=None):
        """
        Runs a forward pass of the Minimal EGRU layer.

        Arguments:
          input: Tensor, a batch of input sequences to pass through the GRU.
            Dimensions (seq_len, batch_size, input_size) if `batch_first` is
            `False`, otherwise (batch_size, seq_len, input_size).
          lengths: (optional) Tensor, list of sequence lengths for each batch
            element. Dimension (batch_size). This argument may be omitted if
            all batch elements are unpadded and have the same sequence length.

        Returns:
          output: Tensor, the output of the EGRU layer. Dimensions
            (seq_len, batch_size, hidden_size) if `batch_first` is `False` (default)
            or (batch_size, seq_len, hidden_size) if `batch_first` is `True`. Note
            that if `lengths` was specified, the `output` tensor will not be
            masked. It's the caller's responsibility to either not use the invalid
            entries or to mask them out before using them.
          h: the hidden state for all sequences. Dimensions
            (seq_len, batch_size, hidden_size).
          o: the output gate for all sequences (values: 0 or 1).
          trace: smoothed output values, can be beneficial for training.
        """
        if self.batch_first:
            input = input.permute(1, 0, 2, 3, 4)
        state_shape = [1, input.shape[1], self.hidden_size, int(np.floor((input.shape[3]+2*self.padding-(self.kernel_size-1)-1)/self.stride)+1), int(np.floor((input.shape[4]+2*self.padding-(self.kernel_size-1)-1)/self.stride)+1)]
        h0 = self._get_state(input, state, state_shape)

        # restrict thresholds to be between 0 and 1
        self.thr.data.clamp_(min=0.0, max=1.0)

        # run forward pass
        y, h, o, trace = self._impl(
            input, h0[0], self.thr, self._get_zoneout_mask(input))

        # prepare outputs
        output = self._permute(y[1:])
        h = self._permute(h[1:])
        o = self._permute(o[1:])
        trace = self._permute(trace[1:])
        if self.binary:
            return o
        else:
            return output #, (h, o, trace)

    def _impl(self, input, state, thr, zoneout_mask):

        return ConvMinEGRUScript(
                self.training,
                self.zoneout,
                self.dampening_factor,
                self.pseudo_derivative_support,
                input.contiguous(),
                state.contiguous(),
                self.conv,
                self.bn,
                thr.contiguous(),
                zoneout_mask.contiguous())