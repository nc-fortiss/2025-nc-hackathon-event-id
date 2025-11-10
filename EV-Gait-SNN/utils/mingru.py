# https://arxiv.org/abs/2410.01201v1

import torch
import torch.nn.functional as F
from torch.nn import Linear, Identity, Module

from lavadl_egru import SpikeFunction

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# appendix B
# https://github.com/glassroom/heinsen_sequence

def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim = 1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim = 1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()

# appendix B.3

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

# log-space version of minGRU - B.3.1
# they enforce the hidden states to be positive

class minGRU(Module):
    def __init__(self, dim_in, dim_inner, proj_out = None, dim_out= None, spike=False):
        super().__init__()

        dim_inner = dim_inner #int(dim * expansion_factor)
        #proj_out = default(proj_out, expansion_factor != 1.)

        self.to_hidden_and_gate = Linear(dim_in, dim_inner * 2, bias = False)
        torch.nn.init.xavier_normal_(self.to_hidden_and_gate.weight, gain=1.0)
        if proj_out:
            self.to_out = Linear(dim_inner, dim_out, bias = False)
        else:
            self.to_out = Identity() # if proj_out else Identity()
        if proj_out:
            torch.nn.init.xavier_normal_(self.to_out.weight, gain=1.0)
        self.spike = spike
        if self.spike:
            self.threshold = torch.nn.Parameter(torch.Tensor([0.8]), requires_grad=False)
            self.dampening_factor = torch.nn.Parameter(torch.Tensor([0.7]), requires_grad=False)
            self.pseudo_derivative_support = torch.nn.Parameter(torch.Tensor([1.0]), requires_grad=False)


    def forward(self, x, prev_hidden = None, return_next_prev_hidden = False):
        seq_len = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim = -1)
        #print(hidden.shape)

        if seq_len == 1:
            # handle sequential

            hidden = g(hidden)
            gate = gate.sigmoid()
            out = torch.lerp(prev_hidden, hidden, gate) if exists(prev_hidden) else (hidden * gate)
        else:
            # parallel

            log_coeffs = -F.softplus(gate)

            log_z = -F.softplus(-gate)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h

            if exists(prev_hidden):
                log_values = torch.cat((prev_hidden.log(), log_values), dim = 1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            out = heinsen_associative_scan_log(log_coeffs, log_values)
            #print(out.shape)
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]

        #print(out.shape)
        out = self.to_out(out)
        #print(out.shape)

        if self.spike:
            out = SpikeFunction.apply(out - self.threshold, self.dampening_factor, self.pseudo_derivative_support)

        if not return_next_prev_hidden:
            return out

        return out, next_prev_hidden