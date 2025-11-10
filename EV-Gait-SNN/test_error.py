
import torch
device = torch.device('cpu')
import os
import sys
lavadlPath = os.path.join('C:/hackathon/nc-libs/lava-dl/src')
sys.path.insert(0, lavadlPath)
import lava.lib.dl.slayer as slayer

error = slayer.loss.SpikeRate(
        true_rate=0.5, false_rate=0.03, reduction='sum').to(device)

output = torch.zeros((1,20,100))
target = torch.Tensor([1])

loss = error(output, target)