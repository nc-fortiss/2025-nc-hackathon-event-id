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

lavadlPath = os.path.join(rootPath, '../nc-libs/lava-dl/src')
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
    trained_folder = 'trained_models/Trained_plif_snn'
    os.makedirs(trained_folder, exist_ok=True)