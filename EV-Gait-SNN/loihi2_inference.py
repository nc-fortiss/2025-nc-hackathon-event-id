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

    # loihi2 inference



        




