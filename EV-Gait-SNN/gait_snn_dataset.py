# Copyright (C) 2025 fortiss GmbH
# SPDX-License-Identifier:  BSD-3-Clause


class DVSGaitDataset(Dataset):
    """DVS Gait dataset class

    Parameters
    ----------
    path : str, optional
        path of dataset root, by default '/home/datasets/dvs_gesture_bs2'
    train : bool, optional
        train/test flag, by default True
    sampling_time : int, optional
        sampling time of event data, by default 1
    sample_length : int, optional
        length of sample data, by default 300
    transform : None or lambda or fx-ptr, optional
        transformation method. None means no transform. By default None.
    random_shift: bool, optional
        shift input sequence randomly in time. By default True.
    data_format: str, optional
        data format of the input data, either 'bs2' or 'npy'. By default 'bs2'.
    ds_factor: int, optional
        factor to downsample event input. By default 1.
    """