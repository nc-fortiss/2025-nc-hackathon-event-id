# 2025-nc-hackathon-event-id
This is the repository for the hackathon working on person identification from event streams.

## Get Started

Python requirements are listed in requirements.txt.

1. Clone repository
2. Step into repository: `cd 2025-nc-hackathon-event-id`
3. Create virtual environment: `python3 -m venv venv`
4. Activate virtual environment: `source venv/bin/activate`
5. Install requirements: `pip install -r requirements.txt`

Additional libraries.
6. Copy nc-libs from USB-Drive or provided laptop

DVS-Gait Dataset.
7. Copy DVS128-Gait-DATASET from USB-Drive or provided laptop

Optionally, DVSGesture Dataset
8. Copy DVSGesture Dataset from USB-Drive or provided laptop


## Overview

The repository contains a folder for event-based gait identification with CNNs (EV-Gait-IMG) taken from: https://github.com/zhangxiann/TPAMI_Gait_Identification

We provide reference code for event-based gesture recognition using SNNs (DVSGesture-SNN) and the initial structure for the target task of event-based gait identification with SNNs (EV-Gait-SNN).

Additionally, we provide links to a collection of relevant papers:


## DVS128-Gait DATASET

We use a [DVS128 Dynamic Vision Sensor](https://inivation.com/support/hardware/dvs128/) from iniVation operating at 128*128 pixel resolution.

We collect two dataset: **DVS128-Gait-Day** and **DVS128-Gait-Night**, which were collected under day and night lighting condition respectively.

For each lighting condition, we recruited  20 volunteers to contribute their data in two experiment sessions spanning over a few days. In each session, the participants were asked to repeat walking in front of the DVS128 sensor for 100 times.


## Run EV-GAIT-IMG

- Generate the image-like representation

  ```
  cd EV-Gait-IMG
  python make_hdf5.py
  ```

- Train EV-Gait-IMG from scratch:

  ```
  python train_gait_cnn.py --img_type counts_only_two_channel --epoch 50 --cuda 0 --batch_size 2
  ```

  
- Run EV-Gait-IMG model with the pretrained model:

  We provide  four options for `--img_type` to correctly test the corresponding  image-like representation

  ```
  python test_gait_cnn.py --img_type four_channel --model_name EV_Gait_IMG_counts_only_two_channel.pkl
  ```

## DVSGesture-SNN example

- Get dvs_gesture_bs2 data

- Run training script: 
  This script contains code to visualize event samples, setup an SNN and train it.
  
  ```
  python train_dvsgesture_snn.py
  ```

- Run evaluation script: 
  This script contains code to evaluate the trained SNN.
  
  ```
  python test_dvsgesture_snn.py
  ```

- Run lava inference script: 
  This script contains code to run trained models in lava. This serves as bit-accurate simulation of the neuromorphic processor Loihi 2.
  
  ```
  python loihi2_inference.py
  ```


## Your tasks: EV-Gait-SNN

- Generate the event representation

  ```
  cd EV-Gait-SNN
  python make_hdf5_events.py
  ```
  
- Create dataset class: gait_snn_dataset.py

- Create SNN model: models_snn.py

- Create training script and run training: train_gait_snn.py

- Create evaluation script and test: test_gait_snn.py


## Live Inference

Requirements:
    - Prophesee Event Camera
    - Get metavision.list file
    - Prophesee Metavision SDK installed: https://docs.prophesee.ai/4.6.2/installation/linux.html#chapter-installation-linux

- Setup live inference script: live_inference.py

- Record own data and use it to retrain models

- Test live person identification

## Loihi 2 Inference

- Loihi 2 benchmarking from dataset: loihi2_inference.py

- optionally: Live Loihi 2 inference via 10G Ethernet IO


## If needed: EV-Gait-3DGraph

Check out EV-Gait-3DGraph from: https://github.com/zhangxiann/TPAMI_Gait_Identification

And EvGNN from: https://github.com/cogsys-tudelft/evgnn
