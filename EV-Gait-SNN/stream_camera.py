import time
import numpy as np
import sys
import metavision_core.event_io
from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import RawReader, initiate_device
from metavision_hal import I_ROI
import os

sys.path.append("/usr/lib/python3/dist-packages")

import metavision_hal
print(dir(metavision_hal))

from metavision_hal import DeviceDiscovery, I_ErcModule
print(DeviceDiscovery.list())

from metavision_sdk_core import RoiFilterAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm


def write_events_npy(path, events) :
        t0 = time.time()

        np.save(path + "events.npy", events)
        if time.time() - t0 > 0.01 : #debugging
            print("saved in ", time.time() - t0)
            print("number of events : ", len(events['x']))
            path_save = path + "events.npy"
            print("saved to: ", path_save)

# to read events from recorded file or live camera
device = initiate_device(path="")
print("\n\nOutput supported packages by device:")
print(dir(device))
if device.get_i_ll_biases():  # set bias
    # prophesee pcb
    # device.get_i_ll_biases().set("bias_fo", 50)
    # device.get_i_ll_biases().set("bias_diff_off", 35)
    # device.get_i_ll_biases().set("bias_diff_on", 125)
    # prophesee evk 4
    device.get_i_ll_biases().set("bias_diff", 12)
    device.get_i_ll_biases().set("bias_diff_off", 50)
    device.get_i_ll_biases().set("bias_diff_on", 50)
    print(device.get_i_ll_biases().get_all_biases())
if device.get_i_roi():  # define roi
    i_roi = device.get_i_roi()
    roi_window = I_ROI.Window(320, 40, 680, 720) #roi_window = I_ROI.Window(320, 40, 960, 680) #roi_window = I_ROI.Window(495, 40, 345, 680)
    i_roi.set_window(roi_window)
    i_roi.enable(True)

    # i_roi.set_window(roi= I_ROI.Window(320, 40, 1280, 680))
    #i_roi.enable(True)
    # roi = DeviceRoi(320, 40, 640, 640)
    # print(roi.to_string())
    # device.get_i_roi().set_ROI(roi, True)

try:
    erc = device.get_i_erc_module()
    erc.set_cd_event_rate(400000) # set max event rate - 200_000 in own recording
    erc.enable(True)
    print("### ERC configured successfully.")
except AttributeError:
    print("!!! ERC module not available on this device.")


# if device.get_i_erc(): # set max event rate
#     device.get_i_erc().set_cd_event_rate(400000)
#     device.get_i_erc().enable(True)


#if device.get_i_noisefilter_module():  # define noisefilter stc
    #device.get_i_noisefilter_module().enable_stc(threshold=1000)
mv_iterator = EventsIterator.from_device(device=device, delta_t=100000)
global_counter = 0  # This will track how many events we processed
global_max_t = 0  # This will track the highest timestamp we processed

# !!! torch.cuda.is_available()

# Process events
for evs in mv_iterator:
    if evs.size != 0:
        events = evs.copy()
        events['x'] = events['x'] - 320
        events['y'] = events['y'] - 40
        min_t = evs['t'][0]   # Get the timestamp of the first event of this callback
        max_t = evs['t'][-1]  # Get the timestamp of the last event of this callback
        global_max_t = max_t  # Events are ordered by timestamp, so the current last event has the highest timestamp

        counter = evs.size  # Local counter
        global_counter += counter  # Increase global counter

        # TODO: replace with path to GUI folder - depending on where the script is running from
        write_events_npy("/home/nc-demonstrator/Documents/2025-nc-hackathon-event-id/data", events)

# Print the global statistics
duration_seconds = global_max_t / 1.0e6
print(f"There were {global_counter} events in total.")
print(f"The total duration was {duration_seconds:.2f} seconds.")
if duration_seconds >= 1:  # No need to print this statistics if the total duration was too short
    print(f"There were {global_counter / duration_seconds :.2f} events per second on average.")
