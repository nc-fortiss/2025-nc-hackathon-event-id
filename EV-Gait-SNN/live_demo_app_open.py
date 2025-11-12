import os
import time
import numpy as np
import torch
import streamlit as st

from models_snn import PLIFSNN
from spikingjelly.activation_based import surrogate, neuron, layer
from live_inference import load_events

# --------------------------
# CONFIGURATION
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./trained_models/finetuned/network.pt"
EVENTS_PATH = "../../dataevents.npy"
CLASSES = ["Paul", "Simon"]
REFRESH_RATE = 3.0  # seconds between updates


# --------------------------
# LOAD MODEL (cached)
# --------------------------
@st.cache_resource
def load_model(full=True):
    inp_features = 2
    channels = 8
    feat_neur = 512
    classes = len(CLASSES)
    delay = False
    dropout = 0.2
    quantize = False

    net = PLIFSNN(inp_features, channels, feat_neur, classes,
                  delay, dropout, quantize, DEVICE).to(DEVICE)
    net.blocks[-2] = layer.Linear(feat_neur, classes, bias=True).to(DEVICE)
    net.blocks[-1] = neuron.ParametricLIFNode(
        surrogate_function=surrogate.ATan(),
        detach_reset=True, v_reset=0.0,
        decay_input=True, init_tau=2.0
    ).to(DEVICE)

    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    if not full:
        net.blocks = net.blocks[:-2]
    net.eval()
    return net

def record_new_class(net, class_name):
    print('new class recorded')
    outputs = []
    for _ in range(40):
        events, _ = load_events(
            EVENTS_PATH, down_input=5, events=None,
            shrinking_factor=1.0, sequence_length=1,
            bins_per_seq=5, sampling_time=20, binary=False
        )
        data = torch.from_numpy(events).type(torch.FloatTensor).to(DEVICE)
        with torch.no_grad():
            output, _ = net(data)
            output = output.mean(dim=-1)
            outputs.append(output)
        time.sleep(0.09)

    stacked_data = torch.stack(outputs)
    pred = stacked_data.mean(dim=0).squeeze(0).cpu()
    path_open_set = "../../open_set_classes.npy"

    if os.path.exists(path_open_set):
        open_set_classes = np.load(path_open_set, allow_pickle=True).tolist()
    else:
        open_set_classes = []
    print(pred.numpy().shape)
    open_set_classes += [{class_name: pred.numpy()}]
    np.save(path_open_set, open_set_classes)
    return pred


def classify_open_set(pred, path="../../open_set_classes.npy"):
    pred = pred.cpu()
    if os.path.exists(path):
        open_set_classes = np.load(path, allow_pickle=True).tolist()
    else:
        open_set_classes = []

    if not open_set_classes:
        return "None", -1  # No classes available

    distances = []
    for entry in open_set_classes:
        class_name, embedding = list(entry.items())[0]
        dist = np.linalg.norm(pred - embedding)
        distances.append((class_name, dist))

    # Find the closest class
    closest_class, min_distance = min(distances, key=lambda x: x[1])
    if min_distance >= 1.2:
        closest_class = "None"
    return closest_class, min_distance
    



# --------------------------
# INFERENCE FUNCTION
# --------------------------
def run_inference(net):
    outputs = []
    for _ in range(40):
        events, _ = load_events(
            EVENTS_PATH, down_input=5, events=None,
            shrinking_factor=1.0, sequence_length=1,
            bins_per_seq=5, sampling_time=20, binary=False
        )
        data = torch.from_numpy(events).type(torch.FloatTensor).to(DEVICE)
        with torch.no_grad():
            output, _ = net(data)
            output = output.mean(dim=-1)
            outputs.append(output)
        time.sleep(0.09)

    stacked_data = torch.stack(outputs)
    pred = stacked_data.mean(dim=0).squeeze(0)

    return pred


# --------------------------
# MAIN STREAMLIT APP
# --------------------------
def main():
    st.set_page_config(page_title="🧠 Gait Classifier", layout="centered")
    st.title("🧠 Real-Time Gait Classifier (SNN)")
    st.caption("Automatically updates every few seconds")

    net = load_model(full=False)

    # --------------------------
    # Record New Class UI
    # --------------------------
    st.subheader("🎙️ Record a New Class")

    new_class_name = st.text_input("Enter new class name:")
    if st.button("Start Recording"):
        if new_class_name.strip() == "":
            st.warning("⚠️ Please enter a class name before recording.")
        else:
            record_new_class(net, new_class_name)
            st.success(f"Recording saved for class: **{new_class_name}**")
            st.session_state["recording_class"] = new_class_name

    st.divider()  # visual separation

    # --------------------------
    # Inference Section
    # --------------------------
    pred = run_inference(net)
    predicted_class = classify_open_set(pred)

    st.subheader(f"Predicted Class: **{predicted_class}**")

    st.caption(f"Last updated: {time.strftime('%H:%M:%S')}")

    # Automatically rerun after REFRESH_RATE seconds
    time.sleep(REFRESH_RATE)
    st.rerun()

if __name__ == "__main__":
    main()
