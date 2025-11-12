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
def load_model():
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
    net.eval()
    return net


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
    y_pred = torch.argmax(pred, dim=0)
    return pred, y_pred


# --------------------------
# MAIN STREAMLIT APP
# --------------------------
def main():
    st.set_page_config(page_title="🧠 Gait Classifier", layout="centered")
    st.title("🧠 Real-Time Gait Classifier (SNN)")
    st.caption("Automatically updates every few seconds")

    net = load_model()

    # Do inference
    pred, y_pred = run_inference(net)
    pred_np = pred.cpu().numpy()
    predicted_class = CLASSES[y_pred.item()]

    # Display
    if pred_np[y_pred.item()] < 0.1:
        st.subheader(f"Predicted Class: **None**")
    else:
        st.subheader(f"Predicted Class: **{predicted_class}**")
    #st.metric(label="Confidence", value=f"{confidence:.2f}")
    #st.bar_chart({"Class": CLASSES, "Confidence": pred_np})
    st.caption(f"Last updated: {time.strftime('%H:%M:%S')}")

    # Automatically rerun after REFRESH_RATE seconds
    time.sleep(REFRESH_RATE)
    st.rerun()  # ✅ works on all modern Streamlit versions


if __name__ == "__main__":
    main()
