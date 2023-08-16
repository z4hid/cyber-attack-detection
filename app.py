import os
import time
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# Configuration settings
st.set_page_config(
    page_title="Cyber Attack Detection using A.I.",
    page_icon=":shield:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Apply custom styles for a polished appearance
st.markdown(
    """
    <style>
        body {
            color: #14171A;
            background-color: #F5F8FA;
        }
        .stButton > button {
            background-color: #1DA1F2;
            color: white;
            font-weight: bold;
            margin: auto;
            display: block;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set environment variable to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Constants
MODEL_PATH = "models/model.h5"
CLASS_NAMES = [
    'AgentTesla', 'Amadey', 'AsyncRAT', 'Emotet', 'FormBook', 'GuLoader',
    'Loki', 'Mirai', 'NetSupportRAT', 'NjRat', 'Non-Malicious', 'QBot',
    'RedLineStealer', 'Remcos', 'Vidar'
]


def load_ml_model(path):
    """Load machine learning model from a given path."""
    return load_model(path)


def classify_uploaded_image(img):
    """Classify the uploaded image using a pre-trained model."""
    img = Image.open(img).convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_class_name = CLASS_NAMES[predicted_class]
    predicted_percentage = predictions[0][predicted_class] * 100
    return predicted_class_name, predicted_percentage


model = load_ml_model(MODEL_PATH)

st.title("Cyber Attack Detection using A.I. 🛡️")
st.write("Utilize our state-of-the-art AI model to identify potential malware from image representations.")
uploaded_file = st.file_uploader(label="Upload File")

if uploaded_file:
    file_details = {
        "FileName": uploaded_file.name,
        "FileType": uploaded_file.type,
        "FileSize": uploaded_file.size / 1024  # Convert size to KB
    }
    st.write(
        "Uploaded File Details:", file_details["FileName"],
        f"- {file_details['FileSize']:.2f}KB"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Scan Now", help="Click to start malware detection"):
            with st.spinner("Scanning..."):
                progress_bar = st.progress(0)
                predicted_class_name = None

                for percent_complete in range(0, 101, 10):
                    time.sleep(0.1)
                    progress_bar.progress(percent_complete)
                    if percent_complete == 50:
                        predicted_class_name, predicted_percentage = classify_uploaded_image(uploaded_file)

            if predicted_class_name == "Non-Malicious":
                st.success("Scan Result: Your File is Safe! 🟢")
                st.markdown("<h2 style='color:green; font-size:24px;'>Non-Malicious</h2>", unsafe_allow_html=True)
            else:
                st.error(f"Scan Result: Threat Detected! 🔴")
                st.markdown(f"<h2 style='color:red; font-size:24px;'>{predicted_class_name}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='font-size:24px;'>Confidence: {predicted_percentage:.2f}%</h2>", unsafe_allow_html=True)
