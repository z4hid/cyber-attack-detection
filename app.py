import os
import time
import math
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# Constants
MODEL_PATH = "models/model.h5"
CLASS_NAMES = [
    'AgentTesla', 'Amadey', 'AsyncRAT', 'Emotet', 'FormBook', 'GuLoader',
    'Loki', 'Mirai', 'NetSupportRAT', 'NjRat', 'Non-Malicious', 'QBot',
    'RedLineStealer', 'Remcos', 'Vidar'
]


def get_binary_data(file_obj):
    """Extract byte values from a binary executable and store them in a list."""
    if isinstance(file_obj, str):
        with open(file_obj, 'rb') as f:
            return list(f.read())
    return list(file_obj.getvalue())


def rgb_from_binary(filename, width=None):
    """Convert binary data to an RGB image."""
    index = 0
    rgb_data = []
    binary_data = get_binary_data(filename)

    while index + 2 < len(binary_data):
        rgb_data.append((binary_data[index], binary_data[index + 1], binary_data[index + 2]))
        index += 3

    size = get_image_size(len(rgb_data), width)
    return create_image(rgb_data, size, 'RGB')


def create_image(data, size, image_type):
    """Create a PIL Image from the provided data."""
    image = Image.new(image_type, size)
    image.putdata(data)
    return image


def get_image_size(data_length, width=None):
    """Calculate the size (width and height) for the image."""
    if width:
        return width, math.ceil(data_length / width)

    size_to_width = {
        range(0, 10240): 32,
        range(10240, 10240*3): 64,
        range(10240*3, 10240*6): 128,
        range(10240*6, 10240*10): 256,
        range(10240*10, 10240*20): 384,
        range(10240*20, 10240*50): 512,
        range(10240*50, 10240*100): 768
    }

    for size_range, w in size_to_width.items():
        if data_length in size_range:
            width = w
            break
    else:
        width = 1024

    return width, math.ceil(data_length / width)


def classify_uploaded_image(img):
    """Classify the uploaded image using the pre-trained model."""
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_class_name = CLASS_NAMES[predicted_class]
    predicted_percentage = predictions[0][predicted_class] * 100
    return predicted_class_name, predicted_percentage


# Streamlit Configuration
st.set_page_config(
    page_title="Cyber Attack Detection using A.I.",
    page_icon=":satellite:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Streamlit Styling
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

# Force TensorFlow to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load the pre-trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Streamlit UI
st.title("Cyber Attack Detection using A.I.")
st.write("Our Cyber Attack Detection platform is a cutting-edge tool that uses advanced Artificial Intelligence to identify potential cyber threats. The platform is trained on 15 of the most prevalent malware classes using the latest samples and statistics from MalwareBazaar.")

uploaded_file = st.file_uploader(label="Upload Binary/Portable Executable File")

if uploaded_file:
    file_details = {
        "FileName": uploaded_file.name,
        "FileType": uploaded_file.type,
        "FileSize": uploaded_file.size / 1024
    }
    st.write("Uploaded File Details:", file_details["FileName"], f"- {file_details['FileSize']:.2f}KB")

    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        if st.button("Scan Now", help="Click to start malware detection"):
            with st.spinner("Scanning..."):
                progress_bar = st.progress(0)
                img = rgb_from_binary(uploaded_file)
                for percent_complete in range(0, 101, 10):
                    time.sleep(0.1)
                    progress_bar.progress(percent_complete)
                    if percent_complete == 50:
                        predicted_class_name, predicted_percentage = classify_uploaded_image(img)

            if predicted_class_name == "Non-Malicious":
                st.success("🟢 Your File is Safe")
                st.markdown("<h2 style='color:green; font-size:24px;'>Non-Malicious</h2>", unsafe_allow_html=True)
            else:
                st.error(f"❌ Threat Detected ❗")
                st.markdown(f"<h2 style='color:red; font-size:24px;'>{predicted_class_name}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='font-size:24px;'>Confidence: {predicted_percentage:.2f}%</h2>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #14171A;
            color: #1DA1F2; 
            padding: 10px 0;
            text-align: center;
            font-family: Arial, Helvetica, sans-serif; 
            border-top: 1px solid #E1E8ED;  
        }
        .footer a {
            color: #1DA1F2;  
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
    <div class="footer">
        Developed with ❤️ by <a href="https://github.com/zahidhasanshuvo" target="_blank">Zahid</a>
    </div>
    """,
    unsafe_allow_html=True,
)
