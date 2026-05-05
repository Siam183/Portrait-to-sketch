import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

st.set_page_config(page_title="Portrait to Sketch", layout="centered")
st.title("🎨 Pix2pix Portrait Sketcher")

# Load the Generator model
@st.cache_resource
def load_gen_model():
    # Ensure this filename matches your file exactly
    model = load_model('g_model_100.h5', compile=False)
    return model

generator = load_gen_model()

uploaded_file = st.file_uploader("Upload a portrait photo", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Display Original
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Original Image", use_container_width=True)
    
    if st.button("Generate Sketch"):
        with st.spinner("Processing..."):
            # 2. Pre-process (Match your training size, e.g., 256x256)
            img = image.resize((256, 256))
            img_array = img_to_array(img)
            # Normalize to [-1, 1] (standard for Pix2pix)
            img_array = (img_array - 127.5) / 127.5
            img_array = np.expand_dims(img_array, axis=0)

            # 3. Predict
            prediction = generator.predict(img_array)

            # 4. Post-process (Back to [0, 255])
            prediction = (prediction[0] + 1) / 2.0
            
            st.image(prediction, caption="Your Sketch", use_container_width=True)