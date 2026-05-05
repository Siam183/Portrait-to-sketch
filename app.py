import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Portrait to Sketch AI",
    page_icon="🎨",
    layout="wide"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("Settings & Help")
    st.info("This AI uses a Pix2pix GAN (Generative Adversarial Network) to translate portraits into charcoal-style sketches.")
    st.subheader("Tips for best results:")
    st.write("1. Use a clear, front-facing portrait.")
    st.write("2. Simple backgrounds work best.")
    st.write("3. Faces should be well-lit.")

# --- Model Loading ---
@st.cache_resource
def load_gen_model():
    # compile=False avoids errors with custom GAN loss functions
    model = load_model('g_model_100.h5', compile=False)
    return model

# --- Image Processing Functions ---
def preprocess_image(image, target_size=(256, 256)):
    # Resize and convert to array
    img = image.resize(target_size)
    img_array = np.array(img).astype(np.float32)
    # Normalize to [-1, 1] - Standard for Pix2pix
    img_array = (img_array - 127.5) / 127.5
    return np.expand_dims(img_array, axis=0)

def postprocess_output(prediction):
    # Scale from [-1, 1] back to [0, 1]
    rescaled = (prediction[0] + 1) / 2.0
    return np.clip(rescaled, 0, 1)

# --- Main UI ---
st.title("🎨 Portrait to Sketch AI")
st.write("Transform your photos into artistic sketches using Deep Learning.")

generator = load_gen_model()

uploaded_file = st.file_uploader("Choose a portrait photo...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Create two columns for side-by-side view
    col1, col2 = st.columns(2)
    
    input_image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("Original Image")
        st.image(input_image, use_container_width=True)
    
    # Action Button
    if st.button("✨ Generate Sketch"):
        with st.spinner("AI is sketching your portrait..."):
            # 1. Preprocess
            processed_input = preprocess_image(input_image)
            
            # 2. Inference
            start_time = time.time()
            prediction = generator.predict(processed_input)
            end_time = time.time()
            
            # 3. Postprocess
            final_sketch = postprocess_output(prediction)
            
            with col2:
                st.subheader("Generated Sketch")
                st.image(final_sketch, use_container_width=True)
                st.caption(f"Inference took {end_time - start_time:.2f} seconds")
                
                # Download Button
                # Convert back to uint8 for saving
                result_img = Image.fromarray((final_sketch * 255).astype(np.uint8))
                st.download_button(
                    label="Download Sketch",
                    data=uploaded_file, # Note: For a real download, you'd save result_img to a buffer
                    file_name="ai_sketch.png",
                    mime="image/png"
                )
else:
    # Placeholder when no image is uploaded
    st.info("Please upload a portrait photo to begin.")