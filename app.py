import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Portrait to Sketch AI",
    page_icon="🎨",
    layout="wide"
)

# --- Model Loading ---
@st.cache_resource
def load_gen_model():
    # Load the generator file you uploaded to LFS
    model = load_model('g_model_100.h5', compile=False)
    return model

# --- Image Processing Functions ---
def preprocess_image(image, target_size=(256, 256)):
    img = image.resize(target_size)
    img_array = np.array(img).astype(np.float32)
    # Standard Pix2pix normalization: scale [0, 255] to [-1, 1]
    img_array = (img_array - 127.5) / 127.5
    return np.expand_dims(img_array, axis=0)

def postprocess_output(prediction):
    # Rescale from [-1, 1] back to [0, 1]
    rescaled = (prediction[0] + 1) / 2.0
    return np.clip(rescaled, 0, 1)

# --- Main UI Layout ---
st.title("🎨 Portrait to Sketch AI")
st.markdown("---")

generator = load_gen_model()

# Sidebar instructions
with st.sidebar:
    st.header("Instructions")
    st.write("1. Upload a clear portrait.")
    st.write("2. Click 'Generate' to see the AI sketch.")
    st.divider()
    st.info("Note: This model works best on images with simple backgrounds.")

uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    input_image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("Original Portrait")
        st.image(input_image, use_container_width=True)
    
    if st.button("✨ Generate Sketch", use_container_width=True):
        with st.spinner("AI is drawing..."):
            # Process
            processed_input = preprocess_image(input_image)
            prediction = generator.predict(processed_input)
            final_sketch = postprocess_output(prediction)
            
            with col2:
                st.subheader("AI Generated Sketch")
                st.image(final_sketch, use_container_width=True)
                
                # Create download buffer
                result_img = Image.fromarray((final_sketch * 255).astype(np.uint8))
                buf = io.BytesIO()
                result_img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Result",
                    data=byte_im,
                    file_name="sketch_output.png",
                    mime="image/png"
                )
else:
    st.info("Upload an image to get started.")