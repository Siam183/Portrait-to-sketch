import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import time

# --- AESTHETIC CONFIGURATION ---
st.set_page_config(
    page_title="Portrait to Sketch Pro",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
    <style>
    /* Dark professional theme */
    .stApp {
        background: #0E1117;
        color: #E0E0E0;
    }
    /* Rounded corners for images */
    img {
        border-radius: 15px;
        border: 1px solid #3d4b5c;
    }
    /* Button styling */
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- BACKEND LOGIC ---
@st.cache_resource
def load_generator():
    """Load model with resource caching to prevent OOM errors."""
    try:
        # Load weights without compiling to save memory
        model = load_model('g_model_100.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_image(img_raw):
    """Refined preprocessing and inference pipeline."""
    # Preprocessing
    img = img_raw.resize((256, 256))
    img_array = np.array(img).astype(np.float32)
    # Using the math that worked best for your model
    img_array = (img_array - 127.5) / 127.5 
    img_array = np.expand_dims(img_array, axis=0)
    
    # Inference
    gen = load_generator()
    if gen:
        pred = gen.predict(img_array)
        # Postprocessing: Rescale to [0, 1]
        rescaled = (pred[0] + 1) / 2.0
        return np.clip(rescaled, 0, 1)
    return None

# --- UI INTERFACE ---
with st.sidebar:
    st.title("AI Artist Studio")
    st.markdown("---")
    st.info("**Instructions:** \n1. Upload a portrait. \n2. Ensure the face is clear. \n3. Click 'Generate' to see the magic.")
    st.divider()
    st.caption("v1.2 | Powered by Pix2pix GAN")

# Main Content
st.title("✨ Portrait to Sketch Pro")
st.write("Professional-grade AI portrait sketching powered by Deep Learning.")

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    input_img = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📷 Original Portrait")
        st.image(input_img, use_container_width=True)
        
    if st.button("✨ GENERATE ARTWORK", use_container_width=True):
        with st.spinner("Our AI is sketching your features..."):
            start = time.time()
            sketch = process_image(input_img)
            duration = time.time() - start
            
            if sketch is not None:
                with col2:
                    st.subheader("🎨 AI Sketch Output")
                    st.image(sketch, use_container_width=True)
                    st.success(f"Generated in {duration:.2f} seconds")
                    
                    # Preparation for download
                    result_pil = Image.fromarray((sketch * 255).astype(np.uint8))
                    buf = io.BytesIO()
                    result_pil.save(buf, format="PNG")
                    st.download_button(
                        label="⬇️ DOWNLOAD SKETCH",
                        data=buf.getvalue(),
                        file_name="ai_portrait_sketch.png",
                        mime="image/png",
                        use_container_width=True
                    )
else:
    st.markdown("""
        <div style="text-align: center; padding: 50px; border: 2px dashed #3d4b5c; border-radius: 20px;">
            <h3>Drop your photo here to begin</h3>
            <p>Faces with high contrast and simple backgrounds yield the most artistic results.</p>
        </div>
    """, unsafe_allow_html=True)