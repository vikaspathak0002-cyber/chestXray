import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺",
    layout="centered",
)


# -----------------------
# Custom CSS (Pro UI)
# -----------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #e3f2fd, #ffffff);
}

.uploadedImage {
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.result-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.15);
}

.footer {
    margin-top: 50px;
    text-align: center;
    color: gray;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_model_cached():
    model_path = "pneumonia_model.keras"
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model_cached()


# -----------------------
# Header Section
# -----------------------
st.markdown("<h1 style='text-align:center;'>🩺 Chest X-Ray Pneumonia Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a chest X-ray below to detect Pneumonia using AI.</p>", unsafe_allow_html=True)

st.write("")
st.write("")


# -----------------------
# File Upload
# -----------------------
uploaded_file = st.file_uploader("📤 Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    # Show image
    st.markdown("### 🖼 Uploaded X-ray")
    image_data = Image.open(uploaded_file)
    st.image(image_data, width=350, caption="Uploaded Image", use_column_width=False)

    # Convert to model input
    img = image_data.resize((224, 224))
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    # Predict Button
    if st.button("🔍 Analyze X-ray"):
        with st.spinner("Analyzing Image... Please wait ⏳"):
            pred = model.predict(img_arr)[0][0]

        st.write("")
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)

        # Result Display
        if pred > 0.5:
            st.error("### 🔴 Pneumonia Detected")
            st.write(f"**Confidence:** `{pred:.4f}`")
        else:
            st.success("### 🟢 Normal Lungs")
            st.write(f"**Confidence:** `{1 - pred:.4f}`")

        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------
# Footer
# -----------------------
st.markdown("""
<div class='footer'>
Made with ❤️ using Streamlit & TensorFlow
</div>
""", unsafe_allow_html=True)
