import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load both models
def load_models():
    model_cnn = tf.keras.models.load_model("mri/best_custom_cnn.h5")
    model_mobilenet = tf.keras.models.load_model("mri/best_mobilenet.h5")
    return model_cnn, model_mobilenet

model_cnn, model_mobilenet = load_models()

# Class names
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Title
st.title("🧠 Brain Tumor MRI Classification")
st.markdown("Upload an MRI image and choose a model to predict the type of brain tumor.")

# File uploader
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

# Model selection
selected_model = st.radio("Choose a model for prediction", ("Custom CNN", "MobileNetV2"))

# Process and predict
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    if selected_model == "Custom CNN":
        preds = model_cnn.predict(img_array)
    else:
        preds = model_mobilenet.predict(img_array)

    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)

    st.success(f"### 🧪 Predicted: `{predicted_class}` ({confidence*100:.2f}%)")

    # Show probabilities
    st.markdown("#### 🔬 Class Probabilities")
    for cls, prob in zip(class_names, preds[0]):
        st.write(f"- {cls}: **{prob:.4f}**")

# Footer
st.markdown("---")
st.markdown("✅ Model 1: Custom CNN  ✅ Model 2: MobileNetV2")
