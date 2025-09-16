import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Load model
model = tf.keras.models.load_model("inception_flower_model.h5")

# Class labels (match your dataset folders)
class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

st.title("🌸 Flower Classification App (InceptionV3)")
st.write("Upload a flower image and the model will predict its class.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    img = image.load_img(uploaded_file, target_size=(299, 299))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Prediction
    preds = model.predict(img_array)
    predicted_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
