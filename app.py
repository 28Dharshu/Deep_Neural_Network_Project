import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json

st.set_page_config(page_title="Waste Classification System", page_icon="♻️")

st.title("♻️ AI-Powered Waste Classification System")
st.write("Upload an image or use your camera to classify waste as **Recyclable (R)** or **Organic / Non-Recyclable (O)**.")

@st.cache_resource
def load_waste_model():
    with open("model1.json", "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights("model1.weights.h5")
    return model

model = load_waste_model()

def predict_image(pil_image):
    # Ensure 3 RGB channels and resize to match model input (128 x 128 x 3)
    img = pil_image.convert("RGB").resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    # Handle both single-output (sigmoid) and two-output (softmax) layers
    if prediction.shape[-1] == 1:
        score = float(prediction[0][0])
        if score >= 0.5:
            return "Recyclable Waste (R)", score
        else:
            return "Organic / Non-Recyclable Waste (O)", 1.0 - score
    else:
        class_idx = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][class_idx])
        label = "Recyclable Waste (R)" if class_idx == 1 else "Organic / Non-Recyclable Waste (O)"
        return label, confidence

# Input options: File upload or browser webcam
input_method = st.radio("Choose input method:", ("Upload Image", "Use Camera"))

image_source = None
if input_method == "Upload Image":
    image_source = st.file_uploader("Choose a waste image...", type=["jpg", "jpeg", "png"])
else:
    image_source = st.camera_input("Take a picture of the waste item")

if image_source is not None:
    image = Image.open(image_source)
    st.image(image, caption="Input Image", use_container_width=True)

    with st.spinner("Classifying..."):
        label, confidence = predict_image(image)

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")