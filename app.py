import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("road_sign_model.keras") # Load the .keras model

# Define the class names (replace with actual class names from GTSRB)
class_names = [
    "Speed limit 20km/h", "Speed limit 30km/h", "Speed limit 50km/h",
    "Speed limit 60km/h", "Speed limit 70km/h", "Speed limit 80km/h",
    "End of speed limit 80km/h", "Speed limit 100km/h", "Speed limit 120km/h",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield",
    "Stop", "No vehicles", "Vehicles over 3.5 metric tons prohibited",
    "No entry", "General caution", "Dangerous curve to the left",
    "Dangerous curve to the right", "Double curve", "Bumpy road",
    "Slippery road", "Narrow road ahead", "Traffic signals", "Pedestrians",
    "Children crossing", "Bicycles crossing", "Beware of ice/snow",
    "Wild animals crossing", "End of all speed and passing limits",
    "Turn right ahead", "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory",
    "End of no passing", "End of no passing by vehicles over 3.5 metric tons"
]


st.title("🚦 Road Sign Recognition System (GTSRB)")

st.write("Upload an image of a road sign to get a prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "ppm"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB) # Convert to RGB

    # Display the uploaded image
    st.image(opencv_image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for the model
    img = cv2.resize(opencv_image, (32, 32))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Make a prediction
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index]

    st.write(f"Prediction: **{predicted_class_name}**")
    st.write(f"Confidence: **{confidence:.2f}**")
