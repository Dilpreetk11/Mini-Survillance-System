import streamlit as st
import cv2
import numpy as np
import zipfile
import os
from ultralytics import YOLO
from deepface import DeepFace

# Function to extract the YOLO model from a ZIP file
def extract_yolo_model(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Title of the app
st.title("Mini Surveillance System")

# Select use case
use_case = st.selectbox("Select Use Case", ["Face Detection", "Weapon Detection"])

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Display the image
    st.image(image, channels="BGR", caption="Uploaded Image")

    if use_case == "Face Detection":
        # Perform face detection using DeepFace
        try:
            analysis = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
            st.write("Face Detection Results:")
            st.write(analysis)
        except Exception as e:
            st.error(f"Error in face detection: {e}")

    elif use_case == "Weapon Detection":
        # Path to the ZIP file containing the YOLO model
        zip_file_path = 'yolov8n_custom_model.zip'  # Update this path if necessary
        extract_path = 'yolo_model'  # Directory to extract the model

        # Check if the model is already extracted
        if not os.path.exists(extract_path):
            extract_yolo_model(zip_file_path, extract_path)

        # Load the YOLO model
        model = YOLO(os.path.join(extract_path, 'yolov8.pt'))  # Adjust the model file name if necessary

        # Perform object detection
        results = model(image)

        # Display results
        st.write("Weapon Detection Results:")
        st.write(results)
