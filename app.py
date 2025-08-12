import streamlit as st
from ultralytics import YOLO
import cv2
from deepface import DeepFace
import numpy as np
import tempfile
import time

# Function to load YOLOv8 model for weapon detection
@st.cache_resource
def load_yolo_model():
    # Load a pre-trained YOLOv8 model for object detection
    # 'yolov8n.pt' is the nano model, good for speed
    return YOLO('yolov8n.pt')

def detect_weapons(frame, model):
    """Detects weapons in a frame using YOLOv8."""
    # Run YOLOv8 inference on the frame
    results = model(frame)
    annotated_frame = results[0].plot()
    return annotated_frame

def detect_faces(frame):
    """Detects and analyzes faces in a frame using DeepFace."""
    try:
        # Use DeepFace to analyze faces for attributes like emotion and age
        # Actions can be 'emotion', 'age', 'gender', 'race'
        detections = DeepFace.analyze(
            frame, 
            actions=['emotion', 'age', 'gender'], 
            enforce_detection=False
        )
        
        # Draw bounding boxes and text on the frame
        for detection in detections:
            box = detection['region']
            x, y, w, h = box['x'], box['y'], box['w'], box['h']
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add text
            emotion = detection['dominant_emotion']
            age = detection['age']
            gender = detection['gender']
            text = f"Emotion: {emotion}, Age: {age}, Gender: {gender}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    except Exception as e:
        st.error(f"DeepFace analysis failed: {e}")
        pass
    
    return frame

def main():
    st.title("Mini Surveillance System")
    st.write("Upload a video or use your webcam for real-time surveillance.")

    yolo_model = load_yolo_model()

    mode = st.radio("Select mode", ["Upload Video", "Webcam"])

    if mode == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
        if uploaded_file:
            st.video(uploaded_file)
            
            # Save uploaded video to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            
            stframe = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Perform detections
                weapon_frame = detect_weapons(frame, yolo_model)
                final_frame = detect_faces(weapon_frame)
                
                stframe.image(final_frame, channels="BGR")
            
            cap.release()
            tfile.close()

    elif mode == "Webcam":
        st.warning("Webcam support is more complex to deploy. For local testing, you can use `cv2.VideoCapture(0)`.")
        
        # This section is for local testing with a webcam.
        # It won't work on public cloud deployments without specific configurations.
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Error: Could not open webcam.")
                return

            stframe = st.empty()
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Error: Failed to capture frame.")
                    break
                
                weapon_frame = detect_weapons(frame, yolo_model)
                final_frame = detect_faces(weapon_frame)
                
                stframe.image(final_frame, channels="BGR")

        except Exception as e:
            st.error(f"Webcam error: {e}")
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
