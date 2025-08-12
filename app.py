import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

# Function to load YOLOv8 model for weapon detection
@st.cache_resource
def load_yolo_model():
    # Load a pre-trained YOLOv8 model for object detection
    # 'yolov8n.pt' is the nano model, good for speed
    return YOLO('yolov8n.pt')

def detect_weapons(frame, model):
    """Detects weapons in a frame using YOLOv8."""
    results = model(frame)
    annotated_frame = results[0].plot()
    return annotated_frame

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
                
                # Perform weapon detection
                final_frame = detect_weapons(frame, yolo_model)
                
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
                
                final_frame = detect_weapons(frame, yolo_model)
                
                stframe.image(final_frame, channels="BGR")

        except Exception as e:
            st.error(f"Webcam error: {e}")
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()

if __name__ == '__main__':
    main()
