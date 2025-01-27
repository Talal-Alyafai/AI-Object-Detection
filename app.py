import streamlit as st
import cv2
import numpy as np
from utils.video_processing import load_yolov8_model, process_frame
from PIL import Image

# Load YOLOv8 model
model = load_yolov8_model()

st.title("🚀 Real-Time Object Detection with YOLOv8")
st.markdown("Detect 80+ objects in live video, images, or uploaded videos!")

# Sidebar settings
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
use_webcam = st.sidebar.checkbox("Use Webcam")
uploaded_file = st.sidebar.file_uploader("Or upload a video/image", type=["mp4", "avi", "jpg", "png"])

# Camera setup
if use_webcam:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open camera. Please check your camera connection.")
        use_webcam = False
    else:
        st.success("Camera is ready!")

# Display logic
if use_webcam or uploaded_file:
    FRAME_WINDOW = st.empty()
    stop_button = st.button("Stop")
    
    # For uploaded file
    if uploaded_file:
        if uploaded_file.type.startswith('image'):
            # Process image
            image = np.array(Image.open(uploaded_file))
            annotated_image = process_frame(image, model, confidence_threshold)
            st.image(annotated_image, channels="BGR")
        else:
            # Process video
            cap = cv2.VideoCapture(uploaded_file.name)
    
    # Process video frames
    while (use_webcam or (uploaded_file and not uploaded_file.type.startswith('image'))) and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("End of video or camera disconnected.")
            break
        
        # Run YOLOv8 inference
        annotated_frame = process_frame(frame, model, confidence_threshold)
        
        # Convert to RGB for Streamlit
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(rgb_frame)
    
    if 'cap' in locals():
        cap.release()

else:
    st.write("👈 Enable webcam or upload a file to start detection!")
