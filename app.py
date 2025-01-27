import streamlit as st
import cv2
import numpy as np
from utils.video_processing import load_yolov8_model, process_frame
from PIL import Image

# Load YOLOv8 model once
model = load_yolov8_model()

st.title("Real-Time Object Detection with YOLOv8 - By Talal Al-Yafai")
st.markdown("Detect 80+ objects in live video, images, or uploaded videos!")


confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
use_webcam = st.sidebar.checkbox("Use Webcam")
uploaded_file = st.sidebar.file_uploader("Or upload a video/image", 
                                       type=["mp4", "avi", "jpg", "png"])

if use_webcam or uploaded_file:
    FRAME_WINDOW = st.empty()
    stop_button = st.button("Stop")
    
    if use_webcam:
        cap = cv2.VideoCapture(0)
    
    elif uploaded_file:
        file_bytes = uploaded_file.read()
        if uploaded_file.type.startswith('image'):
            # Process image
            image = np.array(Image.open(uploaded_file))
            annotated_image = process_frame(image, model, confidence_threshold)
            st.image(annotated_image, channels="BGR")
        else:
            cap = cv2.VideoCapture(uploaded_file.name)
    
    while (use_webcam or (uploaded_file and not uploaded_file.type.startswith('image'))) and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame = process_frame(frame, model, confidence_threshold)
        
        # Convert to RGB for Streamlit
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(rgb_frame)
    
    if 'cap' in locals():
        cap.release()

else:
    st.write("👈 Enable webcam or upload a file to start detection!")