import cv2
import torch
from ultralytics import YOLO

def load_yolov8_model(model_path="models/yolov8n.pt"):
    model = YOLO(model_path) 
    return model

def process_frame(frame, model, confidence_threshold=0.5):
    results = model(frame, verbose=False)[0]  # Run inference
    annotated_frame = frame.copy()
    
   
    for box in results.boxes:
        conf = box.conf.item()
        if conf < confidence_threshold:
            continue
        cls_id = int(box.cls.item())
        label = f"{results.names[cls_id]} {conf:.2f}"
        
    
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated_frame