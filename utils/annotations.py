import cv2
import numpy as np
from time import time

class Annotator:
    """
    Custom annotation utilities for YOLOv8 object detection.
    Handles bounding boxes, labels, FPS counters, and color schemes.
    """
    
    def __init__(self):
        self.colors = np.random.randint(0, 255, (80, 3)).astype('uint8')  # 80 colors for COCO classes
        self.prev_frame_time = 0  # For FPS calculation
        
    def draw_boxes(self, frame, results, confidence_threshold):
        """Draw bounding boxes and labels on a frame."""
        for box in results.boxes:
            conf = box.conf.item()
            if conf < confidence_threshold:
                continue
            
            # Get box coordinates and class ID
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls.item())
            
            # Choose color based on class
            color = tuple(map(int, self.colors[cls_id]))
            
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            
            label = f"{results.names[cls_id]} {conf:.2f}"
            self._draw_label(frame, label, (x1, y1 - 10), color)
            
        return frame
    
    def _draw_label(self, frame, text, position, color):
        """Helper to draw text with background."""
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        
        cv2.rectangle(
            frame,
            (position[0], position[1] - text_height - 2),
            (position[0] + text_width, position[1] + 2),
            color,
            -1
        )
        
       
        cv2.putText(
            frame,
            text,
            (position[0], position[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
    
    def draw_fps(self, frame):
        """Calculate and draw FPS counter."""
        curr_time = time()
        fps = 1 / (curr_time - self.prev_frame_time)
        self.prev_frame_time = curr_time
        
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
            
        )
        return frame
    