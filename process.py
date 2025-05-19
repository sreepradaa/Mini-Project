import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

# Initialize YOLOv8 model and OCR
model = YOLO('yolov8n.pt')
reader = easyocr.Reader(['en'])

def process_input(input_source, is_video=False):
    cap = None
    if is_video:
        cap = cv2.VideoCapture(input_source)
    else:
        frame = cv2.imread(input_source)
    
    violations = []
    output_frames = []
    
    while is_video and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, new_violations = process_frame(frame)
        violations.extend(new_violations)
        
        # Convert frame to base64 for web display
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        output_frames.append(frame_base64)
        
        if len(output_frames) > 10:  # Limit frames for live feed
            break
    
    if not is_video:
        processed_frame, violations = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        output_frames.append(base64.b64encode(buffer).decode('utf-8'))
    
    if cap:
        cap.release()
    
    return {
        'frames': output_frames,
        'violations': violations
    }

def process_frame(frame):
    results = model(frame)
    violations = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cls = int(box.cls)
            label = model.names[cls]
            
            color = (0, 255, 0)  # Green for detected objects
            violation = None
            
            if label == 'motorcycle' and conf > 0.5:
                # Check for helmet
                helmet_detected = check_helmet(frame, x1, y1, x2, y2)
                if not helmet_detected:
                    color = (0, 0, 255)  # Red for violation
                    license_plate = perform_anpr(frame, x1, y1, x2, y2)
                    violation = f"No helmet detected - Plate: {license_plate}"
                    save_violation(frame, x1, y1, x2, y2, violation)
            
            elif label == 'car' and conf > 0.5:
                # Check for seatbelt (simplified)
                seatbelt_detected = check_seatbelt(frame, x1, y1, x2, y2)
                if not seatbelt_detected:
                    color = (0, 0, 255)  # Red for violation
                    license_plate = perform_anpr(frame, x1, y1, x2, y2)
                    violation = f"No seatbelt detected - Plate: {license_plate}"
                    save_violation(frame, x1, y1, x2, y2, violation)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            if violation:
                violations.append({
                    'type': violation,
                    'confidence': conf,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
    
    return frame, violations

def check_helmet(frame, x1, y1, x2, y2):
    # Simplified helmet detection: Check for helmet-like object in upper region
    head_region = frame[y1:y1+int((y2-y1)*0.3), x1:x2]
    if head_region.size == 0:
        return False
    
    results = model(head_region)
    for r in results:
        for box in r.boxes:
            if model.names[int(box.cls)] == 'helmet' and float(box.conf) > 0.4:
                return True
    return False

def check_seatbelt(frame, x1, y1, x2, y2):
    # Simplified seatbelt detection: Placeholder for actual implementation
    return True  # Assume seatbelt is present for demo

def perform_anpr(frame, x1, y1, x2, y2):
    plate_region = frame[y1:y2, x1:x2]
    if plate_region.size == 0:
        return "Unknown"
    
    results = reader.readtext(plate_region)
    for (bbox, text, prob) in results:
        if prob > 0.5:
            return text
    return "Unknown"

def save_violation(frame, x1, y1, x2, y2, violation):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'violations/violation_{timestamp}_{violation.replace(" ", "_")}.jpg'
    cv2.imwrite(filename, frame)