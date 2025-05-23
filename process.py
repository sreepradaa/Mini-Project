import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import pytesseract

# Initialize YOLOv8 model and OCR
model = YOLO('yolov8n.pt')
reader = easyocr.Reader(['en'], gpu=False)

def process_input(input_source, is_video=False):
    cap = None
    if is_video:
        cap = cv2.VideoCapture(input_source)
    else:
        frame = cv2.imread(input_source)
    
    violations = []
    output_frames = []
    
    frame_count = 0
    while True:
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 3 != 0:  # Process every 3rd frame to reduce CPU load
                continue
        else:
            frame = cv2.imread(input_source)

        results = model(frame, imgsz=640, conf=0.3)
        for r in results:
            for box in r.boxes:
                class_name = model.names[int(box.cls)]
                confidence = box.conf.item()
                print(f"Detected: {class_name}, Confidence: {confidence}")
        
        processed_frame, new_violations = process_frame(frame, results)
        violations.extend(new_violations)
        
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        output_frames.append(frame_base64)
        
        if is_video and len(output_frames) > 10:
            break
        
        if not is_video:
            break
    
    if cap:
        cap.release()
    
    return {
        'frames': output_frames,
        'violations': violations
    }

def process_frame(frame, results):
    violations = []
    
    # Look for a license plate detection first (if trained)
    license_plate_box = None
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls)
            label = model.names[cls]
            if label == 'license_plate' and float(box.conf) > 0.3:
                license_plate_box = box
                break
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cls = int(box.cls)
            label = model.names[cls]
            
            color = (0, 255, 0)
            violation = None
            
            if label == 'motorcycle' and conf > 0.3:
                helmet_detected = check_helmet(frame, x1, y1, x2, y2)
                if not helmet_detected:
                    color = (0, 0, 255)
                    # Use license plate bounding box if available, otherwise fall back to vehicle box
                    if license_plate_box:
                        lp_x1, lp_y1, lp_x2, lp_y2 = map(int, license_plate_box.xyxy[0])
                        license_plate = perform_anpr(frame, lp_x1, lp_y1, lp_x2, lp_y2, label)
                    else:
                        license_plate = perform_anpr(frame, x1, y1, x2, y2, label)
                    violation = f"No helmet detected - Plate: {license_plate}"
                    save_violation(frame, x1, y1, x2, y2, violation)
            
            elif label == 'car' and conf > 0.3:
                seatbelt_detected = check_seatbelt(frame, x1, y1, x2, y2)
                if not seatbelt_detected:
                    color = (0, 0, 255)
                    if license_plate_box:
                        lp_x1, lp_y1, lp_x2, lp_y2 = map(int, license_plate_box.xyxy[0])
                        license_plate = perform_anpr(frame, lp_x1, lp_y1, lp_x2, lp_y2, label)
                    else:
                        license_plate = perform_anpr(frame, x1, y1, x2, y2, label)
                    violation = f"No seatbelt detected - Plate: {license_plate}"
                    save_violation(frame, x1, y1, x2, y2, violation)
            
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
    head_region = frame[y1:y1+int((y2-y1)*0.3), x1:x2]
    if head_region.size == 0:
        return False
    
    results = model(head_region, conf=0.4)
    for r in results:
        for box in r.boxes:
            if model.names[int(box.cls)] == 'helmet' and float(box.conf) > 0.4:
                return True
    return False

def check_seatbelt(frame, x1, y1, x2, y2):
    return True

def perform_anpr(frame, x1, y1, x2, y2, vehicle_type):
    # Adjust ROI to capture a larger area
    height = y2 - y1
    width = x2 - x1
    
    if vehicle_type == 'car':
        plate_y1 = y1 + int(height * 0.5)  # Adjusted to lower 50% (from 60%)
        plate_y2 = y2
        plate_x1 = x1 + int(width * 0.1)  # Wider ROI: center 80% (from 60%)
        plate_x2 = x2 - int(width * 0.1)
    elif vehicle_type == 'motorcycle':
        plate_y1 = y1 + int(height * 0.6)  # Adjusted to lower 40% (from 70%)
        plate_y2 = y2
        plate_x1 = x1 + int(width * 0.1)
        plate_x2 = x2 - int(width * 0.1)
    else:
        plate_y1, plate_y2 = y1, y2
        plate_x1, plate_x2 = x1, x2

    plate_region = frame[plate_y1:plate_y2, plate_x1:plate_x2]
    
    if plate_region.size == 0:
        print("ANPR: Plate region is empty")
        return "Unknown"
    
    # Simplified preprocessing
    gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Debug: Save preprocessed and raw plate images
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    debug_raw = f'violations/plate_raw_{timestamp}.jpg'
    debug_preprocessed = f'violations/plate_debug_{timestamp}.jpg'
    cv2.imwrite(debug_raw, plate_region)
    cv2.imwrite(debug_preprocessed, thresh)
    print(f"ANPR: Saved raw plate image to {debug_raw}")
    print(f"ANPR: Saved preprocessed plate image to {debug_preprocessed}")
    
    # Try EasyOCR
    results = reader.readtext(thresh, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', 
                             paragraph=False, detail=1)
    for (bbox, text, prob) in results:
        print(f"ANPR (EasyOCR): Detected text: {text}, Probability: {prob}")
        if prob > 0.1:  # Further lowered threshold to 0.1
            cleaned_text = ''.join(c for c in text.upper() if c.isalnum())
            if cleaned_text:
                return cleaned_text
    
    # Fallback to Tesseract
    try:
        tesseract_config = '--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        text = pytesseract.image_to_string(thresh, config=tesseract_config)
        cleaned_text = ''.join(c for c in text.strip().upper() if c.isalnum())
        if cleaned_text:
            print(f"ANPR (Tesseract): Detected text: {cleaned_text}")
            return cleaned_text
    except Exception as e:
        print(f"ANPR: Tesseract error: {e}")
    
    print("ANPR: No text detected with sufficient confidence")
    return "Unknown"

def save_violation(frame, x1, y1, x2, y2, violation):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'violations/violation_{timestamp}_{violation.replace(" ", "_")}.jpg'
    cv2.imwrite(filename, frame)