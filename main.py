from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *
from OCR import get_car, read_license_plate, write_csv, initialize_ocr
from moviepy.editor import VideoFileClip
import torch
import time

# Check CUDA availability and print device information
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

results = {}
mot_tracker = Sort()
path = r'E:\QRLPR\final (1).mp4'

def get_frame_rate(path):
    clip = VideoFileClip(path)
    return int(clip.fps)

# Load models
coco_model = YOLO('yolov8n.pt').to(device)
license_plate_detector = YOLO(r'E:\QRLPR\t_2.2\runs\detect\train\weights\best.pt').to(device)

# Initialize OCR
ocr_model, ocr_processor = initialize_ocr(device)

# Load video
cap = cv2.VideoCapture(path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
virtual_line_y = frame_height // 2

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, get_frame_rate(path), (frame_width, frame_height))

# Define vehicle classes (car, truck, bus, motorcycle)
vehicles = [2, 3, 5, 7]

frame_rate = get_frame_rate(path)
frame_interval = max(1, frame_rate // 2)  # Process every 0.5 seconds

frame_nmr = -1
detected_plates = {}  # Dictionary to store detected plates
processed_plates = set()  # Set to keep track of processed license plates

while cap.isOpened():
    frame_nmr += frame_interval
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    results[frame_nmr] = {'detections': [], 'processing_time': 0}

    # Draw virtual line
    cv2.line(frame, (0, virtual_line_y), (frame_width, virtual_line_y), (255, 0, 0), 2)

    # Detect vehicles
    detections = coco_model(frame, classes=vehicles)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        detections_.append([x1, y1, x2, y2, score])
        # Draw vehicle bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    if detections_:
        detections_ = np.array(detections_)
        
        # Track vehicles
        track_ids = mot_tracker.update(detections_)

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Draw license plate bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # Only process if plate is below the virtual line
            if y2 > virtual_line_y:
                # Assign license plate to car
                car_data = get_car(license_plate, track_ids)
                if car_data[4] != -1:
                    car_id = car_data[4]
                    
                    # Only process this license plate if we haven't seen it before
                    if car_id not in processed_plates:
                        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                        license_plate_text = read_license_plate(license_plate_crop, ocr_model, ocr_processor, device)
                        
                        if license_plate_text:
                            detected_plates[car_id] = license_plate_text
                            processed_plates.add(car_id)
                        
                            results[frame_nmr]['detections'].append({
                                'car': {'bbox': car_data[:4], 'id': car_id},
                                'license_plate': {
                                    'bbox': [x1, y1, x2, y2],
                                    'text': license_plate_text,
                                    'bbox_score': score,
                                }
                            })
                    
                    # Display the license plate text inside the bounding box
                    if car_id in detected_plates:
                        plate_text = detected_plates[car_id]
                        text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        text_x = int(x1)
                        text_y = int(y1) - 10 if int(y1) - 10 > 10 else int(y1) + int(y2 - y1) + 10
                        
                        # Create a white background for the text
                        bg_rect = [text_x, text_y - text_size[1] - 5, text_x + text_size[0] + 5, text_y + 5]
                        cv2.rectangle(frame, (bg_rect[0], bg_rect[1]), (bg_rect[2], bg_rect[3]), (255, 255, 255), -1)
                        
                        # Draw black text on the white background
                        cv2.putText(frame, plate_text, (text_x + 2, text_y - 2), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Calculate processing time
    results[frame_nmr]['processing_time'] = time.time() - start_time

    # Write the frame to video
    out.write(frame)

cap.release()
out.release()

# Write results
write_csv(results, './test.csv')