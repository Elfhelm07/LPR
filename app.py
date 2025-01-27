from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *
from OCR import get_car, read_license_plate, initialize_ocr
from moviepy.editor import VideoFileClip
import torch
import time
from methonds import DBLinker
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Construct the connection string using environment variables
db_username = os.getenv('DB_USERNAME')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')

path = r'E:\QRLPR\final (1).mp4'
connection_string = f"mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
#connection_string = f"mysql+pymysql://root:2003@localhost:3306/SecureScanAlpha"

# Initialize database connection
db_linker = DBLinker(connection_string)
logging.info("Database connection initialized")

def get_frame_rate(path):
    clip = VideoFileClip(path)
    return int(clip.fps)

def numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    return obj

def format_bounding_box(bbox):
    return ','.join(f'{coord:.2f}' for coord in bbox)

@app.route('/process_video', methods=['POST'])
def process_video():
    video_path = request.json['video_path']
    logging.info(f"Processing video: {video_path}")

    results = {}
    mot_tracker = Sort()

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    coco_model = YOLO('yolov8n.pt').to(device)
    license_plate_detector = YOLO(r'E:\QRLPR\t_2.2\runs\detect\train\weights\best.pt').to(device)
    logging.info("Models loaded successfully")

    # Initialize OCR
    ocr_model, ocr_processor = initialize_ocr(device)
    logging.info("OCR initialized")

    # Load video
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    virtual_line_y = frame_height // 2
    logging.info(f"Video loaded. Dimensions: {frame_width}x{frame_height}")

    # Define vehicle classes (car, truck, bus, motorcycle)
    vehicles = [2, 3, 5, 7]

    frame_rate = get_frame_rate(video_path)
    frame_interval = max(1, frame_rate // 2)  # Process every 0.5 seconds
    logging.info(f"Frame rate: {frame_rate}, Frame interval: {frame_interval}")

    frame_nmr = -1
    detected_plates = {}  # Dictionary to store detected plates
    processed_plates = set()  # Set to keep track of processed license plates

    while cap.isOpened():
        frame_nmr += frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
        ret, frame = cap.read()
        if not ret:
            logging.info("Reached end of video")
            break

        logging.info(f"Processing frame {frame_nmr}")
        start_time = time.time()
        results[frame_nmr] = {'detections': [], 'processing_time': 0}

        # Detect vehicles
        detections = coco_model(frame, classes=vehicles)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            detections_.append([x1, y1, x2, y2, score])

        if detections_:
            detections_ = np.array(detections_)
            
            # Track vehicles
            track_ids = mot_tracker.update(detections_)

            # Detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

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

                                # Add the plate to the database
                                try:
                                    db_linker.recordVehicle(
                                        int(time.time()),
                                        license_plate_text,
                                        format_bounding_box([x1, y1, x2, y2]),  # Use the new formatting function
                                        int(score * 100)
                                    )
                                    logging.info(f"Added plate {license_plate_text} to database")
                                except Exception as e:
                                    logging.error(f"Error adding plate to database: {str(e)}")
                                    db_linker.SESSION.rollback()  # Add this line to rollback the session on error

        # Calculate processing time
        results[frame_nmr]['processing_time'] = time.time() - start_time
        logging.info(f"Frame {frame_nmr} processed in {results[frame_nmr]['processing_time']:.2f} seconds")

    cap.release()
    logging.info("Video processing completed")

    # Convert results to JSON-serializable format
    serializable_results = numpy_to_python(results)

    return jsonify({"message": "Video processed successfully", "results": serializable_results})

@app.route('/')
def home():
    return "Welcome to the ANPR API. Use POST /process_video to process a video."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)