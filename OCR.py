import cv2
import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import string

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {v: k for k, v in dict_char_to_int.items()}

def initialize_ocr(device):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed").to(device)
    return model, processor

def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'license_plate_bbox_score', 
                                          'license_number', 'processing_time'))

        for frame_nmr, frame_data in results.items():
            for detection in frame_data['detections']:
                car_id = detection['car']['id']
                license_plate_bbox_score = detection['license_plate']['bbox_score']
                license_number = detection['license_plate']['text']
                processing_time = frame_data['processing_time']

                f.write('{},{},{},{},{}\n'.format(frame_nmr, car_id,
                                                  license_plate_bbox_score,
                                                  license_number,
                                                  processing_time))

def license_complies_format(text):
    if len(text) != 10:
        return False
    
    format_rules = [
        lambda x: x in string.ascii_uppercase or x in dict_int_to_char,
        lambda x: x in string.ascii_uppercase or x in dict_int_to_char,
        lambda x: x.isdigit() or x in dict_char_to_int,
        lambda x: x.isdigit() or x in dict_char_to_int,
        lambda x: x in string.ascii_uppercase or x in dict_int_to_char,
        lambda x: x in string.ascii_uppercase or x in dict_int_to_char,
        lambda x: x.isdigit() or x in dict_char_to_int,
        lambda x: x.isdigit() or x in dict_char_to_int,
        lambda x: x.isdigit() or x in dict_char_to_int,
        lambda x: x.isdigit() or x in dict_char_to_int
    ]
    
    return all(rule(char) for char, rule in zip(text, format_rules))

def format_license(text):
    return ''.join(dict_int_to_char.get(c, c) if i in (0, 1, 4, 5) else dict_char_to_int.get(c, c) for i, c in enumerate(text))

def read_license_plate(license_plate_crop, model, processor, device):
    # Convert OpenCV image (BGR) to PIL Image (RGB)
    license_plate_crop_rgb = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(license_plate_crop_rgb)
    
    # Preprocess image
    pixel_values = processor(pil_image, return_tensors="pt").pixel_values.to(device)
    
    # Generate
    generated_ids = model.generate(pixel_values)
    
    # Decode
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Format and validate license plate
    formatted_text = format_license(generated_text)
    if license_complies_format(formatted_text):
        return formatted_text
    return None

def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate
    
    for vehicle in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return vehicle
    
    return (-1, -1, -1, -1, -1)