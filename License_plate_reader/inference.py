## Imports
import os
import argparse
import numpy as np
import cv2
from ultralytics import YOLO
from sort.sort import Sort
from utils import get_car, read_license_plate, write_csv
from tqdm import tqdm

## Constants
VEHICLE_CLASSES = [2, 3, 5, 7]
COCO_MODEL_PATH = 'yolov8n.pt'
LICENSE_PLATE_MODEL_PATH = 'models/Licence_plate_YOLOv8n.pt'

## Load Models
coco_model = YOLO(COCO_MODEL_PATH)
license_plate_detector = YOLO(LICENSE_PLATE_MODEL_PATH)

## Tracker
mot_tracker = Sort()

def process_frame(frame, frame_number, results):
    """Process a single video frame to detect vehicles and license plates."""
    license_plate_detections = license_plate_detector.predict(frame, classes=[0], verbose=False)[0]
    detections = coco_model.predict(frame, classes=VEHICLE_CLASSES, verbose=False)[0]
    
    detections_ = []
    for detection in detections.boxes:
        x1, y1, x2, y2 = detection.xyxy.flatten().tolist()
        car_conf = float(detection.conf)
        detections_.append([x1, y1, x2, y2, car_conf])

    track_ids = mot_tracker.update(np.array(detections_))

    results[frame_number] = {}
    for license_plate in license_plate_detections.boxes:
        x1, y1, x2, y2 = license_plate.xyxy.flatten().tolist()
        license_plate_conf = float(license_plate.conf)
        
        xcar1, ycar1, xcar2, ycar2, track_id = get_car(license_plate=license_plate, Vehicle_track_ids=track_ids)
        
        if track_id != -1:
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, thresholded_license_plate_crop = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
            license_plate_text, text_conf = read_license_plate(thresholded_license_plate_crop)

            if license_plate_text:
                results[frame_number][int(track_id)] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2], 'bbox_conf': round(car_conf, 2)},
                    'license_plate': {'bbox': [x1, y1, x2, y2], 'bbox_conf': round(license_plate_conf, 2),
                                      'text': license_plate_text, 'text_conf': text_conf}
                }

def main(video_file, output_file, num_frames):
    """Main function to process the video and save results to a CSV file."""
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file {video_file}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(total_frames, num_frames) if num_frames else total_frames
    results = {}
    frame_number = 0

    with tqdm(total=frames_to_process, desc="Processing Frames") as pbar:
        while frame_number < frames_to_process:
            success, frame = cap.read()
            if not success:
                break

            process_frame(frame, frame_number, results)
            frame_number += 1
            pbar.update(1)

    cap.release()

    if os.path.exists(output_file):
        os.remove(output_file)
    write_csv(results, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video file to detect vehicles and license plates.")
    parser.add_argument('--video_file', type=str, required=True, help="Path to the video file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output CSV file.")
    parser.add_argument('--num_frames', type=int, default=None, help="Number of frames to process (default is all frames).")
    args = parser.parse_args()

    main(args.video_file, args.output_file, args.num_frames)
