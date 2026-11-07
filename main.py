import subprocess
import sys

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from sort.sort import *
from util import *


def detect(model, frame):
    """Run a YOLO detection model on one frame and return its boxes as
    [x1, y1, x2, y2, score, class_id] lists."""
    # model(frame) returns a list with one Results per image; the type hints also
    # allow generators/Tensors (stream=True etc.), so take the first item generically
    result = next(iter(model(frame)))
    assert isinstance(result, Results) and result.boxes is not None
    return result.boxes.data.tolist()


results = {}

# max_age: frames a track survives without a detection (default 1 splits one car into many IDs)
vehicle_tracker = Sort(max_age=30, min_hits=3)


# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./runs/detect/train/weights/best.pt')
# input video: python main.py [video_path]  (default: sample.mp4)
video_path = sys.argv[1] if len(sys.argv) > 1 else './sample.mp4'
cap = cv2.VideoCapture(video_path)

vehicles = [2, 3, 5, 7]
# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections_ = []
        for detection in detect(coco_model, frame):
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        # SORT expects an (N, 5) array; pass an empty (0, 5) array when no vehicles are found.
        # (Pylance infers dets must be exactly (0, 5) from SORT's default argument; false positive)
        track_ids = vehicle_tracker.update(
            np.asarray(detections_) if detections_ else np.empty((0, 5)))  # pyright: ignore[reportArgumentType]

        # detect license plates
        for license_plate in detect(license_plate_detector, frame):
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(
                license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[max(int(y1), 0):int(y2),
                                           max(int(x1), 0):int(x2), :]
                if license_plate_crop.size == 0:
                    continue

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(
                    license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(
                    license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(
                    license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

cap.release()

# write results
write_csv(results, './test.csv')


# Run the interpolation script (sys.executable = the same Python/venv running this script)
subprocess.run([sys.executable, "add_missing_data.py"], check=True)

# Run the visualization script
subprocess.run([sys.executable, "visualize.py", video_path], check=True)
