import sys

import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y),
             color, thickness)  # -- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y),
             color, thickness)  # -- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1),
             color, thickness)  # -- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y),
             color, thickness)  # -- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


def parse_bbox(value):
    """Parse '[x1 y1 x2 y2]' or 'x1 y1 x2 y2' into four floats."""
    return [float(v) for v in str(value).strip().strip('[]').split()]


def vote_plate(texts, scores):
    """Combine many OCR reads of one plate into a single string.

    For each character position, add up the OCR scores of every read that has a
    given character there and keep the character with the highest total. This
    is much more robust than trusting the single highest-scoring read, whose
    confidence is often a noisy outlier (e.g. 'LM15ZZC' beating 26 'LN15ZZC' reads).
    """
    totals = [{} for _ in range(7)]
    for text, score in zip(texts, scores):
        for i, char in enumerate(text):
            totals[i][char] = totals[i].get(char, 0.0) + score
    return ''.join(max(t, key=lambda c: t[c]) for t in totals)


results = pd.read_csv('./test_interpolated.csv')

# load video (path can be passed by main.py: python visualize.py <video>)
video_path = sys.argv[1] if len(sys.argv) > 1 else './sample.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

CROP_HEIGHT = 400   # height of the enlarged plate crop shown above each car
CROP_GAP = 100      # gap between the car box and the plate crop
TEXT_BOX = 300      # height of the white box holding the plate text

license_plate = {}
for car_id in np.unique(results['car_id']):
    car_rows = results[results['car_id'] == car_id]
    # only real OCR reads: interpolated rows have license_number '0' and score 0
    texts = car_rows['license_number'].astype(str)
    scores = pd.to_numeric(car_rows['license_number_score'], errors='coerce').fillna(0)
    reads = (texts.str.len() == 7) & (scores > 0)
    if not reads.any():
        continue
    texts, scores = texts[reads], scores[reads]
    plate_number = vote_plate(texts, scores)

    # crop from the best read that matches the voted plate (fallback: best read overall)
    matching = scores[texts == plate_number]
    best = results.loc[(matching if len(matching) else scores).idxmax()]
    license_plate[car_id] = {'license_crop': None,
                             'license_plate_number': plate_number}

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(best['frame_nmr']))
    ret, frame = cap.read()
    if not ret:
        continue

    x1, y1, x2, y2 = parse_bbox(best['license_plate_bbox'])
    license_crop = frame[max(int(y1), 0):int(y2), max(int(x1), 0):int(x2), :]
    if license_crop.size == 0 or y2 <= y1:
        continue  # degenerate box: draw this car without the enlarged crop

    license_plate[car_id]['license_crop'] = cv2.resize(
        license_crop, (max(int((x2 - x1) * CROP_HEIGHT / (y2 - y1)), 1), CROP_HEIGHT))


frame_nmr = -1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            row = df_.iloc[row_indx]

            # draw car
            car_x1, car_y1, car_x2, car_y2 = parse_bbox(row['car_bbox'])
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                        line_length_x=200, line_length_y=200)

            # draw license plate
            x1, y1, x2, y2 = parse_bbox(row['license_plate_bbox'])
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 0, 255), 12)

            # enlarged plate crop + text above the car
            plate = license_plate.get(row['car_id'])
            if plate is None or plate['license_crop'] is None:
                continue
            license_crop = plate['license_crop']
            H, W, _ = license_crop.shape

            frame_h, frame_w = frame.shape[:2]
            if H + TEXT_BOX > frame_h or W > frame_w:
                continue  # overlay larger than the frame itself

            # place the overlay above the car, but clamp it inside the frame
            # (negative indices would otherwise wrap around and draw in the wrong place)
            text_top = int(car_y1) - CROP_GAP - H - TEXT_BOX
            text_top = min(max(text_top, 0), frame_h - H - TEXT_BOX)
            crop_top = text_top + TEXT_BOX
            crop_bottom = crop_top + H
            left = int((car_x1 + car_x2 - W) / 2)
            left = min(max(left, 0), frame_w - W)
            right = left + W

            frame[crop_top:crop_bottom, left:right, :] = license_crop
            frame[text_top:crop_top, left:right, :] = (255, 255, 255)

            text = plate['license_plate_number']
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)
            cv2.putText(frame,
                        text,
                        (int(left + (W - text_width) / 2),
                         int(text_top + TEXT_BOX / 2 + text_height / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4.3,
                        (0, 0, 0),
                        17)

        out.write(frame)

out.release()
cap.release()
