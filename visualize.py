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


def vote_plate(texts, scores, weights):
    """Combine many OCR reads of one plate into a single string.

    For each character position, add up score * weight of every read that has a
    given character there and keep the character with the highest total. This
    is much more robust than trusting the single highest-scoring read, whose
    confidence is often a noisy outlier (e.g. 'LM15ZZC' beating 26 'LN15ZZC' reads).
    The caller passes the squared plate width (proportional to the plate's pixel
    area): reads of larger, closer plates are far more accurate than reads of
    small, distant ones (e.g. a van's '9' is read as '6' mostly when far away).
    """
    totals = [{} for _ in range(7)]
    for text, score, weight in zip(texts, scores, weights):
        for i, char in enumerate(text):
            totals[i][char] = totals[i].get(char, 0.0) + score * weight
    return ''.join(max(t, key=lambda c: t[c]) for t in totals)


def overlap_area(a, b):
    """Intersection area of two (left, top, right, bottom) rectangles."""
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def place_overlays(items, car_boxes, frame_w, frame_h, prev):
    """Choose where to draw each car's plate overlay so overlays don't cover each other.

    items: dicts with 'id', 'car' (x1, y1, x2, y2) and overlay size 'w', 'h'.
    car_boxes: every car box in the frame (overlays should avoid covering cars too).
    prev: car_id -> overlay offset (dx, dy) relative to the car's top centre last
          frame; updated in place so overlays stay put instead of jumping around.

    Greedy: closer (larger) cars pick first; each tries a set of candidate spots
    and takes the one with the lowest penalty. Returns car_id -> rectangle.
    """
    placed = {}
    for item in sorted(items, key=lambda it: it['car'][2] - it['car'][0], reverse=True):
        x1, y1, x2, y2 = item['car']
        w, h = item['w'], item['h']
        cx = (x1 + x2) / 2
        pref = (cx - w / 2, y1 - CROP_GAP - h)  # centred above the car

        candidates = [(pref[0] + dx * w, pref[1] + dy * h)
                      for dy in (0, -1) for dx in (0, -0.5, 0.5, -1, 1, -1.5, 1.5)]
        candidates += [(x1 - CROP_GAP - w, y1), (x2 + CROP_GAP, y1)]  # beside the car
        if item['id'] in prev:
            candidates.append((cx + prev[item['id']][0], y1 + prev[item['id']][1]))

        best, best_score = None, None
        for left, top in candidates:
            left = int(min(max(left, 0), frame_w - w))
            top = int(min(max(top, 0), frame_h - h))
            rect = (left, top, left + w, top + h)
            area = w * h
            score = (100 * sum(overlap_area(rect, r) for r in placed.values()) / area
                     + 3 * sum(overlap_area(rect, c) for c in car_boxes) / area
                     + np.hypot(left - pref[0], top - pref[1]) / w)
            if item['id'] in prev:
                pdx, pdy = prev[item['id']]
                score += 2 * np.hypot(left - (cx + pdx), top - (y1 + pdy)) / w
            if best_score is None or score < best_score:
                best, best_score = rect, score

        placed[item['id']] = best
        prev[item['id']] = (best[0] - cx, best[1] - y1)
    return placed


results = pd.read_csv('./test_interpolated.csv')

# load video (path can be passed by main.py: python visualize.py <video>)
video_path = sys.argv[1] if len(sys.argv) > 1 else './sample.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

CROP_HEIGHT = 240   # height of the enlarged plate crop shown above each car
CROP_GAP = 60       # gap between the car box and the plate crop
TEXT_BOX = 170      # height of the white box holding the plate text
FONT_SCALE = 2.6    # plate text size (shrunk further if wider than the crop)
FONT_THICKNESS = 10
LINK_COLOR = (0, 255, 0)  # line joining a displaced overlay to its car
# OCR reads below this mean character confidence are ignored, so unreadable
# (blurred/distant) plates get no text instead of a confident-looking wrong one
MIN_READ_SCORE = 0.6

license_plate = {}
for car_id in np.unique(results['car_id']):
    car_rows = results[results['car_id'] == car_id]
    # only real OCR reads: interpolated rows have license_number '0' and score 0
    texts = car_rows['license_number'].astype(str)
    scores = pd.to_numeric(car_rows['license_number_score'], errors='coerce').fillna(0)
    reads = (texts.str.len() == 7) & (scores >= MIN_READ_SCORE)
    if not reads.any():
        continue
    texts, scores = texts[reads], scores[reads]
    widths = car_rows.loc[reads, 'license_plate_bbox'].map(
        lambda b: max(parse_bbox(b)[2] - parse_bbox(b)[0], 0.0))
    plate_number = vote_plate(texts, scores, widths ** 2)

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
overlay_offsets = {}       # car_id -> last overlay offset, keeps overlays steady
frames_with_overlap = 0

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        frame_h, frame_w = frame.shape[:2]
        df_ = results[results['frame_nmr'] == frame_nmr]

        # pass 1: car borders and plate boxes for every car
        car_boxes, overlays = [], []
        for row_indx in range(len(df_)):
            row = df_.iloc[row_indx]

            car_x1, car_y1, car_x2, car_y2 = parse_bbox(row['car_bbox'])
            car_boxes.append((car_x1, car_y1, car_x2, car_y2))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                        line_length_x=200, line_length_y=200)

            x1, y1, x2, y2 = parse_bbox(row['license_plate_bbox'])
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 0, 255), 12)

            plate = license_plate.get(row['car_id'])
            if plate is None or plate['license_crop'] is None:
                continue
            H, W, _ = plate['license_crop'].shape
            if H + TEXT_BOX > frame_h or W > frame_w:
                continue  # overlay larger than the frame itself
            overlays.append({'id': row['car_id'], 'car': (car_x1, car_y1, car_x2, car_y2),
                             'w': W, 'h': H + TEXT_BOX, 'plate': plate})

        # one overlay per plate number: when the tracker splits one car into two
        # overlapping tracks, both carry the same plate; keep the larger box
        by_plate = {}
        for item in overlays:
            key = item['plate']['license_plate_number']
            size = item['car'][2] - item['car'][0]
            if key not in by_plate or size > by_plate[key]['car'][2] - by_plate[key]['car'][0]:
                by_plate[key] = item
        overlays = list(by_plate.values())

        # pass 2: overlays (white text box + enlarged plate crop) on top of everything,
        # placed so they don't cover each other
        rects = place_overlays(overlays, car_boxes, frame_w, frame_h, overlay_offsets)
        placed_rects = list(rects.values())
        if any(overlap_area(a, b) for i, a in enumerate(placed_rects) for b in placed_rects[i + 1:]):
            frames_with_overlap += 1

        for item in overlays:
            left, text_top, right, bottom = rects[item['id']]
            crop_top = text_top + TEXT_BOX
            license_crop = item['plate']['license_crop']

            # link the overlay to its car when it isn't sitting right above it
            car_x1, car_y1, car_x2, _ = item['car']
            cx = int((car_x1 + car_x2) / 2)
            if not (left <= cx <= right) or not (0 <= car_y1 - bottom <= 1.5 * CROP_GAP):
                anchor = (min(max(cx, left), right), min(max(int(car_y1), text_top), bottom))
                cv2.line(frame, anchor, (cx, int(car_y1)), LINK_COLOR, 6)

            frame[crop_top:bottom, left:right, :] = license_crop
            frame[text_top:crop_top, left:right, :] = (255, 255, 255)

            # fit the text inside the white box
            text = item['plate']['license_plate_number']
            scale, thickness = FONT_SCALE, FONT_THICKNESS
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            if text_width > 0.92 * (right - left):
                scale *= 0.92 * (right - left) / text_width
                thickness = max(2, int(round(FONT_THICKNESS * scale / FONT_SCALE)))
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            cv2.putText(frame,
                        text,
                        (int(left + (right - left - text_width) / 2),
                         int(text_top + TEXT_BOX / 2 + text_height / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        scale,
                        (0, 0, 0),
                        thickness)

        out.write(frame)

out.release()
cap.release()
print(f'frames where plate overlays still overlap: {frames_with_overlap} of {frame_nmr}')
