import sys
import time
from pathlib import Path

import cv2
import pandas as pd

min_time = 0
if len(sys.argv) > 1:
    video_path = Path(sys.argv[1]).expanduser()
    if len(sys.argv) == 3:
        min_time = float(sys.argv[2])
else:
    video_path = None
webcam = video_path is None
fps = 15
diameter = 400

frame_idx = 0
records = []

slow_factor = 3

def click_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        h, w = image.shape[:2]
        point = x, y
        s = frame_idx / fps
        print('Adding', point, 'at time', s)
        if not records:
            records.append({'ms': 0, 'x': x / w, 'y': y / h, 'diameter': diameter})
        records.append({'ms': s * 1000, 'x': x / w, 'y': y / h, 'diameter': diameter})

cv2.namedWindow("image")
cv2.setMouseCallback("image", click_point)

cap = cv2.VideoCapture(0 if webcam else str(video_path))
while cap.isOpened():
    success, image = cap.read()
    t = frame_idx / fps
    # print(f'{t:.2f}')
    if t < min_time:
        frame_idx += 1
        continue
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        if webcam:
            continue
        else:
            break
    cv2.imshow("image", image)
    time.sleep(slow_factor * 1 / fps)
    if cv2.waitKey(5) & 0xFF == 27:
        break
    frame_idx += 1
cap.release()
cv2.destroyAllWindows()

record = dict(**records[-1])
record['ms'] = frame_idx / fps * 1000
records.append(record)

df = pd.DataFrame.from_records(records)
file_index = 0
while True:
    csv_path = video_path.with_name(f'{video_path.stem}_{file_index:02d}.csv')
    if not csv_path.is_file():
        df.to_csv(csv_path)
        print('Checkpoints saved in', csv_path)
        break
    else:
        file_index += 1
