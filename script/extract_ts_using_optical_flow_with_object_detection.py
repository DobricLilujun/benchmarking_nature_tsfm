import numpy as np
import csv
from collections import defaultdict
import os 
import cv2
# from matplotlib import pyplot as plt
import argparse
import os

from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def parse_args():
    parser = argparse.ArgumentParser(description="KLT tracking with forward-backward check, save tracks.csv next to input.")
    parser.add_argument("input_path", type=str,
                        help="the input path to images or video, in the child directory it should contain all files")
    return parser.parse_args()


args = parse_args()


INPUT_PATH = args.input_path
parent_dir = os.path.dirname(INPUT_PATH)
print(parent_dir)
OUTPUT_CSV = os.path.join(parent_dir, f"tracks_ts{timestamp}.csv")
OUTPUT_mp4 = os.path.join(parent_dir, f"output_ts{timestamp}.mp4")

MAX_CORNERS = 30
QUALITY_LEVEL = 0.05
MIN_DISTANCE = 12
BLOCK_SIZE = 3
USE_HARRIS = False  
K_HARRIS = 0.04 

WIN_SIZE = (40, 40)  ## Original size 21 21 
MAX_LEVEL = 3
TERM_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

REDETECT_INTERVAL = 50  
FB_ERR_THRESH = 50.0 # Original 1.5  
ERR_THRESH = 80.0  # Original 30.0

DRAW_TRAJ_LEN = 100      

 


class FrameReader:
    def __init__(self, input_path):
        if isinstance(input_path, int) or input_path.endswith(".mp4") or input_path.endswith(".avi"):
            self.mode = "video"
            self.cap = cv2.VideoCapture(input_path)
        elif os.path.isdir(input_path):
            self.mode = "images"
            files = sorted(os.listdir(input_path))
            self.files = [os.path.join(input_path, f) for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            self.idx = 0
        else:
            raise ValueError("inputs are illegal")

    def read(self):
        if self.mode == "video":
            return self.cap.read()
        elif self.mode == "images":
            if self.idx >= len(self.files):
                return False, None
            frame = cv2.imread(self.files[self.idx])
            # print (self.idx)
            self.idx += 1
            return frame is not None, frame

    def release(self):
        if self.mode == "video":
            self.cap.release()
        

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=200,
    varThreshold=25, 
    detectShadows=False 
)

def detect_features(frame, max_corners=200):

    fgmask = fgbg.apply(frame)
    fgmask = cv2.medianBlur(fgmask, 5)  

    pts = cv2.goodFeaturesToTrack(
        frame,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=10,
        mask=fgmask
    )
    return np.float32(pts) if pts is not None else np.empty((0,1,2), dtype=np.float32)


def forward_backward_check(prev_gray, curr_gray, p0):
    # Forward checking
    p1, st1, err1 = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, p0, None,
        winSize=WIN_SIZE, maxLevel=MAX_LEVEL, criteria=TERM_CRITERIA
    )
    # Backward checking
    p0_back, st2, err2 = cv2.calcOpticalFlowPyrLK(
        curr_gray, prev_gray, p1, None,
        winSize=WIN_SIZE, maxLevel=MAX_LEVEL, criteria=TERM_CRITERIA
    )
    fb_err = np.linalg.norm(p0 - p0_back, axis=2).reshape(-1)
    err1_ = err1.reshape(-1) if err1 is not None else np.full(len(p0), np.inf)
    st = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1)
    return p1, st, fb_err, err1_

def in_bounds(p, w, h):
    x, y = p[:,0], p[:,1]
    return (x >= 0) & (x < w) & (y >= 0) & (y < h)




reader = FrameReader(INPUT_PATH)
ok, frame0 = reader.read()
if not ok:
    raise RuntimeError("Cannot read this folder!")

h, w = frame0.shape[:2]
prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(OUTPUT_mp4, fourcc, 10, (w, h)) 

import matplotlib.pyplot as plt
from IPython.display import display, clear_output


p0 = detect_features(prev_gray, max_corners=MAX_CORNERS)
next_track_id = 0

active_ids = []
tracks = defaultdict(list)  # track_id -> [(t,x,y)]
colors = {}

def add_points(points, t):
    global next_track_id
    for pt in points.reshape(-1,2):
        tracks[next_track_id].append((t, float(pt[0]), float(pt[1])))
        active_ids.append(next_track_id)
        colors[next_track_id] = tuple(np.random.randint(0, 255, 3).tolist())
        next_track_id += 1

add_points(p0, t=0)
t = 0

vis_traj = defaultdict(list)  # track_id -> [(x,y), ...]

plt.ion()  # interactive mode
fig, ax = plt.subplots()

while True:
    ok, frame = reader.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t += 1

    if len(active_ids) > 0:

        p0_arr = []
        for tid in active_ids:
            xlast, ylast = tracks[tid][-1][1], tracks[tid][-1][2]
            p0_arr.append([xlast, ylast])
        p0_arr = np.float32(p0_arr).reshape(-1,1,2)

        p1, st, fb_err, err1_ = forward_backward_check(prev_gray, gray, p0_arr) 

        st = st & (fb_err < FB_ERR_THRESH) & (err1_ < ERR_THRESH)
        st = st & in_bounds(p1.reshape(-1,2), w, h)

        new_active_ids = []
        p1_flat = p1.reshape(-1,2)
        for keep, tid, new_pt in zip(st, active_ids, p1_flat):
            if keep:
                tracks[tid].append((t, float(new_pt[0]), float(new_pt[1])))
                new_active_ids.append(tid)
                vis_traj[tid].append((int(new_pt[0]), int(new_pt[1])))
                if len(vis_traj[tid]) > DRAW_TRAJ_LEN:
                    vis_traj[tid] = vis_traj[tid][-DRAW_TRAJ_LEN:]
        active_ids = new_active_ids

    if t % REDETECT_INTERVAL == 0:
        mask = np.full((h, w), 255, dtype=np.uint8)
        for tid in active_ids:
            x, y = tracks[tid][-1][1], tracks[tid][-1][2]
            cv2.circle(mask, (int(x), int(y)), MIN_DISTANCE, 0, -1)
        new_pts = detect_features(gray, max_corners=MAX_CORNERS // 2)
        if new_pts is not None and len(new_pts) > 0:
            add_points(new_pts, t)

    vis = frame.copy()
    for tid in active_ids:
        c = colors[tid]
        pts_list = vis_traj[tid]
        for i in range(1, len(pts_list)):
            cv2.line(vis, pts_list[i-1], pts_list[i], c, 2)
        if len(pts_list) > 0:
            cv2.circle(vis, pts_list[-1], 3, c, -1)

    cv2.putText(vis, f"t={t} active_tracks={len(active_ids)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)


    out.write(vis)
    # clear_output(wait=True)
    # plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    # plt.title(f"t={t}, active_tracks={len(active_ids)}")
    # plt.axis("off")
    # display(plt.gcf())            
    # plt.pause(0.1)
    prev_gray = gray

out.release() 
# plt.ioff()
# plt.show()

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["track_id", "t", "x", "y"])
    for tid, seq in tracks.items():
        for (tt, xx, yy) in seq:
            writer.writerow([tid, tt, f"{xx:.3f}", f"{yy:.3f}"])
    print(f"Save_to: {OUTPUT_CSV}")



# python script/extract_ts_using_optical_flow_with_object_detection.py /home/snt/projects_lujun/benchmarking_nature_tsfm/data/video/LaSOT/swing/swing/swing-14/img