import numpy as np
import csv
from collections import defaultdict
import os 
import cv2
# from matplotlib import pyplot as plt
import argparse
import os
from datetime import datetime

# Generate a timestamp string for output file naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args():
    # Parse command-line arguments for input path to images or video
    parser = argparse.ArgumentParser(description="KLT tracking with forward-backward check, save tracks.csv next to input.")
    parser.add_argument("input_path", type=str,
                        help="the input path to images or video, in the child directory it should contain all files")
    return parser.parse_args()


args = parse_args()


# Setup input and output paths
INPUT_PATH = args.input_path
parent_dir = os.path.dirname(INPUT_PATH)
print(parent_dir)
OUTPUT_CSV = os.path.join(parent_dir, f"tracks_ts{timestamp}.csv")  # CSV file to save track data
OUTPUT_mp4 = os.path.join(parent_dir, f"output_ts{timestamp}.mp4")  # Video output with trajectories overlay


# Parameters for Shi-Tomasi corner detection
MAX_CORNERS = 30     # Max features to detect
QUALITY_LEVEL = 0.05 # Minimal accepted quality of corners (not used in detect_features, but declared)
MIN_DISTANCE = 12    # Minimum distance between detected corners
BLOCK_SIZE = 3       # Size of an average block for computing derivative covariation matrix (not used explicitly here)
USE_HARRIS = False   # Whether to use Harris detector
K_HARRIS = 0.04     # Harris detector free parameter

# Parameters for Lucas-Kanade optical flow
WIN_SIZE = (40, 40)            # Search window size at each pyramid level
MAX_LEVEL = 3                  # Number of pyramid layers
TERM_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)  # Termination criteria for iterative search


# Miscellaneous thresholds and constants
REDETECT_INTERVAL = 50         # Frames interval to re-detect new features
FB_ERR_THRESH = 50.0           # Threshold for forward-backward error to reject bad tracks
ERR_THRESH = 80.0              # Threshold for tracking error to reject bad tracks

DRAW_TRAJ_LEN = 100            # Number of points to draw for each trajectory trace


class FrameReader:
    def __init__(self, input_path):
        # Initialize reader for video file, webcam (int), or directory of images
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
        # Read a single frame either from video or from image sequence
        if self.mode == "video":
            return self.cap.read()
        elif self.mode == "images":
            if self.idx >= len(self.files):
                return False, None
            frame = cv2.imread(self.files[self.idx])
            self.idx += 1
            return frame is not None, frame


    def release(self):
        # Release video capture resources if video mode
        if self.mode == "video":
            self.cap.release()


# Initialize background subtractor for motion mask generation
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=200,
    varThreshold=25,
    detectShadows=False
)


def detect_features(frame, max_corners=200):
    # Detect good features to track using Shi-Tomasi algorithm with motion-based mask
    fgmask = fgbg.apply(frame)            # Apply background subtraction
    fgmask = cv2.medianBlur(fgmask, 5)   # Reduce noise in the mask

    # Detect corners only in foreground regions
    pts = cv2.goodFeaturesToTrack(
        frame,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=10,
        mask=fgmask
    )
    # Return points as float32 array or empty array if none found
    return np.float32(pts) if pts is not None else np.empty((0,1,2), dtype=np.float32)


def forward_backward_check(prev_gray, curr_gray, p0):
    # Track points forward from prev_gray to curr_gray
    p1, st1, err1 = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, p0, None,
        winSize=WIN_SIZE, maxLevel=MAX_LEVEL, criteria=TERM_CRITERIA
    )
    # Track points backward from curr_gray to prev_gray
    p0_back, st2, err2 = cv2.calcOpticalFlowPyrLK(
        curr_gray, prev_gray, p1, None,
        winSize=WIN_SIZE, maxLevel=MAX_LEVEL, criteria=TERM_CRITERIA
    )
    # Compute forward-backward error (distance between original points p0 and backward-tracked points p0_back)
    fb_err = np.linalg.norm(p0 - p0_back, axis=2).reshape(-1)
    # Tracking error from forward pass (reshape or fill with infinity if err1 is None)
    err1_ = err1.reshape(-1) if err1 is not None else np.full(len(p0), np.inf)
    # Status flags that indicate valid tracking both forward and backward
    st = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1)
    return p1, st, fb_err, err1_


def in_bounds(p, w, h):
    # Check if points p (Nx2) are within image boundaries (width w, height h)
    x, y = p[:,0], p[:,1]
    return (x >= 0) & (x < w) & (y >= 0) & (y < h)


# Initialize frame reader from input path
reader = FrameReader(INPUT_PATH)
ok, frame0 = reader.read()
if not ok:
    raise RuntimeError("Cannot read this folder!")

# Get frame size and convert first frame to grayscale
h, w = frame0.shape[:2]
prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

# Prepare video writer to save output visualization (drawing trajectories on frames)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_mp4, fourcc, 10, (w, h))


import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# Detect initial feature points
p0 = detect_features(prev_gray, max_corners=MAX_CORNERS)
next_track_id = 0

active_ids = []                # List of currently tracked point IDs
tracks = defaultdict(list)    # Dictionary mapping track_id -> list of (frame_index, x, y)
colors = {}                   # Random color for each track_id for visualization


def add_points(points, t):
    # Add new feature points as new tracks with unique IDs at time t
    global next_track_id
    for pt in points.reshape(-1,2):
        tracks[next_track_id].append((t, float(pt[0]), float(pt[1])))
        active_ids.append(next_track_id)
        colors[next_track_id] = tuple(np.random.randint(0, 255, 3).tolist())  # Assign random color
        next_track_id += 1


add_points(p0, t=0)
t = 0

vis_traj = defaultdict(list)  # Keep recent points to visualize trajectories track_id -> [(x,y), ...]


plt.ion()  # Enable interactive mode for plotting
fig, ax = plt.subplots()

while True:
    ok, frame = reader.read()
    if not ok:
        break

    # Convert current frame to grayscale for optical flow tracking
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t += 1

    if len(active_ids) > 0:
        # Collect last known points for all active tracks
        p0_arr = []
        for tid in active_ids:
            xlast, ylast = tracks[tid][-1][1], tracks[tid][-1][2]
            p0_arr.append([xlast, ylast])
        p0_arr = np.float32(p0_arr).reshape(-1,1,2)

        # Perform forward-backward optical flow check to track points to next frame
        p1, st, fb_err, err1_ = forward_backward_check(prev_gray, gray, p0_arr) 

        # Filter tracked points based on forward-backward error, tracking error, and image bounds
        st = st & (fb_err < FB_ERR_THRESH) & (err1_ < ERR_THRESH)
        st = st & in_bounds(p1.reshape(-1,2), w, h)

        new_active_ids = []
        p1_flat = p1.reshape(-1,2)
        # Update tracks and visualization trajectories for kept points
        for keep, tid, new_pt in zip(st, active_ids, p1_flat):
            if keep:
                tracks[tid].append((t, float(new_pt[0]), float(new_pt[1])))
                new_active_ids.append(tid)
                vis_traj[tid].append((int(new_pt[0]), int(new_pt[1])))
                # Keep only last DRAW_TRAJ_LEN points for drawing trajectory lines
                if len(vis_traj[tid]) > DRAW_TRAJ_LEN:
                    vis_traj[tid] = vis_traj[tid][-DRAW_TRAJ_LEN:]
        active_ids = new_active_ids

    # Every REDETECT_INTERVAL frames, detect new points avoiding existing ones
    if t % REDETECT_INTERVAL == 0:
        mask = np.full((h, w), 255, dtype=np.uint8)
        # Mask out areas near existing tracked points to avoid duplication
        for tid in active_ids:
            x, y = tracks[tid][-1][1], tracks[tid][-1][2]
            cv2.circle(mask, (int(x), int(y)), MIN_DISTANCE, 0, -1)
        new_pts = detect_features(gray, max_corners=MAX_CORNERS // 2)
        if new_pts is not None and len(new_pts) > 0:
            add_points(new_pts, t)

    # Visualization: draw trajectories and current points on the frame copy
    vis = frame.copy()
    for tid in active_ids:
        c = colors[tid]
        pts_list = vis_traj[tid]
        for i in range(1, len(pts_list)):
            cv2.line(vis, pts_list[i-1], pts_list[i], c, 2)
        if len(pts_list) > 0:
            cv2.circle(vis, pts_list[-1], 3, c, -1)

    # Draw info text about current frame and active tracks count
    cv2.putText(vis, f"t={t} active_tracks={len(active_ids)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    # Write visualization frame to output video
    out.write(vis)

    # Update previous gray frame for next iteration
    prev_gray = gray


out.release()  # Release video writer

# Save all tracks to CSV file: each row has track ID, frame index, and (x,y) coordinates
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["track_id", "t", "x", "y"])
    for tid, seq in tracks.items():
        for (tt, xx, yy) in seq:
            writer.writerow([tid, tt, f"{xx:.3f}", f"{yy:.3f}"])
    print(f"Save_to: {OUTPUT_CSV}")


# Example usage comment:
# python script/extract_ts_using_optical_flow_with_object_detection.py /path/to/image_folder_or_video
