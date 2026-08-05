from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd


@dataclass
class TrackingConfig:
    """Runtime config for KLT-based point tracking."""

    max_corners: int = 30
    min_distance: int = 12
    win_size: Tuple[int, int] = (40, 40)
    max_level: int = 3
    term_count: int = 30
    term_eps: float = 0.01
    redetect_interval: int = 50
    fb_err_thresh: float = 50.0
    lk_err_thresh: float = 80.0
    draw_traj_len: int = 100
    video_fps: float = 10.0
    detector: str = "foreground"


@dataclass
class TrackingResult:
    """Output container for tracking artifacts."""

    tracks: pd.DataFrame
    output_csv: Path
    output_video: Path | None


class FrameReader:
    """Read frames from either a video file or an image directory."""

    def __init__(self, input_path: Path) -> None:
        self._cap = None
        self._files: List[Path] = []
        self._idx = 0

        if input_path.is_file() and input_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            self.mode = "video"
            self._cap = cv2.VideoCapture(str(input_path))
        elif input_path.is_dir():
            self.mode = "images"
            self._files = sorted(
                p for p in input_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
        else:
            raise ValueError(f"Unsupported input: {input_path}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.mode == "video":
            assert self._cap is not None
            return self._cap.read()

        if self._idx >= len(self._files):
            return False, None

        frame = cv2.imread(str(self._files[self._idx]))
        self._idx += 1
        return frame is not None, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


def _in_bounds(points: np.ndarray, width: int, height: int) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    return (x >= 0) & (x < width) & (y >= 0) & (y < height)


def _detect_features(gray: np.ndarray, config: TrackingConfig, fgbg: cv2.BackgroundSubtractorMOG2 | None) -> np.ndarray:
    mask = None
    if config.detector == "foreground":
        if fgbg is None:
            fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
        mask = fgbg.apply(gray)
        mask = cv2.medianBlur(mask, 5)

    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=config.max_corners,
        qualityLevel=0.01,
        minDistance=10,
        mask=mask,
    )
    if pts is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return np.float32(pts)


def _forward_backward_check(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    points: np.ndarray,
    config: TrackingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, config.term_count, config.term_eps)

    p1, st1, err1 = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        points,
        None,
        winSize=config.win_size,
        maxLevel=config.max_level,
        criteria=criteria,
    )
    p0_back, st2, _ = cv2.calcOpticalFlowPyrLK(
        curr_gray,
        prev_gray,
        p1,
        None,
        winSize=config.win_size,
        maxLevel=config.max_level,
        criteria=criteria,
    )

    fb_err = np.linalg.norm(points - p0_back, axis=2).reshape(-1)
    err = err1.reshape(-1) if err1 is not None else np.full(len(points), np.inf)
    st = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1)
    return p1, st, fb_err, err


def _default_output_paths(input_path: Path, output_dir: Path | None) -> tuple[Path, Path]:
    parent = output_dir if output_dir is not None else input_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return parent / f"tracks_ts{stamp}.csv", parent / f"tracking_overlay_{stamp}.mp4"


def _tracks_to_dataframe(tracks: Dict[int, List[Tuple[int, float, float]]]) -> pd.DataFrame:
    rows = []
    for track_id, seq in tracks.items():
        for t, x, y in seq:
            rows.append({"track_id": track_id, "t": t, "x": round(x, 3), "y": round(y, 3)})
    return pd.DataFrame(rows, columns=["track_id", "t", "x", "y"])


def extract_tracks(
    input_path: str | Path,
    config: TrackingConfig | None = None,
    output_dir: str | Path | None = None,
    output_csv: str | Path | None = None,
    save_video: bool = True,
) -> TrackingResult:
    """Extract point trajectories and export track time series.

    Args:
        input_path: Video file or image directory.
        config: Tracker configuration.
        output_dir: Optional output directory.
        output_csv: Optional explicit csv path.
        save_video: Whether to export an annotated overlay video.
    """

    cfg = config or TrackingConfig()
    input_path = Path(input_path)
    output_dir_path = Path(output_dir) if output_dir is not None else None

    reader = FrameReader(input_path)
    ok, first_frame = reader.read()
    if not ok or first_frame is None:
        reader.release()
        raise RuntimeError(f"Cannot read frames from {input_path}")

    height, width = first_frame.shape[:2]
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    default_csv, default_video = _default_output_paths(input_path, output_dir_path)
    csv_path = Path(output_csv) if output_csv else default_csv
    video_path = default_video

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, cfg.video_fps, (width, height))

    fgbg = None
    if cfg.detector == "foreground":
        fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)

    points = _detect_features(prev_gray, cfg, fgbg)
    next_track_id = 0

    active_ids: List[int] = []
    tracks: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    colors: Dict[int, Tuple[int, int, int]] = {}
    vis_traj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    def add_points(new_points: np.ndarray, t: int) -> None:
        nonlocal next_track_id
        if len(new_points) == 0:
            return
        for pt in new_points.reshape(-1, 2):
            tracks[next_track_id].append((t, float(pt[0]), float(pt[1])))
            active_ids.append(next_track_id)
            colors[next_track_id] = tuple(np.random.randint(0, 255, 3).tolist())
            next_track_id += 1

    add_points(points, t=0)
    t = 0

    while True:
        ok, frame = reader.read()
        if not ok or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t += 1

        if active_ids:
            point_array = np.float32(
                [[tracks[track_id][-1][1], tracks[track_id][-1][2]] for track_id in active_ids]
            ).reshape(-1, 1, 2)

            p1, st, fb_err, lk_err = _forward_backward_check(prev_gray, gray, point_array, cfg)
            st = st & (fb_err < cfg.fb_err_thresh) & (lk_err < cfg.lk_err_thresh)
            st = st & _in_bounds(p1.reshape(-1, 2), width, height)

            new_active_ids: List[int] = []
            p1_flat = p1.reshape(-1, 2)
            for keep, track_id, new_pt in zip(st, active_ids, p1_flat):
                if not keep:
                    continue
                tracks[track_id].append((t, float(new_pt[0]), float(new_pt[1])))
                new_active_ids.append(track_id)
                vis_traj[track_id].append((int(new_pt[0]), int(new_pt[1])))
                if len(vis_traj[track_id]) > cfg.draw_traj_len:
                    vis_traj[track_id] = vis_traj[track_id][-cfg.draw_traj_len :]
            active_ids = new_active_ids

        if t % cfg.redetect_interval == 0:
            new_pts = _detect_features(gray, cfg, fgbg)
            add_points(new_pts, t)

        if writer is not None:
            vis = frame.copy()
            for track_id in active_ids:
                color = colors[track_id]
                pts = vis_traj[track_id]
                for i in range(1, len(pts)):
                    cv2.line(vis, pts[i - 1], pts[i], color, 2)
                if pts:
                    cv2.circle(vis, pts[-1], 3, color, -1)

            cv2.putText(
                vis,
                f"t={t} active_tracks={len(active_ids)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            writer.write(vis)

        prev_gray = gray

    reader.release()
    if writer is not None:
        writer.release()

    df_tracks = _tracks_to_dataframe(tracks)
    df_tracks.to_csv(csv_path, index=False)

    return TrackingResult(
        tracks=df_tracks,
        output_csv=csv_path,
        output_video=video_path if writer is not None else None,
    )
