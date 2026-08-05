from __future__ import annotations

import argparse

from real_v_tsfm.extraction import TrackingConfig, extract_tracks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract time-series tracks from a video or image directory using optical flow.",
    )
    parser.add_argument("input_path", type=str, help="Path to a video file or frame directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store outputs.")
    parser.add_argument("--output-csv", type=str, default=None, help="Explicit output CSV path.")
    parser.add_argument("--no-video", action="store_true", help="Disable annotated MP4 export.")

    parser.add_argument("--detector", choices=["foreground", "corners"], default="foreground")
    parser.add_argument("--max-corners", type=int, default=30)
    parser.add_argument("--min-distance", type=int, default=12)
    parser.add_argument("--redetect-interval", type=int, default=50)
    parser.add_argument("--fb-err-thresh", type=float, default=50.0)
    parser.add_argument("--lk-err-thresh", type=float, default=80.0)
    parser.add_argument("--draw-traj-len", type=int, default=100)
    parser.add_argument("--fps", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = TrackingConfig(
        max_corners=args.max_corners,
        min_distance=args.min_distance,
        redetect_interval=args.redetect_interval,
        fb_err_thresh=args.fb_err_thresh,
        lk_err_thresh=args.lk_err_thresh,
        draw_traj_len=args.draw_traj_len,
        video_fps=args.fps,
        detector=args.detector,
    )

    result = extract_tracks(
        input_path=args.input_path,
        config=config,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        save_video=not args.no_video,
    )

    print(f"Saved tracks CSV: {result.output_csv}")
    if result.output_video is not None:
        print(f"Saved overlay video: {result.output_video}")
    print(f"Total rows: {len(result.tracks)}")


if __name__ == "__main__":
    main()
