import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from detector_tracker import VehicleTrackerPipeline
from export_csv import save_results_csv

def parse_args():
    parser = argparse.ArgumentParser(description="Detect, track, and estimate vehicle speed from video.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", default="yolo11n.pt", help="Path to YOLO model weights, e.g. yolo11n.pt or best.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config name, e.g. bytetrack.yaml or botsort.yaml")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--output-dir", default="data/outputs", help="Folder for output video and CSV")
    return parser.parse_args()

def main():
    args = parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video = output_dir / f"{video_path.stem}_annotated.mp4"
    output_csv = output_dir / f"{video_path.stem}_tracks.csv"

    model = YOLO(args.model)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    pipeline = VehicleTrackerPipeline(
        model=model,
        fps=fps,
        tracker_name=args.tracker,
        conf=args.conf
    )

    all_rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, rows = pipeline.process_frame(frame, frame_idx)
        writer.write(annotated_frame)
        all_rows.extend(rows)
        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    save_results_csv(output_csv, all_rows)

    print(f"Done. Annotated video saved to: {output_video}")
    print(f"Done. CSV saved to: {output_csv}")

if __name__ == "__main__":
    main()
