"""
Fine-tune YOLOv8n on a PPE dataset (YOLO format, with a data.yaml).

Expected layout after downloading a Roboflow Universe "YOLOv8" export into data/:
  data/ppe/
    data.yaml         # names: [...], train: ..., val: ...
    train/images, train/labels
    valid/images, valid/labels

Usage:
  python src/train.py --data data/ppe/data.yaml --epochs 25
Best weights are copied to models/best.pt when done.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="starting weights (yolov8n.pt / yolov8s.pt)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="16 fits an 8 GB GPU at 640 for yolov8n/s")
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="ppe_yolov8n")
    args = ap.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project="runs",
        exist_ok=True,
        patience=8,       # early stop if val mAP stalls
        workers=4,
        plots=True,
    )

    # evaluate best checkpoint and report the number for the resume line
    best = Path(results.save_dir) / "weights" / "best.pt"
    metrics = YOLO(best).val(data=args.data, device=args.device, plots=False)
    print("\n=== Validation (best.pt) ===")
    print(f"mAP@0.5      : {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.3f}")

    Path("models").mkdir(exist_ok=True)
    shutil.copy(best, "models/best.pt")
    print("\nCopied best weights -> models/best.pt")
    print("Next: python src/ppe_pipeline.py --source samples/<video>.mp4 --weights models/best.pt --show-ppe")


if __name__ == "__main__":
    main()
