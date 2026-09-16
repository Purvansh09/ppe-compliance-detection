"""
Real-time PPE compliance pipeline.

video/webcam -> YOLOv8 detection -> ByteTrack IDs -> rule-based violation check
             -> annotated .mp4 + violations.csv

Works with:
  * stock COCO yolov8n.pt   (person only -> pipeline smoke test, no PPE classes)
  * a fine-tuned PPE model  (person + helmet/no-helmet/vest/no-vest style classes)

Usage:
  python src/ppe_pipeline.py --source samples/site.mp4 --weights models/best.pt
  python src/ppe_pipeline.py --source 0                  # webcam
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------------------------------- #
# Class-name normalisation
# Public PPE datasets name things differently ("Hardhat", "NO-Hardhat", "Safety Vest",
# "no_helmet", ...). Map every model class to one of our canonical roles.
# --------------------------------------------------------------------------- #
ROLE_PERSON = "person"
ROLE_HELMET = "helmet"
ROLE_NO_HELMET = "no-helmet"
ROLE_VEST = "vest"
ROLE_NO_VEST = "no-vest"


def classify_role(name: str) -> str | None:
    n = name.lower().replace("_", "-").replace(" ", "-")
    negative = n.startswith("no-") or n.startswith("without")
    if "person" in n or n in ("worker", "people"):
        return ROLE_PERSON
    if "helmet" in n or "hardhat" in n or "hard-hat" in n:
        return ROLE_NO_HELMET if negative else ROLE_HELMET
    if "vest" in n:
        return ROLE_NO_VEST if negative else ROLE_VEST
    return None  # mask, cone, machinery, etc. -> ignored


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def containment(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction of `inner` box area that lies inside `outer` box (xyxy)."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return float(inter / area)


# --------------------------------------------------------------------------- #
# Violation rule
# --------------------------------------------------------------------------- #
@dataclass
class PersonStatus:
    track_id: int
    box: np.ndarray
    missing: list[str] = field(default_factory=list)

    @property
    def compliant(self) -> bool:
        return not self.missing


def evaluate_frame(
    boxes: np.ndarray,
    roles: list[str | None],
    track_ids: list[int | None],
    require: tuple[str, ...],
    overlap_thr: float,
    require_positive: bool,
) -> list[PersonStatus]:
    """
    For each tracked person box decide which PPE items are missing.

    Primary rule (PRD): a person is in violation if a `no-<item>` detection box sits
    mostly inside their person box.
    Optional rule (--require-positive): also flag if no positive `<item>` box is found
    inside the person box (useful for datasets that have no "no-*" classes).
    """
    people = [
        (tid, boxes[i])
        for i, (r, tid) in enumerate(zip(roles, track_ids))
        if r == ROLE_PERSON and tid is not None
    ]
    ppe = [(roles[i], boxes[i]) for i in range(len(boxes)) if roles[i] not in (None, ROLE_PERSON)]

    statuses: list[PersonStatus] = []
    for tid, pbox in people:
        st = PersonStatus(track_id=tid, box=pbox)
        for item in require:  # "helmet" / "vest"
            neg_role, pos_role = f"no-{item}", item
            has_neg = any(r == neg_role and containment(b, pbox) >= overlap_thr for r, b in ppe)
            has_pos = any(r == pos_role and containment(b, pbox) >= overlap_thr for r, b in ppe)
            if has_neg or (require_positive and not has_pos):
                st.missing.append(item)
        statuses.append(st)
    return statuses


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
GREEN, RED, GREY, WHITE = (60, 200, 60), (40, 40, 230), (160, 160, 160), (255, 255, 255)


def draw_box(img, box, color, label: str, thick=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)


def annotate(frame, boxes, roles, confs, names_raw, statuses, show_ppe: bool):
    out = frame.copy()
    if show_ppe:
        for b, r, c, raw in zip(boxes, roles, confs, names_raw):
            if r in (None, ROLE_PERSON):
                continue
            col = RED if r.startswith("no-") else GREY
            draw_box(out, b, col, f"{raw} {c:.2f}", thick=1)
    for st in statuses:
        col = GREEN if st.compliant else RED
        lab = f"ID {st.track_id}" + ("" if st.compliant else " | missing " + ",".join(st.missing))
        draw_box(out, st.box, col, lab)
    return out


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def source_fps(source, fallback: float) -> float:
    if isinstance(source, int):
        return fallback
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
    cap.release()
    return fps if fps and fps > 1 else fallback


def run(args):
    model = YOLO(args.weights)
    role_of = {i: classify_role(n) for i, n in model.names.items()}
    print("Class -> role mapping:")
    for i, n in model.names.items():
        if role_of[i]:
            print(f"  {i:3d} {n:20s} -> {role_of[i]}")
    if not any(r == ROLE_PERSON for r in role_of.values()):
        raise SystemExit("Model has no person-like class; the pipeline needs one to track.")
    have_ppe = any(r not in (None, ROLE_PERSON) for r in role_of.values())
    if not have_ppe:
        print("NOTE: model has no PPE classes -> running in tracking-only smoke-test mode.")

    source = int(args.source) if args.source.isdigit() else args.source
    fps = source_fps(source, args.fps_assume)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "webcam" if isinstance(source, int) else Path(source).stem
    video_path = out_dir / f"{stem}_annotated.mp4"
    csv_path = out_dir / f"{stem}_violations.csv"

    writer = None
    csv_f = open(csv_path, "w", newline="")
    log = csv.writer(csv_f)
    log.writerow(["frame", "time_s", "track_id", "missing", "x1", "y1", "x2", "y2"])

    # per-track state so a violation is logged once per episode, not every frame
    active_violation: dict[int, str] = {}
    violation_frames: dict[int, int] = {}
    t0 = time.time()
    n_frames = 0

    results = model.track(
        source=source,
        tracker=args.tracker,
        conf=args.conf,
        iou=0.5,
        persist=True,
        stream=True,
        verbose=False,
        device=args.device,
        imgsz=args.imgsz,
    )
    for res in results:
        frame = res.orig_img
        n_frames += 1

        if res.boxes is None or len(res.boxes) == 0:
            boxes, roles, confs, raw, tids = np.zeros((0, 4)), [], [], [], []
        else:
            boxes = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy().tolist()
            roles = [role_of[c] for c in cls]
            raw = [model.names[c] for c in cls]
            tids = (
                res.boxes.id.cpu().numpy().astype(int).tolist()
                if res.boxes.id is not None
                else [None] * len(cls)
            )

        statuses = evaluate_frame(
            boxes, roles, tids, tuple(args.require), args.overlap, args.require_positive
        )

        # --- violation logging (edge-triggered) ---
        t_s = n_frames / fps
        seen = set()
        for st in statuses:
            seen.add(st.track_id)
            key = ",".join(st.missing)
            if not st.compliant:
                violation_frames[st.track_id] = violation_frames.get(st.track_id, 0) + 1
                if active_violation.get(st.track_id) != key:
                    active_violation[st.track_id] = key
                    x1, y1, x2, y2 = map(int, st.box)
                    log.writerow([n_frames, f"{t_s:.2f}", st.track_id, key, x1, y1, x2, y2])
            else:
                active_violation.pop(st.track_id, None)
        for tid in list(active_violation):
            if tid not in seen:
                active_violation.pop(tid)

        out = annotate(frame, boxes, roles, confs, raw, statuses, show_ppe=args.show_ppe)
        n_viol = sum(not s.compliant for s in statuses)
        hud = (f"frame {n_frames} | people {len(statuses)} | violations {n_viol} | "
               f"{n_frames / (time.time() - t0):.1f} fps")
        cv2.putText(out, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)

        if writer is None:
            h, w = out.shape[:2]
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(out)

        if args.show:
            cv2.imshow("PPE compliance", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer:
        writer.release()
    csv_f.close()
    cv2.destroyAllWindows()

    # --- summary ---
    print(f"\nDone. {n_frames} frames in {time.time() - t0:.1f}s")
    print(f"Annotated video : {video_path}")
    print(f"Violation log   : {csv_path}")
    if violation_frames:
        print("Frames-in-violation per track ID:")
        for tid, n in sorted(violation_frames.items(), key=lambda kv: -kv[1]):
            print(f"  ID {tid:4d}: {n} frames")
    else:
        print("No violations flagged." + ("" if have_ppe else " (expected: no PPE classes in model)"))


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video path or webcam index (0)")
    ap.add_argument("--weights", default="yolov8n.pt", help="model weights (.pt)")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml | botsort.yaml")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="0 for GPU, cpu for CPU (auto if omitted)")
    ap.add_argument("--require", nargs="+", default=["helmet", "vest"], help="PPE items that are mandatory")
    ap.add_argument("--overlap", type=float, default=0.5, help="min fraction of PPE box inside person box")
    ap.add_argument("--require-positive", action="store_true",
                    help="also flag when no positive helmet/vest box is found on the person")
    ap.add_argument("--show-ppe", action="store_true", help="draw raw helmet/vest boxes too")
    ap.add_argument("--show", action="store_true", help="live preview window (press q to quit)")
    ap.add_argument("--fps-assume", type=float, default=30.0, help="fps used when source has no fps (webcam)")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse())
