# PRD: Real-Time PPE Compliance Detection System

**Type:** Personal portfolio project (weekend build)
**Owner:** Purvansh
**Timeline:** 1–2 days
**Goal:** A resume-ready computer vision project demonstrating real-world detection, tracking, and video analytics skills — an end-to-end pipeline covering object detection, multi-object tracking, and rule-based decision logic on video.

---

## 1. Problem Statement

Industrial and energy-sector companies (like Tata Power) need to verify workers wear PPE (helmets, safety vests) in real time from CCTV/webcam feeds — manual monitoring doesn't scale across sites. This project builds a lightweight system that detects people, classifies PPE compliance, and tracks individuals across frames to flag violations.

## 2. Goals

- Detect people and PPE items (helmet, vest) in video with reasonable accuracy (target: mAP@0.5 > 0.6 on validation set — pretrained/fine-tuned model, not production-grade)
- Track each detected person across frames with a stable ID (no re-identification drift within a short clip)
- Flag and log frames/timestamps where a tracked person lacks required PPE
- Produce a demo (video output + short write-up) usable in interviews and on GitHub/resume

## 3. Non-Goals

- No real CCTV integration or live camera deployment — a recorded video/webcam demo is sufficient
- No custom-labeled dataset from scratch — use an existing public PPE dataset (labeling from scratch isn't feasible in 1–2 days)
- No multi-camera re-identification — single video stream only
- No mobile/edge deployment (Jetson, etc.) — this is a scope-creep trap, skip it
- No web app/API deployment for v1 — a script + demo video is enough; API wrapper is a stretch goal only if time remains

## 4. Scope (What You're Actually Building)

**Pipeline:** Video/webcam input → YOLOv8 detects `person`, `helmet`, `no-helmet`, `vest`, `no-vest` → ByteTrack assigns persistent IDs to each person → simple rule logic checks whether each tracked person's bounding box overlaps with a "no-helmet"/"no-vest" detection → violations logged with timestamp + track ID → annotated output video.

### Must-Have (Day 1)
- [ ] Working YOLOv8 inference on a pretrained/fine-tuned PPE detection model
- [ ] Object tracking layer (ByteTrack via `supervision` library or Ultralytics' built-in `track()` mode) giving stable IDs across frames
- [ ] Bounding boxes + labels + track IDs rendered on output video
- [ ] Basic violation rule: person detected without helmet/vest → flagged visually (red box) and logged to a CSV/console with timestamp + track ID

### Nice-to-Have (Day 2, if time allows)
- [ ] Simple Streamlit UI: upload video, see annotated output + violation log table
- [ ] Violation count summary (per person, per video)
- [ ] A short "confidence over time" chart per tracked ID (nice visual for the demo)

### Explicitly Cut (Future/Not This Build)
- Multi-camera tracking, live RTSP stream support, edge deployment, custom dataset labeling, model retraining from scratch — all out of scope; mention as "future work" in the writeup only.

## 5. Technical Plan

| Component | Choice | Why |
|---|---|---|
| Detection model | YOLOv8n or YOLOv8s (Ultralytics) | Fast to fine-tune, good docs, industry-standard for real-time detection |
| Dataset | Public "Hard Hat Detection" / "PPE Detection" dataset on Kaggle or Roboflow Universe | Pre-labeled, avoids the biggest time sink |
| Tracking | ByteTrack (via `supervision` pip package, or `model.track()` in Ultralytics) | Lightweight, no extra training needed |
| Language/libs | Python, OpenCV, Ultralytics, supervision, Pandas (for violation log) | Standard, resume-recognizable stack |
| Demo interface | CLI script → output .mp4 (Day 1); optional Streamlit (Day 2) | Keeps Day 1 unblocked; UI is a bonus, not a dependency |
| Compute | Google Colab (free GPU tier) or local CPU if dataset is small | No budget/hardware assumptions |

## 6. Day-by-Day Plan

**Day 1 (core pipeline):**
1. Set up environment, install `ultralytics`, `supervision`, `opencv-python` (30 min)
2. Download a pretrained PPE-detection YOLOv8 model from Roboflow Universe (many are shared pretrained — check licenses) OR fine-tune YOLOv8n on a small hard-hat dataset for a few epochs on Colab GPU (2–3 hrs)
3. Run inference on a sample video, get raw bounding boxes working (1 hr)
4. Add tracking (`model.track(source=video, tracker="bytetrack.yaml")`) and confirm stable IDs across frames (1–2 hrs)
5. Write violation logic: for each tracked person box, check IoU/overlap with "no-helmet"/"no-vest" detections; log violations to CSV with timestamp + track ID (1–2 hrs)

**Day 2 (polish + demo):**
1. Clean up annotated video output (color-coded boxes: green = compliant, red = violation) (1 hr)
2. Optional: wrap in a minimal Streamlit app for upload → process → display (2–3 hrs)
3. Write README with problem statement, approach, sample output GIF/screenshot, and metrics (1 hr)
4. Push to GitHub with a clear repo name (e.g., `ppe-compliance-detection`), record a 30–60 sec demo clip for LinkedIn/portfolio (1 hr)

## 7. Success Metrics (for you, not a company)

- Pipeline runs end-to-end on at least one sample video without crashing
- Visually correct detection + tracking on ≥80% of frames in a short test clip (eyeballed, not benchmarked rigorously)
- A README + demo video/GIF good enough to link directly from your resume/LinkedIn
- You can explain, in an interview, the full pipeline (detection → tracking → rule-based flagging) and name the specific components (YOLOv8, ByteTrack, IoU-based rule logic) without hesitation

## 8. Open Questions

- Which PPE dataset to use — needs a quick check for license terms if you plan to publish the repo publicly (you'll pick this at Day 1, Step 2)
- Whether to fine-tune or use an off-the-shelf pretrained PPE model — fine-tuning is more resume-credible ("I trained a model") but costs more Day 1 time; using pretrained + writing clearly about it is a safe fallback if time runs short

## 9. Resume Line (draft)

> Built a real-time PPE compliance detection system using YOLOv8 for object detection and ByteTrack for multi-object tracking; implemented rule-based violation flagging with per-person tracking IDs, achieving [X]% detection accuracy on a public hard-hat dataset.

(Fill in the actual mAP/accuracy number once you've measured it — don't leave a placeholder on the real resume.)
