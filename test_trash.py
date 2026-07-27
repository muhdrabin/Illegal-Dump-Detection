"""
YOLOv8 Trash Detection Model — Testing & Evaluation Script

Modes
-----
  image     : Run detection on a single image
  folder    : Batch inference on all images in a folder
  video     : Run detection on a video file
  webcam    : Live detection from webcam
  evaluate  : Full mAP / Precision / Recall / F1 on validation set

Usage examples
--------------
  python test_trash.py --model best.pt --mode image  --input photo.jpg
  python test_trash.py --model best.pt --mode folder --input ./test_images/
  python test_trash.py --model best.pt --mode video  --input clip.mp4 --output out.mp4
  python test_trash.py --model best.pt --mode webcam
  python test_trash.py --model best.pt --mode evaluate --data trash_dataset.yaml
"""

import cv2
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────
# SUPPORTED IMAGE EXTENSIONS
# ─────────────────────────────────────────────────────────────
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


# ─────────────────────────────────────────────────────────────
# DETECTOR CLASS
# ─────────────────────────────────────────────────────────────
class TrashDetector:
    """Inference and evaluation wrapper for a trained YOLOv8 trash model."""

    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"\n  Loading model : {model_path}")
        self.model         = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.class_names    = self.model.names  # dict {0: 'trash', ...}

        print(f"  ✓ Classes     : {list(self.class_names.values())}")
        print(f"  ✓ Conf thresh : {conf_threshold}")
        print(f"  ✓ IoU  thresh : {iou_threshold}\n")

    # ── Color per class ──────────────────────────────────────
    @staticmethod
    def get_class_color(class_id):
        palette = [
            (0, 200, 80),    # green
            (255, 80, 80),   # red
            (80, 120, 255),  # blue
            (255, 200, 0),   # yellow
            (200, 0, 255),   # purple
            (0, 220, 220),   # cyan
        ]
        return palette[class_id % len(palette)]

    # ── Single-frame detection ───────────────────────────────
    def detect(self, image):
        """
        Run inference on a single BGR image (numpy array).
        Returns list of dicts: {bbox, class, class_name, confidence}
        """
        results = self.model(
            image,
            conf    = self.conf_threshold,
            iou     = self.iou_threshold,
            verbose = False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    'bbox'      : (x1, y1, x2, y2),
                    'class'     : cls,
                    'class_name': self.class_names[cls],
                    'confidence': conf,
                })
        return detections

    # ── Draw bounding boxes ──────────────────────────────────
    def draw_detections(self, image, detections, fps=None):
        """Draw boxes, labels, and optional FPS overlay on image."""
        annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color  = self.get_class_color(det['class'])
            label  = f"{det['class_name']}  {det['confidence']:.2f}"

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y1 = max(y1 - th - 10, 0)
            cv2.rectangle(annotated,
                          (x1, label_y1), (x1 + tw + 4, label_y1 + th + 8),
                          color, -1)
            cv2.putText(annotated, label, (x1 + 2, label_y1 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)

        # FPS overlay
        if fps is not None:
            fps_label = f"FPS: {fps:.1f}"
            cv2.putText(annotated, fps_label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                        cv2.LINE_AA)

        return annotated

    # ── MODE 1: Single image ─────────────────────────────────
    def test_image(self, image_path, output_path=None, display=True):
        image_path = Path(image_path)
        print(f"  Processing : {image_path.name}")

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  ✗ Could not read image: {image_path}")
            return

        t0         = time.perf_counter()
        detections = self.detect(image)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        annotated  = self.draw_detections(image, detections)

        print(f"  ✓ Detections : {len(detections)}  ({elapsed_ms:.1f} ms)")
        for det in detections:
            print(f"     - {det['class_name']} : {det['confidence']:.4f}")

        if output_path is None:
            output_path = f"detected_{image_path.name}"
        cv2.imwrite(str(output_path), annotated)
        print(f"  ✓ Saved      : {output_path}")

        if display:
            cv2.imshow('Trash Detection', annotated)
            print("  (Press any key to close)")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # ── MODE 2: Folder batch inference ───────────────────────
    def test_folder(self, folder_path, output_dir=None, display=False):
        folder_path = Path(folder_path)
        if not folder_path.exists():
            print(f"  ✗ Folder not found: {folder_path}")
            return

        images = sorted([p for p in folder_path.rglob('*')
                         if p.suffix.lower() in IMG_EXTS])
        if not images:
            print(f"  ✗ No images found in {folder_path}")
            return

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Found {len(images)} images — running batch inference...\n")
        print(f"  {'Image':<40} {'Detections':>12} {'Time (ms)':>10}")
        print("  " + "-" * 65)

        results_log   = []
        total_dets    = 0
        total_time_ms = 0.0

        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  [SKIP] {img_path.name} — unreadable")
                continue

            t0         = time.perf_counter()
            detections = self.detect(image)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            total_dets    += len(detections)
            total_time_ms += elapsed_ms

            print(f"  {img_path.name:<40} {len(detections):>12} {elapsed_ms:>9.1f}ms")

            annotated = self.draw_detections(image, detections)

            if output_dir:
                out = output_dir / f"detected_{img_path.name}"
                cv2.imwrite(str(out), annotated)

            if display:
                cv2.imshow('Batch Detection', annotated)
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break

            results_log.append({
                'file'      : str(img_path),
                'detections': len(detections),
                'time_ms'   : round(elapsed_ms, 2),
                'objects'   : [
                    {'class': d['class_name'], 'confidence': round(d['confidence'], 4)}
                    for d in detections
                ],
            })

        if display:
            cv2.destroyAllWindows()

        avg_ms = total_time_ms / max(len(results_log), 1)
        print("  " + "-" * 65)
        print(f"\n  ✓ Images processed : {len(results_log)}")
        print(f"  ✓ Total detections : {total_dets}")
        print(f"  ✓ Avg time/image   : {avg_ms:.1f} ms  ({1000/avg_ms:.1f} img/sec)")

        log_path = (output_dir / 'batch_results.json') if output_dir \
                   else Path('batch_results.json')
        with open(log_path, 'w') as f:
            json.dump(results_log, f, indent=2)
        print(f"  ✓ Results log      : {log_path}")

    # ── MODE 3: Video ────────────────────────────────────────
    def test_video(self, video_path, output_path=None, display=True):
        video_path = Path(video_path)
        print(f"  Processing video : {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  ✗ Could not open video: {video_path}")
            return

        src_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"  Resolution : {width}x{height} @ {src_fps} FPS | {total} frames")

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_path), fourcc, src_fps, (width, height))

        frame_count    = 0
        total_dets     = 0
        fps_timer      = time.perf_counter()
        display_fps    = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Calculate actual display FPS every 10 frames
            if frame_count % 10 == 0:
                now         = time.perf_counter()
                display_fps = 10.0 / (now - fps_timer + 1e-9)
                fps_timer   = now

            detections  = self.detect(frame)
            total_dets += len(detections)
            annotated   = self.draw_detections(frame, detections, fps=display_fps)

            # Progress overlay
            progress = f"Frame {frame_count}/{total} | Dets: {len(detections)}"
            cv2.putText(annotated, progress, (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2,
                        cv2.LINE_AA)

            if writer:
                writer.write(annotated)

            if display:
                cv2.imshow('Trash Detection — Video', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("  Stopped by user.")
                    break

            if frame_count % 30 == 0:
                pct = frame_count / max(total, 1) * 100
                print(f"  {pct:5.1f}%  frame {frame_count}/{total}"
                      f"  |  FPS: {display_fps:.1f}"
                      f"  |  dets: {len(detections)}")

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        print(f"\n  ✓ Frames processed : {frame_count}")
        print(f"  ✓ Total detections : {total_dets}")
        if output_path:
            print(f"  ✓ Saved video      : {output_path}")

    # ── MODE 4: Webcam ───────────────────────────────────────
    def test_webcam(self, camera_id=0):
        print(f"\n  Starting webcam (id={camera_id})")
        print("  Controls: Q = quit | S = screenshot\n")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"  ✗ Could not open camera {camera_id}")
            return

        frame_count = 0
        fps_timer   = time.perf_counter()
        display_fps = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("  ✗ Camera read failed.")
                break

            frame_count += 1

            # Update FPS every 10 frames
            if frame_count % 10 == 0:
                now         = time.perf_counter()
                display_fps = 10.0 / (now - fps_timer + 1e-9)
                fps_timer   = now

            detections = self.detect(frame)
            annotated  = self.draw_detections(frame, detections, fps=display_fps)

            # Hint overlay
            hint = f"Dets: {len(detections)}  |  Q:Quit  S:Screenshot"
            cv2.putText(annotated, hint,
                        (10, annotated.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2,
                        cv2.LINE_AA)

            cv2.imshow('Trash Detection — Webcam', annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"screenshot_{ts}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"  ✓ Screenshot saved: {filename}")

        cap.release()
        cv2.destroyAllWindows()

    # ── MODE 5: Evaluate (mAP / Precision / Recall / F1) ────
    def evaluate(self, data_yaml, output_dir='runs/trash_detection/eval'):
        """
        Full evaluation on the validation split.
        Prints and saves mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1.
        """
        if not Path(data_yaml).exists():
            print(f"  ✗ data yaml not found: {data_yaml}")
            return

        print(f"  Running evaluation on : {data_yaml}")
        print("  (This may take a minute...)\n")

        metrics = self.model.val(
            data    = data_yaml,
            imgsz   = 640,
            plots   = True,
            verbose = True,
            project = output_dir,
            name    = 'results',
            exist_ok= True,
        )

        map50   = float(metrics.box.map50)
        map5095 = float(metrics.box.map)
        prec    = float(metrics.box.mp)
        recall  = float(metrics.box.mr)
        f1      = (2 * prec * recall / (prec + recall)) \
                  if (prec + recall) > 0 else 0.0

        print("\n" + "=" * 55)
        print("  EVALUATION RESULTS")
        print("=" * 55)
        print(f"  mAP @ 0.5        : {map50:.4f}  ({map50*100:.2f}%)")
        print(f"  mAP @ 0.5:0.95   : {map5095:.4f}  ({map5095*100:.2f}%)")
        print(f"  Precision        : {prec:.4f}  ({prec*100:.2f}%)")
        print(f"  Recall           : {recall:.4f}  ({recall*100:.2f}%)")
        print(f"  F1-Score         : {f1:.4f}  ({f1*100:.2f}%)")
        print("=" * 55)

        # Save JSON summary
        summary = {
            "data_yaml" : data_yaml,
            "mAP50"     : round(map50, 6),
            "mAP50_95"  : round(map5095, 6),
            "precision" : round(prec, 6),
            "recall"    : round(recall, 6),
            "f1"        : round(f1, 6),
        }
        out_dir = Path(output_dir) / 'results'
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / 'eval_summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  ✓ Summary saved  : {json_path}")
        print(f"  ✓ Plots saved    : {out_dir}")

        return metrics


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='YOLOv8 Trash Detection — Test & Evaluate',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--model',      required=True,
                        help='Path to trained model (best.pt)')
    parser.add_argument('--mode',       required=True,
                        choices=['image', 'folder', 'video', 'webcam', 'evaluate'],
                        help='Inference mode')
    parser.add_argument('--input',      default=None,
                        help='Input path (image / folder / video)')
    parser.add_argument('--output',     default=None,
                        help='Output path (image / folder / video)')
    parser.add_argument('--data',       default=None,
                        help='Dataset YAML path (required for evaluate mode)')
    parser.add_argument('--camera',     type=int, default=0,
                        help='Camera ID for webcam mode')
    parser.add_argument('--conf',       type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--iou',        type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable OpenCV window (headless mode)')

    args = parser.parse_args()

    # ── Argument validation ──────────────────────────────────
    if args.mode in ('image', 'folder', 'video') and not args.input:
        parser.error(f'--input is required for mode "{args.mode}"')

    if args.mode == 'evaluate' and not args.data:
        parser.error('--data (dataset YAML path) is required for evaluate mode')

    # ── Build detector ───────────────────────────────────────
    detector = TrashDetector(
        model_path     = args.model,
        conf_threshold = args.conf,
        iou_threshold  = args.iou,
    )

    display = not args.no_display

    # ── Dispatch mode ────────────────────────────────────────
    if args.mode == 'image':
        out = args.output or f"detected_{Path(args.input).name}"
        detector.test_image(args.input, out, display)

    elif args.mode == 'folder':
        detector.test_folder(args.input, args.output, display)

    elif args.mode == 'video':
        detector.test_video(args.input, args.output, display)

    elif args.mode == 'webcam':
        detector.test_webcam(args.camera)

    elif args.mode == 'evaluate':
        detector.evaluate(args.data)


if __name__ == '__main__':
    main()
