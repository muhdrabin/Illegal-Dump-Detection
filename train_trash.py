import os
import json
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
from datetime import datetime
import argparse 

SAFE_BATCH = {
    'n': 32,
    's': 16,
    'm': 12,
    'l': 8,
    'x': 6,
}


class TrashModelTrainer:
    """Train a YOLOv8 model for trash detection."""

    def __init__(self, data_yaml_path, model_size='m', pretrained=True):
        self.data_yaml_path = data_yaml_path
        self.model_size = model_size
        self.pretrained = pretrained

        model_map = {
            'n': 'yolov8n.pt',
            's': 'yolov8s.pt',
            'm': 'yolov8m.pt',
            'l': 'yolov8l.pt',
            'x': 'yolov8x.pt',
        }

        if pretrained:
            self.model_name = model_map[model_size]
            print(f"  ✓ Pretrained weights : {self.model_name}")
        else:
            self.model_name = f'yolov8{model_size}.yaml'
            print(f"  ✓ Training from scratch : {self.model_name}")

        self.model = YOLO(self.model_name)
        print(f"  ✓ Model loaded successfully")

    # ── Training ─────────────────────────────────────────────
    def train(self,
              epochs=100,
              imgsz=640,
              batch=None,
              patience=20,
              device='0',
              workers=6,
              project='runs/trash_detection',
              name=None,
              **kwargs):

        # Use safe default batch if not specified
        if batch is None:
            batch = SAFE_BATCH[self.model_size]

        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f'trash_{self.model_size}_{timestamp}'

        print("\n" + "=" * 65)
        print("  STARTING TRAINING")
        print("=" * 65)
        print(f"  Model        : {self.model_name}")
        print(f"  Dataset      : {self.data_yaml_path}")
        print(f"  Epochs       : {epochs}")
        print(f"  Batch size   : {batch}")
        print(f"  Image size   : {imgsz}")
        print(f"  Patience     : {patience}")
        print(f"  Workers      : {workers}")
        print(f"  Device       : {device}")
        print(f"  Mixed prec.  : Enabled (amp=True)")
        print(f"  Output       : {project}/{name}")
        print("=" * 65 + "\n")

        results = self.model.train(
            data        = self.data_yaml_path,
            epochs      = epochs,
            imgsz       = imgsz,
            batch       = batch,
            patience    = patience,
            save        = True,
            device      = device,
            workers     = workers,
            project     = project,
            name        = name,
            exist_ok    = True,
            pretrained  = self.pretrained,
            verbose     = True,
            amp         = True,           # mixed precision — saves ~30% VRAM
            plots       = True,           # saves training curves, PR curve etc.
            optimizer   = 'auto',
            lr0         = 0.01,
            lrf         = 0.01,
            # ── Augmentation (tuned for outdoor surveillance trash) ──
            hsv_h       = 0.015,          # hue shift — keep small
            hsv_s       = 0.7,            # saturation
            hsv_v       = 0.4,            # brightness/exposure
            fliplr      = 0.5,            # horizontal flip 50%
            flipud      = 0.0,            # no vertical flip — trash is never upside-down
            degrees     = 15.0,           # rotation ±15°
            translate   = 0.1,            # random translate
            scale       = 0.2,            # random scale
            shear       = 5.0,            # slight shear
            perspective = 0.0005,         # slight perspective distortion
            mosaic      = 1.0,            # mosaic augmentation — great for small objects
            mixup       = 0.1,            # mild mixup
            copy_paste  = 0.1,            # copy-paste — helps with occlusion cases
            **kwargs
        )

        best_path = Path(project) / name / 'weights' / 'best.pt'

        print("\n" + "=" * 65)
        print("  TRAINING COMPLETE!")
        print(f"  ✓ Best model  : {best_path}")
        print(f"  ✓ Last model  : {Path(project) / name / 'weights' / 'last.pt'}")
        print(f"  ✓ Plots saved : {Path(project) / name}")
        print("=" * 65 + "\n")

        return results, str(best_path)

    # ── Validation ───────────────────────────────────────────
    def validate(self, model_path=None, save_json=True,
                 project='runs/trash_detection'):
        """
        Run validation and print full metrics.
        Optionally saves results to eval_results.json.
        """
        if model_path:
            self.model = YOLO(model_path)

        print("\n  Running validation...")
        metrics = self.model.val(
            data    = self.data_yaml_path,
            imgsz   = 640,
            device  = '0',
            plots   = True,
            verbose = True,
        )

        map50    = metrics.box.map50
        map5095  = metrics.box.map
        prec     = metrics.box.mp
        recall   = metrics.box.mr
        # F1 from precision and recall
        f1 = (2 * prec * recall / (prec + recall)) if (prec + recall) > 0 else 0.0

        print("\n" + "=" * 65)
        print("  VALIDATION RESULTS")
        print("=" * 65)
        print(f"  mAP@0.5          : {map50:.4f}  ({map50*100:.2f}%)")
        print(f"  mAP@0.5:0.95     : {map5095:.4f}  ({map5095*100:.2f}%)")
        print(f"  Precision        : {prec:.4f}  ({prec*100:.2f}%)")
        print(f"  Recall           : {recall:.4f}  ({recall*100:.2f}%)")
        print(f"  F1-Score         : {f1:.4f}  ({f1*100:.2f}%)")
        print("=" * 65 + "\n")

        if save_json:
            summary = {
                "model"       : model_path or "trained_model",
                "mAP50"       : round(float(map50), 6),
                "mAP50_95"    : round(float(map5095), 6),
                "precision"   : round(float(prec), 6),
                "recall"      : round(float(recall), 6),
                "f1"          : round(float(f1), 6),
            }
            out = Path(project) / "eval_results.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  ✓ Results saved : {out}")

        return metrics

# DATASET HELPERS

def create_dataset_yaml(dataset_root, class_names,
                        output_path='trash_dataset.yaml'):
    dataset_root = Path(dataset_root).absolute()

    config = {
        'path'  : str(dataset_root),
        'train' : 'train/images',
        'val'   : 'valid/images',
        'nc'    : len(class_names),
        'names' : class_names,
    }

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  ✓ Dataset YAML created : {output_path}")
    return output_path


def verify_dataset(dataset_root):
    dataset_root = Path(dataset_root)

    print("\n" + "=" * 65)
    print("  VERIFYING DATASET")
    print("=" * 65)

    required_dirs = [
        'train/images', 'train/labels',
        'valid/images', 'valid/labels',
    ]
    all_valid = True

    for dir_path in required_dirs:
        full_path = dataset_root / dir_path
        if full_path.exists():
            num_files = len(list(full_path.glob('*')))
            print(f"  ✓ {dir_path:22s} : {num_files} files")
        else:
            print(f"  ✗ {dir_path:22s} : NOT FOUND")
            all_valid = False

    # Image/label count mismatch warning
    for split in ['train', 'valid']:
        img_dir = dataset_root / split / 'images'
        lbl_dir = dataset_root / split / 'labels'
        if img_dir.exists() and lbl_dir.exists():
            imgs = len(list(img_dir.glob('*')))
            lbls = len(list(lbl_dir.glob('*.txt')))
            if imgs != lbls:
                print(f"  ⚠  {split}: {imgs} images but {lbls} labels — image/label count mismatch!")
                all_valid = False

    if all_valid:
        print("\n  ✓ Dataset looks good!")
    print("=" * 65 + "\n")
    return all_valid

# MAIN

def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 Trash Detection Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--dataset',      type=str, required=True,
                        help='Path to dataset root folder')
    parser.add_argument('--classes',      nargs='+', default=['trash'],
                        help='Class names (space separated)')
    parser.add_argument('--model-size',   choices=['n', 's', 'm', 'l', 'x'],
                        default='m', help='YOLOv8 model variant')
    parser.add_argument('--epochs',       type=int, default=100)
    parser.add_argument('--batch',        type=int, default=None,
                        help='Batch size (auto-selected if not set)')
    parser.add_argument('--patience',     type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--imgsz',        type=int, default=640)
    parser.add_argument('--device',       type=str, default='0',
                        help='CUDA device id, e.g. 0  (use cpu for CPU)')
    parser.add_argument('--workers',      type=int, default=6)
    parser.add_argument('--project',      type=str,
                        default='runs/trash_detection')
    parser.add_argument('--verify-only',  action='store_true',
                        help='Only verify dataset, do not train')
    parser.add_argument('--validate-only',action='store_true',
                        help='Only validate a saved checkpoint')
    parser.add_argument('--model',        type=str, default=None,
                        help='Checkpoint path for --validate-only')
    parser.add_argument('--no-pretrained',action='store_true',
                        help='Train from scratch (no ImageNet weights)')

    args = parser.parse_args()

    # ── Step 1: create YAML ──────────────────────────────────
    yaml_path = create_dataset_yaml(args.dataset, args.classes)

    # ── Step 2: verify dataset ───────────────────────────────
    valid = verify_dataset(args.dataset)
    if not valid and not args.verify_only:
        response = input("  Dataset has issues. Continue anyway? (yes/no): ")
        if response.strip().lower() != 'yes':
            return

    if args.verify_only:
        return

    # ── Step 3: build trainer ────────────────────────────────
    trainer = TrashModelTrainer(
        data_yaml_path = yaml_path,
        model_size     = args.model_size,
        pretrained     = not args.no_pretrained,
    )

    # ── Step 4: validate-only mode ───────────────────────────
    if args.validate_only:
        if not args.model:
            print("  ✗ Error: --model path is required with --validate-only")
            return
        trainer.validate(model_path=args.model, project=args.project)
        return

    # ── Step 5: confirm and train ────────────────────────────
    batch = args.batch if args.batch else SAFE_BATCH[args.model_size]
    print(f"\n  Ready to train YOLOv8{args.model_size} | "
          f"batch={batch} | epochs={args.epochs} | patience={args.patience}")
    response = input("  Start training? (yes/no): ")
    if response.strip().lower() != 'yes':
        return

    results, best_path = trainer.train(
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = batch,
        patience = args.patience,
        device   = args.device,
        workers  = args.workers,
        project  = args.project,
    )

    # ── Step 6: auto-validate best model ────────────────────
    print("  Running final validation on best model...")
    trainer.validate(model_path=best_path, project=args.project)


if __name__ == '__main__':
    main()
