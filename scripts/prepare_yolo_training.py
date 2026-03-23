#!/usr/bin/env python3
"""Generate YOLO training data from existing SAM detection results.

Uses the masks and bounding boxes already produced by Colony_detection_SAM
to create YOLO-format annotations for training a custom colony detector.

Usage:
    python scripts/prepare_yolo_training.py \
        --results-dir results/ \
        --images-dir Image_input/ \
        --output-dir yolo_dataset/

Produces:
    yolo_dataset/
      images/train/  images/val/
      labels/train/  labels/val/
      dataset.yaml
"""

import argparse
import json
import logging
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def masks_to_yolo_labels(
    detailed_results: list[dict],
    img_width: int,
    img_height: int,
    class_id: int = 0,
) -> list[str]:
    """Convert colony detection results to YOLO format labels.

    YOLO format: <class> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1].
    """
    labels = []
    for colony in detailed_results:
        bbox = colony.get("bbox")
        if not bbox:
            # Try to compute from contour or area
            x = colony.get("centroid_x", 0)
            y = colony.get("centroid_y", 0)
            area = colony.get("area", 0)
            if area > 0:
                side = int(area ** 0.5)
                bbox = [x - side // 2, y - side // 2, x + side // 2, y + side // 2]
            else:
                continue

        if isinstance(bbox, dict):
            x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
            x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
        else:
            continue

        # Normalize
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        w = abs(x2 - x1) / img_width
        h = abs(y2 - y1) / img_height

        # Clamp
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w = max(0.001, min(1, w))
        h = max(0.001, min(1, h))

        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return labels


def collect_paired_data(results_dir: Path, images_dir: Path) -> list[dict]:
    """Walk results directory and find matching image files."""
    pairs = []
    for json_file in results_dir.rglob("detailed_results.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if not isinstance(data, list) or len(data) == 0:
                continue

            # Find corresponding image
            view_dir = json_file.parent.parent  # e.g. .../Front
            annotated = list(view_dir.glob("annotated_*.png")) + list(view_dir.glob("annotated_*.jpg"))

            # Also look for the original image in images_dir
            # Try to reconstruct filename from path
            parts = json_file.parts
            img_candidates = list(images_dir.rglob("*.jpg")) + list(images_dir.rglob("*.png"))

            image_path = None
            if annotated:
                image_path = annotated[0]
            elif img_candidates:
                # Match by sample name
                for candidate in img_candidates:
                    for part in parts:
                        if part.lower() in candidate.stem.lower():
                            image_path = candidate
                            break
                    if image_path:
                        break

            if image_path and image_path.exists():
                pairs.append({
                    "image": image_path,
                    "results": data,
                    "json_path": json_file,
                })
        except Exception as e:
            logger.warning(f"Skipping {json_file}: {e}")

    logger.info(f"Found {len(pairs)} image-result pairs")
    return pairs


def create_yolo_dataset(
    pairs: list[dict],
    output_dir: Path,
    val_split: float = 0.2,
):
    """Create YOLO-format dataset from paired data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_split))
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    total = 0
    for split_name, split_pairs in splits.items():
        for pair in split_pairs:
            img = cv2.imread(str(pair["image"]))
            if img is None:
                continue
            h, w = img.shape[:2]

            labels = masks_to_yolo_labels(pair["results"], w, h)
            if not labels:
                continue

            # Copy image
            idx = total
            img_name = f"colony_{idx:04d}.jpg"
            lbl_name = f"colony_{idx:04d}.txt"

            cv2.imwrite(str(output_dir / "images" / split_name / img_name), img)
            (output_dir / "labels" / split_name / lbl_name).write_text("\n".join(labels))
            total += 1

    # Write dataset.yaml
    yaml_content = f"""# Colony Detection YOLO Dataset
# Generated from Colony_detection_SAM results

path: {output_dir.resolve()}
train: images/train
val: images/val

names:
  0: colony

nc: 1
"""
    (output_dir / "dataset.yaml").write_text(yaml_content)
    logger.info(f"Created YOLO dataset: {total} images ({len(splits['train'])} train, {len(splits['val'])} val)")
    return total


def main():
    parser = argparse.ArgumentParser(description="Generate YOLO training data from SAM results")
    parser.add_argument("--results-dir", "-r", required=True, help="Colony_detection_SAM results directory")
    parser.add_argument("--images-dir", "-i", required=True, help="Original images directory")
    parser.add_argument("--output-dir", "-o", default="yolo_dataset", help="Output YOLO dataset directory")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    pairs = collect_paired_data(Path(args.results_dir), Path(args.images_dir))
    if not pairs:
        print("No paired data found. Check your results and images directories.")
        return

    n = create_yolo_dataset(pairs, Path(args.output_dir), args.val_split)
    print(f"\n✅ YOLO dataset created: {n} images in {args.output_dir}/")
    print(f"   To train: yolo train data={args.output_dir}/dataset.yaml model=yolov8n.pt epochs=100 imgsz=1280")


if __name__ == "__main__":
    main()
