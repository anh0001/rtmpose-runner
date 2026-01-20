#!/usr/bin/env python3
"""
RTMPose-X Inference Script using MMPose Inferencer

This script runs RTMPose-X model on images in the dataset folder and saves:
1. JSON predictions with keypoint coordinates
2. Visualizations with pose overlays
3. Performance metrics and statistics
"""

# Suppress verbose output from MMCV and MMPose
import os
os.environ['MMENGINE_LOG_LEVEL'] = 'WARNING'

# Apply MMCV extension fix before importing mmpose
try:
    import mmcv_fix  # noqa: F401
except ImportError:
    pass

import json
import time
from pathlib import Path
import argparse
from datetime import datetime
import logging

import cv2
import numpy as np
try:
    import yaml
except Exception:
    yaml = None

# Suppress verbose logging from MMPose and MMCV
logging.getLogger('mmengine').setLevel(logging.WARNING)
logging.getLogger('mmpose').setLevel(logging.WARNING)
logging.getLogger('mmcv').setLevel(logging.WARNING)
logging.getLogger('mmdet').setLevel(logging.WARNING)

from mmpose.apis import MMPoseInferencer

RTMPOSE_X_CONFIG = 'rtmpose-x_8xb32-270e_coco-wholebody-384x288'
RTMPOSE_X_WEIGHTS = (
    'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/'
    'rtmpose-x_simcc-coco-wholebody_pt-body7_270e-384x288-401dfc90_20230629.pth'
)


def _find_mmpose_config(relative_path):
    try:
        import mmpose
    except Exception:
        return None

    package_path = Path(mmpose.__path__[0]).resolve()
    repo_configs = package_path.parent / 'configs'
    if repo_configs.is_dir():
        candidate = repo_configs / relative_path
        if candidate.is_file():
            return str(candidate)

    mim_configs = package_path / '.mim' / 'configs'
    candidate = mim_configs / relative_path
    if candidate.is_file():
        return str(candidate)

    return None


def resolve_model_config_and_weights(model_config, pose2d_weights):
    if pose2d_weights:
        return model_config, pose2d_weights

    if Path(model_config).is_file():
        return model_config, pose2d_weights

    if model_config == RTMPOSE_X_CONFIG:
        config_path = _find_mmpose_config(
            'wholebody_2d_keypoint/rtmpose/coco-wholebody/'
            'rtmpose-x_8xb32-270e_coco-wholebody-384x288.py'
        )
        if config_path:
            return config_path, RTMPOSE_X_WEIGHTS

    return model_config, pose2d_weights


def _collect_images_in_dir(image_dir):
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(image_dir.rglob(ext)))
    return [str(f) for f in image_files]


def _load_yolo_dataset_config(dataset_dir):
    if yaml is None:
        return None

    dataset_dir = Path(dataset_dir)
    for name in ("dataset.yaml", "data.yaml"):
        yaml_path = dataset_dir / name
        if yaml_path.is_file():
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                data["_config_path"] = yaml_path
                return data
    return None


def _read_split_list(list_path, dataset_root):
    paths = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entry = Path(line)
            if not entry.is_absolute():
                entry = dataset_root / entry
            paths.append(str(entry))
    return paths


def _get_yolo_image_files(dataset_dir, split):
    dataset_config = _load_yolo_dataset_config(dataset_dir)
    if not dataset_config:
        return None

    config_path = dataset_config.get("_config_path")
    if split not in dataset_config:
        available = [key for key in ("train", "val", "test") if key in dataset_config]
        raise ValueError(
            f"Split '{split}' not found in {config_path}. Available splits: {available}"
        )

    dataset_root = Path(dataset_dir)
    dataset_path = dataset_config.get("path")
    if dataset_path:
        dataset_root = (dataset_root / dataset_path).resolve()

    split_entry = dataset_config.get(split)
    entries = split_entry if isinstance(split_entry, (list, tuple)) else [split_entry]

    image_files = []
    for entry in entries:
        if entry is None:
            continue
        entry_path = Path(str(entry))
        if not entry_path.is_absolute():
            entry_path = dataset_root / entry_path

        if entry_path.is_dir():
            image_files.extend(_collect_images_in_dir(entry_path))
            continue

        if entry_path.is_file():
            if entry_path.suffix.lower() == ".txt":
                image_files.extend(_read_split_list(entry_path, dataset_root))
            else:
                image_files.append(str(entry_path))
            continue

        entry_str = str(entry)
        if any(token in entry_str for token in ["*", "?", "[", "]"]):
            image_files.extend(
                [str(p) for p in dataset_root.glob(entry_str) if p.is_file()]
            )

    return sorted(set(image_files))


def get_image_files(dataset_dir, split='val'):
    """Get list of image files from dataset directory."""
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    yolo_images = _get_yolo_image_files(dataset_dir, split)
    if yolo_images is not None:
        return yolo_images

    yolo_images_dir = dataset_dir / "images" / split
    if yolo_images_dir.is_dir():
        return sorted(_collect_images_in_dir(yolo_images_dir))

    split_dir = dataset_dir / split

    if not split_dir.exists():
        raise ValueError(f"Dataset split '{split}' not found at {split_dir}")

    return sorted(_collect_images_in_dir(split_dir))


def get_image_files_from_zoo(
    zoo_dataset,
    zoo_split='validation',
    dataset_name=None,
    max_samples=None,
    persistent=False,
):
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except Exception as exc:
        raise RuntimeError("FiftyOne is required to load zoo datasets") from exc

    if dataset_name and dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(dataset_name)
    else:
        dataset = foz.load_zoo_dataset(
            zoo_dataset,
            split=zoo_split,
            dataset_name=dataset_name,
            persistent=persistent,
            max_samples=max_samples,
        )

    return sorted(dataset.values("filepath"))


def run_inference(
    model_config='rtmpose-x_8xb32-270e_coco-wholebody-384x288',
    pose2d_weights=None,
    dataset_dir='dataset/polar',
    split='val',
    output_dir='outputs',
    zoo_dataset=None,
    zoo_split='validation',
    zoo_dataset_name=None,
    zoo_max_samples=None,
    detection_mode='whole_image',  # or 'person_detection'
    batch_size=1,
    device='cuda:0',
    visualize=True
):
    """
    Run RTMPose-X inference on dataset.

    Args:
        model_config: RTMPose model config name (e.g., 'rtmpose-x_8xb32-270e_coco-wholebody-384x288', 'rtmpose-l', 'rtmpose-m')
        pose2d_weights: Optional checkpoint path/URL for pose model weights
        dataset_dir: Path to dataset directory (supports YOLO-style datasets)
        split: Dataset split to use ('train', 'val', or 'test')
        output_dir: Directory to save outputs
        detection_mode: 'whole_image' (no detector, for single person) or 'person_detection'
        batch_size: Batch size for inference
        device: Device to use ('cuda:0' or 'cpu')
        visualize: Whether to save visualization images
    """

    print("="*60)
    print("RTMPose-X Inference Configuration")
    print("="*60)
    print(f"Model: {model_config}")
    if zoo_dataset:
        print(f"Dataset: zoo/{zoo_dataset}:{zoo_split}")
    else:
        print(f"Dataset: {dataset_dir}/{split}")
    print(f"Detection mode: {detection_mode}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Visualize: {visualize}")
    print("="*60)

    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    vis_dir = output_dir / 'visualizations'
    pred_dir = output_dir / 'predictions'
    vis_dir.mkdir(exist_ok=True)
    pred_dir.mkdir(exist_ok=True)

    # Get image files
    results_split = zoo_split if zoo_dataset else split
    print("\nLoading image files...")
    if zoo_dataset:
        image_files = get_image_files_from_zoo(
            zoo_dataset,
            zoo_split=zoo_split,
            dataset_name=zoo_dataset_name,
            max_samples=zoo_max_samples,
        )
    else:
        image_files = get_image_files(dataset_dir, split)
    print(f"Found {len(image_files)} images in {results_split} split")

    if len(image_files) == 0:
        print("ERROR: No images found! Please check your dataset directory.")
        return

    # Initialize MMPose Inferencer
    resolved_model_config, resolved_pose2d_weights = resolve_model_config_and_weights(
        model_config, pose2d_weights
    )
    if resolved_model_config != model_config:
        print(f"Resolved model config: {resolved_model_config}")
    if resolved_pose2d_weights:
        print("Using explicit pose2d weights.")

    print(f"\nInitializing {resolved_model_config} inferencer...")

    # Configure detector based on detection_mode
    if detection_mode == 'whole_image':
        # Skip detector and use a whole-image bbox per frame.
        det_model = None
        det_cat_ids = None
        print("Using whole_image mode (no detector, whole-image bboxes)")
    else:
        # Use default person detector
        det_model = 'rtmdet-m'  # or 'faster-rcnn' for better accuracy
        det_cat_ids = [0]  # COCO person category
        print(f"Using person detection mode with {det_model}")

    inferencer_kwargs = {
        'pose2d': resolved_model_config,
        'pose2d_weights': resolved_pose2d_weights,
        'device': device
    }
    if det_model is None:
        inferencer_kwargs.update({
            'det_model': None,
            'det_weights': None,
            'det_cat_ids': None
        })
    else:
        inferencer_kwargs.update({
            'det_model': det_model,
            'det_weights': None,
            'det_cat_ids': det_cat_ids
        })

    inferencer = MMPoseInferencer(**inferencer_kwargs)

    print("Inferencer initialized successfully!")

    # Run inference
    print(f"\nRunning inference on {len(image_files)} images...")
    print("This may take a while...")

    all_results = []
    start_time = time.time()

    for idx, img_path in enumerate(image_files):
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (len(image_files) - idx - 1)
            print(f"Progress: {idx+1}/{len(image_files)} images "
                  f"({100*(idx+1)/len(image_files):.1f}%) - "
                  f"Est. remaining: {remaining/60:.1f} min")

        # Prepare output paths
        img_name = Path(img_path).stem
        out_vis = str(vis_dir / f"{img_name}_pose.jpg") if visualize else None
        out_pred = str(pred_dir / f"{img_name}.json")

        # Run inference on single image
        try:
            if detection_mode == 'whole_image':
                # In whole_image mode, create a bounding box for the entire image
                import cv2
                img = cv2.imread(img_path)
                if img is None:
                    print(f"\nERROR: Could not read image {img_path}")
                    continue
                h, w = img.shape[:2]

                # Create bounding box for entire image with format [x, y, x, y, score]
                bboxes = [[0, 0, w, h, 1.0]]

                result_generator = inferencer(
                    img_path,
                    bboxes=bboxes,
                    show=False,
                    out_dir=str(vis_dir) if visualize else None,
                    pred_out_dir=str(pred_dir),
                    return_vis=False
                )
            else:
                # Use person detection
                result_generator = inferencer(
                    img_path,
                    show=False,
                    out_dir=str(vis_dir) if visualize else None,
                    pred_out_dir=str(pred_dir),
                    return_vis=False
                )

            # Get results
            result = next(result_generator)

            # Store result with metadata
            result_data = {
                'image_path': img_path,
                'image_name': img_name,
                'predictions': result['predictions']
            }
            all_results.append(result_data)

        except Exception as e:
            print(f"\nERROR processing {img_path}: {e}")
            continue

    # Calculate statistics
    total_time = time.time() - start_time
    avg_time_per_image = total_time / len(image_files)
    fps = 1.0 / avg_time_per_image if avg_time_per_image > 0 else 0

    # Save all results
    results_file = output_dir / f'all_results_{results_split}.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate performance report
    print("\n" + "="*60)
    print("INFERENCE COMPLETED!")
    print("="*60)
    print(f"Total images processed: {len(all_results)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time_per_image*1000:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Predictions: {pred_dir}")
    if visualize:
        print(f"  - Visualizations: {vis_dir}")
    print(f"  - All results JSON: {results_file}")
    print("="*60)

    # Save performance metrics
    metrics = {
        'model': model_config,
        'resolved_model_config': resolved_model_config,
        'pose2d_weights': resolved_pose2d_weights,
        'dataset_split': results_split,
        'total_images': len(all_results),
        'total_time_seconds': total_time,
        'avg_time_per_image_ms': avg_time_per_image * 1000,
        'fps': fps,
        'device': device,
        'detection_mode': detection_mode,
        'timestamp': datetime.now().isoformat()
    }

    metrics_file = output_dir / f'performance_metrics_{results_split}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nPerformance metrics saved to: {metrics_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run RTMPose-X inference on dataset')
    parser.add_argument('--model', type=str, default='rtmpose-x_8xb32-270e_coco-wholebody-384x288',
                        help='RTMPose model config name (e.g., rtmpose-x_8xb32-270e_coco-wholebody-384x288 for wholebody, rtmpose-l_8xb256-420e_aic-coco-384x288 for body-only)')
    parser.add_argument('--pose2d-weights', type=str, default=None,
                        help='Optional pose2d checkpoint path/URL. Useful for models not in the meta index.')
    parser.add_argument('--dataset-dir', type=str, default='dataset/polar',
                        help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to use')
    parser.add_argument('--zoo-dataset', type=str, default=None,
                        help='FiftyOne zoo dataset name (overrides dataset-dir/split)')
    parser.add_argument('--zoo-split', type=str, default='validation',
                        help='FiftyOne zoo dataset split')
    parser.add_argument('--zoo-dataset-name', type=str, default=None,
                        help='FiftyOne dataset name to reuse for zoo loading')
    parser.add_argument('--zoo-max-samples', type=int, default=None,
                        help='Maximum number of zoo samples to load')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory for results')
    parser.add_argument('--detection-mode', type=str, default='whole_image',
                        choices=['whole_image', 'person_detection'],
                        help='Detection mode: whole_image (default, no detector) or person_detection (uses RTMDet)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0 or cpu)')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Disable visualization saving')

    args = parser.parse_args()

    run_inference(
        model_config=args.model,
        pose2d_weights=args.pose2d_weights,
        dataset_dir=args.dataset_dir,
        split=args.split,
        output_dir=args.output_dir,
        zoo_dataset=args.zoo_dataset,
        zoo_split=args.zoo_split,
        zoo_dataset_name=args.zoo_dataset_name,
        zoo_max_samples=args.zoo_max_samples,
        detection_mode=args.detection_mode,
        batch_size=args.batch_size,
        device=args.device,
        visualize=not args.no_visualize
    )


if __name__ == '__main__':
    main()
