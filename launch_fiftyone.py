#!/usr/bin/env python3
"""
FiftyOne Dataset Viewer for RTMPose Results

This script launches FiftyOne to visualize:
1. Images from the dataset
2. RTMPose predictions (keypoints)
3. Ground truth annotations (if available)
4. Interactive exploration and filtering
"""

import json
import argparse
import signal
import sys
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

COCO_KEYPOINT_LABELS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_SKELETON_EDGES = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [5, 6],
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],
    [5, 11],
    [6, 12],
    [11, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
]

COCO_FOOT_KEYPOINT_LABELS = [
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
]


def load_rtmpose_predictions(pred_dir):
    """Load RTMPose predictions from JSON files."""
    pred_dir = Path(pred_dir)
    predictions = {}

    if not pred_dir.exists():
        print(f"Warning: Predictions directory not found: {pred_dir}")
        return predictions

    for json_file in pred_dir.glob("*.json"):
        image_name = json_file.stem
        try:
            with open(json_file, 'r') as f:
                pred_data = json.load(f)
                if isinstance(pred_data, dict):
                    pred_data = (
                        pred_data.get("predictions")
                        or pred_data.get("instances")
                        or ([pred_data] if "keypoints" in pred_data else pred_data)
                    )
                predictions[image_name] = pred_data
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return predictions


def _is_yolo_dataset_dir(dataset_dir):
    dataset_dir = Path(dataset_dir)
    for name in ("dataset.yaml", "data.yaml"):
        if (dataset_dir / name).is_file():
            return True
    return False


def build_coco_skeleton(include_labels=True):
    labels = COCO_KEYPOINT_LABELS if include_labels else None
    return fo.KeypointSkeleton(
        labels=labels,
        edges=COCO_SKELETON_EDGES,
    )


def _build_face_edges(offset):
    edges = []

    def line(start, end):
        edges.extend([[i + offset, i + 1 + offset] for i in range(start, end)])

    def loop(start, count):
        edges.extend([[start + i + offset, start + i + 1 + offset]
                      for i in range(count - 1)])
        edges.append([start + count - 1 + offset, start + offset])

    # Jawline 0-16
    line(0, 16)
    # Eyebrows 17-21 and 22-26
    line(17, 21)
    line(22, 26)
    # Nose bridge 27-30 and nose bottom 31-35
    line(27, 30)
    line(31, 35)
    edges.append([30 + offset, 31 + offset])
    # Eyes 36-41 and 42-47
    loop(36, 6)
    loop(42, 6)
    # Mouth outer 48-59 and inner 60-67
    loop(48, 12)
    loop(60, 8)

    return edges


def _build_hand_edges(offset):
    edges = []
    wrist = offset
    finger_starts = [1, 5, 9, 13, 17]

    for start in finger_starts:
        edges.append([wrist, offset + start])

    finger_ranges = [
        [1, 2, 3, 4],      # thumb
        [5, 6, 7, 8],      # index
        [9, 10, 11, 12],   # middle
        [13, 14, 15, 16],  # ring
        [17, 18, 19, 20],  # pinky
    ]
    for finger in finger_ranges:
        for i in range(len(finger) - 1):
            edges.append([offset + finger[i], offset + finger[i + 1]])

    return edges


def _build_wholebody_labels():
    face_labels = [f"face_{i}" for i in range(68)]
    left_hand = [
        "left_wrist",
        "left_thumb1", "left_thumb2", "left_thumb3", "left_thumb4",
        "left_index1", "left_index2", "left_index3", "left_index4",
        "left_middle1", "left_middle2", "left_middle3", "left_middle4",
        "left_ring1", "left_ring2", "left_ring3", "left_ring4",
        "left_pinky1", "left_pinky2", "left_pinky3", "left_pinky4",
    ]
    right_hand = [
        "right_wrist",
        "right_thumb1", "right_thumb2", "right_thumb3", "right_thumb4",
        "right_index1", "right_index2", "right_index3", "right_index4",
        "right_middle1", "right_middle2", "right_middle3", "right_middle4",
        "right_ring1", "right_ring2", "right_ring3", "right_ring4",
        "right_pinky1", "right_pinky2", "right_pinky3", "right_pinky4",
    ]
    return (
        COCO_KEYPOINT_LABELS
        + COCO_FOOT_KEYPOINT_LABELS
        + face_labels
        + left_hand
        + right_hand
    )


def build_coco_wholebody_skeleton(include_labels=True):
    labels = _build_wholebody_labels() if include_labels else None
    edges = list(COCO_SKELETON_EDGES)

    # Foot edges (body indices 15/16, foot indices 17-22)
    edges.extend([
        [15, 17],
        [17, 18],
        [18, 19],
        [15, 19],
        [16, 20],
        [20, 21],
        [21, 22],
        [16, 22],
    ])

    # Face edges (68 points starting at index 23)
    edges.extend(_build_face_edges(23))

    # Hand edges (left hand index 91, right hand index 112)
    edges.extend(_build_hand_edges(91))
    edges.extend(_build_hand_edges(112))

    return fo.KeypointSkeleton(labels=labels, edges=edges)


def _parse_keypoints(raw_keypoints):
    points = []
    scores = []

    if not raw_keypoints:
        return points, scores

    if isinstance(raw_keypoints[0], (list, tuple)):
        for entry in raw_keypoints:
            if len(entry) >= 2:
                points.append([float(entry[0]), float(entry[1])])
            if len(entry) >= 3:
                scores.append(float(entry[2]))
        return points, scores

    raw = [float(v) for v in raw_keypoints]
    if len(raw) % 3 == 0:
        for i in range(0, len(raw), 3):
            points.append([raw[i], raw[i + 1]])
            scores.append(raw[i + 2])
        return points, scores

    for i in range(0, len(raw) - 1, 2):
        points.append([raw[i], raw[i + 1]])
    return points, scores


def _normalize_points(points, width, height):
    if width <= 0 or height <= 0:
        return points

    normalized = []
    for x, y in points:
        nx = max(0.0, min(1.0, x / width))
        ny = max(0.0, min(1.0, y / height))
        normalized.append([nx, ny])
    return normalized


def _bbox_from_points(points):
    if not points:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)

    width = max(0.0, max_x - min_x)
    height = max(0.0, max_y - min_y)

    return [min_x, min_y, width, height]


def build_keypoints_label(
    pred_instances,
    width,
    height,
    keypoint_limit=None,
    label=None,
):
    keypoints = []

    if pred_instances is None:
        return fo.Keypoints(keypoints=keypoints)

    for instance in pred_instances:
        instance_label = None
        if isinstance(instance, dict):
            raw_points = instance.get("keypoints") or []
            instance_label = instance.get("label")
        else:
            raw_points = instance

        points, scores_from_points = _parse_keypoints(raw_points or [])
        scores = []

        if isinstance(instance, dict):
            scores = instance.get("keypoint_scores") or scores_from_points
        else:
            scores = scores_from_points

        if keypoint_limit:
            points = points[:keypoint_limit]
            scores = scores[:keypoint_limit] if scores else scores

        if not points:
            continue

        normalized = _normalize_points(points, width, height)

        instance_confidence = None
        if isinstance(instance, dict):
            instance_confidence = instance.get("bbox_score")
            if instance_confidence is None:
                instance_confidence = instance.get("score")

        if instance_confidence is None and scores:
            instance_confidence = sum(scores) / len(scores)

        label_to_use = instance_label or label

        # Create keypoint without confidence parameter
        keypoint = fo.Keypoint(points=normalized, label=label_to_use)

        # Keypoint.confidence expects a per-point list of floats
        if scores:
            keypoint.confidence = [float(s) for s in scores]

        # Store instance-level confidence separately as a scalar
        if instance_confidence is not None:
            if isinstance(instance_confidence, (list, tuple)):
                instance_confidence = (
                    float(instance_confidence[0]) if instance_confidence else None
                )
            else:
                instance_confidence = float(instance_confidence)
            if instance_confidence is not None:
                keypoint["instance_confidence"] = instance_confidence

        bbox = _bbox_from_points(normalized)
        if bbox:
            keypoint["bbox"] = bbox

        keypoints.append(keypoint)

    return fo.Keypoints(keypoints=keypoints)


def infer_keypoint_count(pred_dir):
    pred_dir = Path(pred_dir)
    for json_file in pred_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                pred_data = json.load(f)
            if isinstance(pred_data, dict):
                pred_data = (
                    pred_data.get("predictions")
                    or pred_data.get("instances")
                    or ([pred_data] if "keypoints" in pred_data else pred_data)
                )
            if isinstance(pred_data, list) and pred_data:
                instance = pred_data[0]
                raw_points = instance.get("keypoints") if isinstance(instance, dict) else instance
                points, _ = _parse_keypoints(raw_points or [])
                if points:
                    return len(points)
        except Exception:
            continue
    return None


def _infer_keypoints_field(dataset, exclude_fields=None):
    exclude_fields = set(exclude_fields or [])
    schema = dataset.get_field_schema()
    preferred = ["ground_truth", "keypoints", "gt", "annotations"]
    for field_name in preferred:
        field = schema.get(field_name)
        if (
            field_name not in exclude_fields
            and isinstance(field, fo.EmbeddedDocumentField)
            and field.document_type == fo.Keypoints
        ):
            return field_name
    for field_name, field in schema.items():
        if (
            field_name not in exclude_fields
            and isinstance(field, fo.EmbeddedDocumentField)
            and field.document_type == fo.Keypoints
        ):
            return field_name
    return None


def _infer_keypoint_label(dataset, gt_field):
    if gt_field is None:
        return None
    try:
        for sample in dataset.iter_samples(progress=False):
            gt = sample.get(gt_field)
            if gt is None:
                continue
            keypoints = getattr(gt, "keypoints", None)
            if not keypoints:
                continue
            for kp in keypoints:
                if kp.label:
                    return kp.label
            return keypoints[0].label
    except Exception:
        return None
    return None


def _save_view(dataset, name, view):
    try:
        dataset.save_view(name, view)
    except Exception:
        try:
            dataset.delete_saved_view(name)
            dataset.save_view(name, view)
        except Exception as exc:
            print(f"Warning: Could not save view '{name}': {exc}")


def attach_predictions(
    dataset,
    pred_dir,
    pred_field,
    keypoint_limit=None,
    pred_label=None,
):
    predictions = load_rtmpose_predictions(pred_dir)
    if not predictions:
        print("No predictions found to attach.")
        return 0

    updated = 0
    for sample in dataset.iter_samples(progress=True):
        image_name = Path(sample.filepath).stem
        if image_name not in predictions:
            continue

        pred_instances = predictions[image_name]
        if isinstance(pred_instances, dict):
            if "keypoints" in pred_instances:
                pred_instances = [pred_instances]
            else:
                print(f"Warning: Unexpected prediction format for {image_name}")
                continue
        if not isinstance(pred_instances, list):
            print(f"Warning: Skipping predictions for {image_name} (unsupported type)")
            continue
        metadata = sample.metadata
        if metadata is None:
            metadata = fo.ImageMetadata.build_for(sample.filepath)
            sample.metadata = metadata

        keypoints = build_keypoints_label(
            pred_instances,
            metadata.width,
            metadata.height,
            keypoint_limit=keypoint_limit,
            label=pred_label,
        )
        sample[pred_field] = keypoints
        sample["num_persons"] = len(keypoints.keypoints)
        sample.save()
        updated += 1

    print(f"Attached predictions to {updated} samples.")
    return updated


def evaluate_pose(
    dataset,
    pred_field,
    gt_field,
    eval_key,
    low_oks_min=0.1,
    low_oks_max=0.5,
    fp_confidence=0.8,
    save_views=True,
    tag_failures=False,
    occlusion_tag=None,
):
    if pred_field not in dataset.get_field_schema():
        print("Skipping evaluation: predictions field not found.")
        return None

    if gt_field is None:
        print("Skipping evaluation: no ground truth keypoints field found.")
        return None

    print(
        f"Running COCO OKS evaluation: pred='{pred_field}', "
        f"gt='{gt_field}', eval_key='{eval_key}'"
    )
    results = dataset.evaluate_detections(
        pred_field,
        gt_field=gt_field,
        eval_key=eval_key,
        method="coco",
    )

    # Print detailed metrics
    print("\n" + "="*60)
    print("COCO OKS Evaluation Results")
    print("="*60)
    results.print_report()
    print("="*60 + "\n")

    low_oks_filter = (
        (F(eval_key) == "tp")
        & (F(f"{eval_key}_oks") >= low_oks_min)
        & (F(f"{eval_key}_oks") <= low_oks_max)
    )
    low_oks_view = dataset.filter_labels(pred_field, low_oks_filter)
    if save_views:
        _save_view(dataset, "low_oks", low_oks_view)
    if tag_failures:
        low_oks_view.tag_samples("needs_review")

    fp_filter = (F(eval_key) == "fp") & (F("instance_confidence") >= fp_confidence)
    fp_view = dataset.filter_labels(pred_field, fp_filter)
    if save_views:
        _save_view(dataset, "high_conf_fp", fp_view)
    if tag_failures:
        fp_view.tag_samples("hallucinated")

    if occlusion_tag:
        occlusion_view = dataset.match_tags(occlusion_tag)
        if save_views:
            _save_view(dataset, "occlusion_focus", occlusion_view)

    return results


def create_fiftyone_dataset(
    source='local',
    dataset_dir='dataset/polar',
    split='val',
    output_dir='outputs',
    dataset_name='polar-rtmpose',
    persistent=False,
    zoo_dataset='coco-2017',
    zoo_split='validation',
    zoo_max_samples=None,
    pred_field='predictions',
    gt_field=None,
    eval_key='eval_pose',
    keypoint_limit=None,
    pred_label=None,
    apply_coco_skeleton=False,
    run_eval=True,
    save_views=True,
    tag_failures=False,
    occlusion_tag=None,
):
    """
    Create a FiftyOne dataset with images and RTMPose predictions.

    Args:
        dataset_dir: Path to dataset directory (YOLO datasets supported)
        split: Dataset split to use ('train', 'val', or 'test')
        output_dir: Directory containing RTMPose outputs
        dataset_name: Name for the FiftyOne dataset
        persistent: Whether to make the dataset persistent
        pred_label: Label to assign to predicted keypoints
    """

    # Check if dataset already exists
    if dataset_name in fo.list_datasets():
        print(f"Loading existing dataset: {dataset_name}")
        dataset = fo.load_dataset(dataset_name)
    else:
        if source == 'zoo':
            print(f"Loading zoo dataset: {zoo_dataset} ({zoo_split})")
            dataset = foz.load_zoo_dataset(
                zoo_dataset,
                split=zoo_split,
                label_types=["keypoints"],
                dataset_name=dataset_name,
                persistent=persistent,
                max_samples=zoo_max_samples,
            )
        else:
            print(f"Creating new FiftyOne dataset: {dataset_name}")
            dataset_dir = Path(dataset_dir)
            if _is_yolo_dataset_dir(dataset_dir):
                dataset = fo.Dataset.from_dir(
                    dataset_dir=str(dataset_dir),
                    dataset_type=fo.types.YOLOv5Dataset,
                    split=split,
                    label_field="ground_truth",
                    name=dataset_name,
                )
                dataset.persistent = persistent
            else:
                dataset = fo.Dataset(dataset_name, persistent=persistent)

                # Get image directory
                image_dir = dataset_dir / split
                if not image_dir.exists():
                    raise ValueError(f"Image directory not found: {image_dir}")

                # Get all images
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_files.extend(list(image_dir.rglob(ext)))

                print(f"Found {len(image_files)} images in {split} split")

                # Add samples to dataset
                samples = []
                for img_path in image_files:
                    img_path = Path(img_path)

                    # Create sample
                    sample = fo.Sample(filepath=str(img_path))

                    # Add metadata
                    sample["split"] = split
                    samples.append(sample)

                dataset.add_samples(samples)

    # Load RTMPose predictions
    pred_dir = Path(output_dir) / 'predictions'
    inferred_count = infer_keypoint_count(pred_dir) if pred_dir.exists() else None
    if apply_coco_skeleton:
        effective_count = keypoint_limit or inferred_count
        if effective_count == 17:
            dataset.default_skeleton = build_coco_skeleton(include_labels=True)
        elif effective_count == 133:
            dataset.default_skeleton = build_coco_wholebody_skeleton(include_labels=True)
        elif effective_count and effective_count > 17:
            dataset.default_skeleton = build_coco_skeleton(include_labels=False)

    if gt_field is None:
        gt_field = _infer_keypoints_field(dataset, exclude_fields=[pred_field])

    if pred_label is None:
        pred_label = _infer_keypoint_label(dataset, gt_field)

    # Set default label to "person" if not specified and no GT found
    if pred_label is None and source == 'zoo' and 'coco' in zoo_dataset:
        pred_label = "person"

    # Set default label to "person" even for local datasets if no label inferred
    # This ensures predictions have labels for evaluation compatibility
    if pred_label is None:
        pred_label = "person"
        print(f"No ground truth keypoint labels found. Setting prediction label to: {pred_label}")

    if pred_dir.exists():
        attach_predictions(
            dataset,
            pred_dir,
            pred_field=pred_field,
            keypoint_limit=keypoint_limit,
            pred_label=pred_label,
        )
    else:
        print(f"Warning: Predictions directory not found: {pred_dir}")

    if run_eval:
        evaluate_pose(
            dataset,
            pred_field=pred_field,
            gt_field=gt_field,
            eval_key=eval_key,
            save_views=save_views,
            tag_failures=tag_failures,
            occlusion_tag=occlusion_tag,
        )

    print(f"\nDataset created successfully!")
    print(f"  - Name: {dataset_name}")
    print(f"  - Samples: {len(dataset)}")
    print(f"  - Persistent: {persistent}")

    return dataset


def launch_fiftyone_app(dataset, port=5151):
    """Launch FiftyOne app."""

    print("\n" + "="*60)
    print("Launching FiftyOne App")
    print("="*60)
    print(f"Dataset: {dataset.name}")
    print(f"Samples: {len(dataset)}")
    print(f"Port: {port}")
    print("\nPress Ctrl+C to stop the app")
    print("="*60 + "\n")

    fo.config.show_skeletons = True
    fo.config.multicolor_keypoints = True

    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nShutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Create session and disable auto-opening browser to avoid interrupt issues
        session = fo.launch_app(dataset, port=port, auto=False)

        print(f"FiftyOne app is running at: http://localhost:{port}")
        print("Open the URL in your browser to view the dataset.\n")

        # Keep session alive
        session.wait()
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
    except Exception as e:
        print(f"\nError during session: {e}")
        raise
    finally:
        print("FiftyOne app stopped.")


def main():
    parser = argparse.ArgumentParser(
        description='Launch FiftyOne to visualize RTMPose results'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='local',
        choices=['local', 'zoo'],
        help='Dataset source: local directory (YOLO supported) or FiftyOne zoo'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='dataset/polar',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Dataset split to use'
    )
    parser.add_argument(
        '--zoo-dataset',
        type=str,
        default='coco-2017',
        help='FiftyOne zoo dataset name'
    )
    parser.add_argument(
        '--zoo-split',
        type=str,
        default='validation',
        help='FiftyOne zoo dataset split'
    )
    parser.add_argument(
        '--zoo-max-samples',
        type=int,
        default=None,
        help='Maximum number of zoo samples to load'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directory containing RTMPose outputs'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='polar-rtmpose',
        help='Name for the FiftyOne dataset'
    )
    parser.add_argument(
        '--pred-field',
        type=str,
        default='predictions',
        help='Field name for RTMPose predictions'
    )
    parser.add_argument(
        '--pred-label',
        type=str,
        default=None,
        help='Label to assign to predicted keypoints (defaults to GT label)'
    )
    parser.add_argument(
        '--gt-field',
        type=str,
        default=None,
        help='Ground truth keypoints field (auto-detect if omitted)'
    )
    parser.add_argument(
        '--eval-key',
        type=str,
        default='eval_pose',
        help='Evaluation key for OKS metrics'
    )
    parser.add_argument(
        '--keypoint-limit',
        type=int,
        default=None,
        help='Limit predictions to first N keypoints (use 17 for COCO body)'
    )
    parser.add_argument(
        '--no-coco-skeleton',
        action='store_true',
        help='Disable skeleton overlay'
    )
    parser.add_argument(
        '--skip-eval',
        action='store_true',
        help='Skip OKS evaluation'
    )
    parser.add_argument(
        '--no-save-views',
        action='store_true',
        help='Skip saving curated views'
    )
    parser.add_argument(
        '--tag-failures',
        action='store_true',
        help='Tag low-OKS samples for review and high-conf FPs'
    )
    parser.add_argument(
        '--occlusion-tag',
        type=str,
        default=None,
        help='Sample tag to use for occlusion-focused view'
    )
    parser.add_argument(
        '--persistent',
        action='store_true',
        help='Make the dataset persistent (saved to disk)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5151,
        help='Port for FiftyOne app'
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help='Delete existing dataset and recreate'
    )

    args = parser.parse_args()

    # Delete existing dataset if requested
    if args.delete and args.dataset_name in fo.list_datasets():
        print(f"Deleting existing dataset: {args.dataset_name}")
        fo.delete_dataset(args.dataset_name)

    # Create or load dataset
    keypoint_limit = args.keypoint_limit
    if keypoint_limit is None and args.source == 'zoo' and 'coco' in args.zoo_dataset:
        keypoint_limit = 17

    apply_coco_skeleton = not args.no_coco_skeleton

    dataset = create_fiftyone_dataset(
        source=args.source,
        dataset_dir=args.dataset_dir,
        split=args.split,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        persistent=args.persistent,
        zoo_dataset=args.zoo_dataset,
        zoo_split=args.zoo_split,
        zoo_max_samples=args.zoo_max_samples,
        pred_field=args.pred_field,
        gt_field=args.gt_field,
        eval_key=args.eval_key,
        keypoint_limit=keypoint_limit,
        pred_label=args.pred_label,
        apply_coco_skeleton=apply_coco_skeleton,
        run_eval=not args.skip_eval,
        save_views=not args.no_save_views,
        tag_failures=args.tag_failures,
        occlusion_tag=args.occlusion_tag,
    )

    # Launch app
    launch_fiftyone_app(dataset, port=args.port)


if __name__ == '__main__':
    main()
