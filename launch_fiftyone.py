#!/usr/bin/env python3
"""
FiftyOne Dataset Viewer for RTMPose Results

This script launches FiftyOne to visualize:
1. Images from the dataset
2. RTMPose predictions (keypoints)
3. Ground truth annotations (if available)
4. Interactive exploration and filtering
"""

import os
import json
import argparse
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz


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
                predictions[image_name] = pred_data
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return predictions


def create_fiftyone_dataset(
    dataset_dir='dataset',
    split='val',
    output_dir='outputs',
    dataset_name='rtmpose-results',
    persistent=False
):
    """
    Create a FiftyOne dataset with images and RTMPose predictions.

    Args:
        dataset_dir: Path to dataset directory
        split: Dataset split to use ('train' or 'val')
        output_dir: Directory containing RTMPose outputs
        dataset_name: Name for the FiftyOne dataset
        persistent: Whether to make the dataset persistent
    """

    # Check if dataset already exists
    if dataset_name in fo.list_datasets():
        print(f"Loading existing dataset: {dataset_name}")
        dataset = fo.load_dataset(dataset_name)
        return dataset

    print(f"Creating new FiftyOne dataset: {dataset_name}")

    # Create dataset
    dataset = fo.Dataset(dataset_name, persistent=persistent)

    # Get image directory
    image_dir = Path(dataset_dir) / split
    if not image_dir.exists():
        raise ValueError(f"Image directory not found: {image_dir}")

    # Get all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(image_dir.rglob(ext)))

    print(f"Found {len(image_files)} images in {split} split")

    # Load RTMPose predictions
    pred_dir = Path(output_dir) / 'predictions'
    predictions = load_rtmpose_predictions(pred_dir)
    print(f"Loaded predictions for {len(predictions)} images")

    # Add samples to dataset
    samples = []
    for img_path in image_files:
        img_path = Path(img_path)
        image_name = img_path.stem

        # Create sample
        sample = fo.Sample(filepath=str(img_path))

        # Add metadata
        sample["split"] = split

        # Add RTMPose predictions if available
        if image_name in predictions:
            pred_data = predictions[image_name]

            # Add keypoints for each detected person
            keypoints_list = []
            for instance in pred_data:
                if 'keypoints' in instance:
                    kpts = instance['keypoints']  # Shape: (num_keypoints, 2 or 3)
                    scores = instance.get('keypoint_scores', [])

                    # Convert to FiftyOne format (normalized coordinates)
                    # Note: You may need to normalize based on image dimensions
                    keypoints_list.append({
                        'points': kpts,
                        'scores': scores,
                        'bbox': instance.get('bbox', None)
                    })

            sample["rtmpose_predictions"] = keypoints_list
            sample["num_persons"] = len(keypoints_list)
        else:
            sample["num_persons"] = 0

        samples.append(sample)

    # Add samples to dataset
    dataset.add_samples(samples)

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

    # Create session
    session = fo.launch_app(dataset, port=port)

    # Keep session alive
    session.wait()


def main():
    parser = argparse.ArgumentParser(
        description='Launch FiftyOne to visualize RTMPose results'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='dataset',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val'],
        help='Dataset split to use'
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
        default='rtmpose-results',
        help='Name for the FiftyOne dataset'
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
    dataset = create_fiftyone_dataset(
        dataset_dir=args.dataset_dir,
        split=args.split,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        persistent=args.persistent
    )

    # Launch app
    launch_fiftyone_app(dataset, port=args.port)


if __name__ == '__main__':
    main()
