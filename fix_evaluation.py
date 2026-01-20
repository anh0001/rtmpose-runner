#!/usr/bin/env python3
"""
Fix evaluation by ensuring prediction labels match ground truth labels.

This script:
1. Updates prediction labels to "person" to match ground truth
2. Re-runs evaluation if ground truth keypoints exist
3. Shows clear results
"""

import fiftyone as fo

print("=" * 60)
print("Fixing RTMPose Evaluation")
print("=" * 60)

# Load dataset
dataset_name = "polar-rtmpose"
if dataset_name not in fo.list_datasets():
    print(f"Error: Dataset '{dataset_name}' not found!")
    print("Please create the dataset first by running:")
    print("  python launch_fiftyone.py --source local --dataset-dir dataset/polar --split val --dataset-name polar-rtmpose --skip-eval")
    exit(1)

dataset = fo.load_dataset(dataset_name)
print(f"\nDataset: {dataset_name}")
print(f"Samples: {len(dataset)}")

# Check for ground truth keypoints field
schema = dataset.get_field_schema()
keypoint_fields = []
for field_name, field in schema.items():
    if (
        field_name != "predictions"
        and isinstance(field, fo.EmbeddedDocumentField)
        and field.document_type == fo.Keypoints
    ):
        keypoint_fields.append(field_name)

print(f"\nFound keypoint fields: {['predictions'] + keypoint_fields}")

if not keypoint_fields:
    print("\n" + "=" * 60)
    print("WARNING: No ground truth keypoint field found!")
    print("=" * 60)
    print("\nTo evaluate predictions, you need ground truth keypoints.")
    print("You have two options:")
    print("\n1. Run VitPose plugin through FiftyOne UI:")
    print("   a. Launch FiftyOne: python launch_fiftyone.py --source local --dataset-dir dataset/polar --split val --dataset-name polar-rtmpose --skip-eval")
    print("   b. In the FiftyOne UI, run the VitPose plugin on your dataset")
    print("   c. This will create a 'vitpose-base-keypoints' field with ground truth")
    print("   d. Run this script again to evaluate")
    print("\n2. Or load ground truth from annotation files if available")
    print("\nFor now, fixing prediction labels to 'person'...")
else:
    gt_field = keypoint_fields[0]
    print(f"\nUsing ground truth field: {gt_field}")

    # Infer label from ground truth
    gt_label = None
    for sample in dataset.iter_samples(progress=False):
        gt = sample[gt_field]
        if gt and gt.keypoints:
            for kp in gt.keypoints:
                if kp.label:
                    gt_label = kp.label
                    break
            if gt_label:
                break

    print(f"Ground truth label: {gt_label}")

# Fix prediction labels
print("\n" + "=" * 60)
print("Updating Prediction Labels")
print("=" * 60)

target_label = "person"  # Use "person" as default
print(f"Setting all prediction labels to: '{target_label}'")

count_updated = 0
count_total = 0

for sample in dataset.iter_samples(progress=True):
    if not hasattr(sample, "predictions") or not sample.predictions:
        continue

    count_total += 1
    keypoints = sample.predictions.keypoints
    updated = False

    for kp in keypoints:
        if kp.label != target_label:
            kp.label = target_label
            updated = True

    if updated:
        sample.save()
        count_updated += 1

print(f"\nUpdated {count_updated} of {count_total} samples")

# Run evaluation if ground truth exists
if keypoint_fields:
    gt_field = keypoint_fields[0]
    eval_key = "eval_pose"

    # Clear previous evaluation
    if eval_key in dataset.list_evaluations():
        print(f"\nClearing previous evaluation '{eval_key}'...")
        dataset.delete_evaluation(eval_key)

    # Run new evaluation
    print("\n" + "=" * 60)
    print(f"Running COCO OKS Evaluation")
    print("=" * 60)
    print(f"Predictions: 'predictions'")
    print(f"Ground truth: '{gt_field}'")
    print(f"Evaluation key: '{eval_key}'")

    try:
        results = dataset.evaluate_detections(
            "predictions",
            gt_field=gt_field,
            eval_key=eval_key,
            method="coco",
        )

        # Print results
        print("\n" + "=" * 60)
        print("COCO OKS Evaluation Results")
        print("=" * 60)
        results.print_report()
        print("=" * 60 + "\n")

        # Print summary statistics
        print("Evaluation saved to dataset.")
        print(f"Use 'eval_key={eval_key}' to filter results in FiftyOne UI")

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        print("\nThis might be because:")
        print("1. Ground truth and predictions have different label types")
        print("2. Ground truth keypoints don't have bounding boxes")
        print("3. There are no matching instances between GT and predictions")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
