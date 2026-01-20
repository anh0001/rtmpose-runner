# RTMPose-X Inference Setup

This project provides scripts to run RTMPose-X pose estimation model on your dataset.
The default dataset is `dataset/polar` in YOLO format.

## Quick Start

### Step 1: Install Dependencies

Run the setup script to install all required packages:

```bash
./setup.sh
```

This will install:
- PyTorch with CUDA support
- MMPose and related OpenMMLab libraries (mmengine, mmcv, mmdet)
- FiftyOne for dataset management
- Additional utilities (opencv, matplotlib, etc.)

### Step 2: Run Inference

After installation, activate the environment and run inference:

```bash
conda activate rtmpose_env
python run_rtmpose_inference.py --split val
```

## Usage Examples

### Basic inference on validation set
```bash
python run_rtmpose_inference.py --split val
```

### Run on training set
```bash
python run_rtmpose_inference.py --split train
```

### Use different RTMPose model
```bash
python run_rtmpose_inference.py --model rtmpose-l --split val
```

### Use person detection (slower but more accurate for multi-person images)
```bash
python run_rtmpose_inference.py --detection-mode person_detection --split val
```

### Run on CPU (if no GPU available)
```bash
python run_rtmpose_inference.py --device cpu --split val
```

### Disable visualization to save disk space
```bash
python run_rtmpose_inference.py --no-visualize --split val
```

## Visualize with FiftyOne

```bash
python launch_fiftyone.py --source local --dataset-dir dataset/polar --split val
```

## Model Evaluation and Comparison

### Comparing RTMPose-X with VitPose

This project includes a script to evaluate RTMPose-X predictions against VitPose outputs as a reference baseline. This is useful for:
- Quick quality checks without manual annotations
- Comparing two different pose estimation models
- Understanding relative performance differences

**Important Note**: This comparison uses VitPose predictions as pseudo-ground truth, NOT human-annotated keypoints. The evaluation metrics show how well RTMPose-X agrees with VitPose, not the absolute accuracy of either model.

### Running the Evaluation

1. **Generate RTMPose-X predictions** (if not already done):
   ```bash
   python run_rtmpose_inference.py --split val
   ```

2. **Load dataset in FiftyOne and run VitPose**:
   ```bash
   python launch_fiftyone.py --source local --dataset-dir dataset/polar --split val --dataset-name polar-rtmpose --skip-eval
   ```

   Then in the FiftyOne UI:
   - Run the VitPose plugin on your dataset
   - This creates a 'vitpose-base-keypoints' field with reference keypoints

3. **Run the evaluation script**:
   ```bash
   python fix_evaluation.py
   ```

### What the Evaluation Does

The `fix_evaluation.py` script:
1. Ensures prediction labels match ground truth labels (both set to "person")
2. Runs COCO OKS (Object Keypoint Similarity) evaluation
3. Compares RTMPose-X predictions against VitPose outputs
4. Generates metrics including:
   - Average Precision (AP) at different OKS thresholds
   - AP at OKS=0.50, 0.75, etc.
   - Performance across different keypoint types

### Understanding the Results

The evaluation outputs COCO-style metrics:
- **AP**: Average Precision averaged over OKS thresholds from 0.50 to 0.95
- **AP50**: Average Precision at OKS threshold of 0.50 (more lenient)
- **AP75**: Average Precision at OKS threshold of 0.75 (stricter)

Higher values indicate better agreement between RTMPose-X and VitPose. However, remember that:
- High agreement doesn't necessarily mean both models are correct
- Low agreement could mean RTMPose-X is wrong, VitPose is wrong, or they have different strengths
- For true accuracy assessment, you need human-annotated ground truth

## Command Line Arguments

- `--model`: RTMPose model variant (default: 'rtmpose-x')
  - Options: rtmpose-x, rtmpose-l, rtmpose-m, rtmpose-s
- `--dataset-dir`: Path to dataset directory (default: 'dataset/polar')
- `--split`: Dataset split to use (default: 'val')
  - Options: train, val, test
- `--output-dir`: Output directory for results (default: 'outputs')
- `--detection-mode`: Detection mode (default: 'whole_image')
  - `whole_image`: Faster, assumes single person per image
  - `person_detection`: Uses person detector first (slower but handles multiple people)
- `--device`: Device to use (default: 'cuda:0')
  - Options: cuda:0, cpu
- `--no-visualize`: Disable saving visualization images

## Output Structure

After running inference, you'll find:

```
outputs/
├── visualizations/          # Pose overlay images
│   ├── image1_pose.jpg
│   └── ...
├── predictions/             # Individual JSON predictions
│   ├── image1.json
│   └── ...
├── all_results_val.json     # All predictions combined
└── performance_metrics_val.json  # Performance statistics
```

## Performance Metrics

The script outputs:
- Total processing time
- Average time per image
- FPS (frames per second)
- Device and model information

## Dataset Structure

Expected dataset structure (YOLO format):
```
dataset/polar/
├── dataset.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## Troubleshooting

### CUDA Out of Memory
If you get CUDA OOM errors, try:
- Use smaller model: `--model rtmpose-m` or `--model rtmpose-s`
- Run on CPU: `--device cpu`

### Slow inference
- Use `--detection-mode whole_image` for single-person images
- Use smaller model variant
- Disable visualization: `--no-visualize`

### Model download issues
The first run will download pretrained model weights (~100-500 MB). Ensure you have:
- Stable internet connection
- Sufficient disk space (~2-5 GB for models and dependencies)

## Available RTMPose Models

- **rtmpose-x**: Extra large, highest accuracy
- **rtmpose-l**: Large, good balance
- **rtmpose-m**: Medium, faster
- **rtmpose-s**: Small, fastest

## References

- [MMPose Documentation](https://mmpose.readthedocs.io/)
- [RTMPose Paper](https://arxiv.org/abs/2303.07399)
- [Model Zoo](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
