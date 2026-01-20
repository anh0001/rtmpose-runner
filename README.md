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
