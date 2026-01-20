#!/usr/bin/env bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "=========================================="
echo "RTMPose-X Environment Setup Script"
echo "=========================================="

# Prevent user site-packages from interfering with conda env
export PYTHONNOUSERSITE=1

# Define Conda directory and environment name
CONDA_DIR="$HOME/miniconda3"
ENV_NAME="rtmpose_env"

# Version configurations
CUDA_VERSION="12.4"
PYTORCH_CUDA="cu124"
TORCH_VERSION="2.5.1"
TORCHVISION_VERSION="0.20.1"
MMENGINE_VERSION="0.10.7"
MMCV_VERSION="2.1.0"
MMDET_VERSION="3.2.0"
MMPOSE_VERSION="1.3.2"
NUMPY_VERSION="1.26.4"

# --- Conda Installation and Initialization ---
if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing Miniconda in $CONDA_DIR..."
    MINICONDA_SCRIPT_PATH="$HOME/miniconda.sh"
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$MINICONDA_SCRIPT_PATH"
    /bin/bash "$MINICONDA_SCRIPT_PATH" -b -p "$CONDA_DIR"
    rm "$MINICONDA_SCRIPT_PATH"
    # Initialize conda for the current shell session
    eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"
    # Configure conda for future shell sessions
    conda init bash
else
    echo "Conda already installed."
    # Still need to hook it for the current script session
    eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"
fi

# --- Environment Creation ---
echo "Cleaning Conda cache..."
conda clean -afy

# Create conda environment if it doesn't exist
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists. Removing to ensure clean state..."
    conda env remove -n "${ENV_NAME}" -y
fi

echo "Creating conda environment '${ENV_NAME}' with Python 3.10..."
conda create -n "${ENV_NAME}" python=3.10 -y

# --- Activate environment ---
echo "Activating conda environment '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

# Get the full path to the python executable in the active environment
PYTHON_IN_ENV=$(which python)
echo "--- Will use Python from: $PYTHON_IN_ENV ---"

# Upgrade pip first
echo "Upgrading pip..."
"$PYTHON_IN_ENV" -m pip install --upgrade pip

# --- Install PyTorch ---
echo ""
echo "Installing PyTorch ${TORCH_VERSION} with CUDA ${CUDA_VERSION}..."
echo "Using conda to install PyTorch (more reliable for large packages)..."
conda install -y "pytorch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    "pytorch-cuda=${CUDA_VERSION}" -c pytorch -c nvidia

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
if ! "$PYTHON_IN_ENV" -c "import torch; print('PyTorch already installed')" 2>/dev/null; then
    echo "Conda installation failed, trying pip with increased retries..."
    # Try pip as fallback
    "$PYTHON_IN_ENV" -m pip install --retries 10 --timeout 600 --no-cache-dir \
        "torch==${TORCH_VERSION}+${PYTORCH_CUDA}" "torchvision==${TORCHVISION_VERSION}+${PYTORCH_CUDA}" \
        --extra-index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}"
fi

# Final verification that PyTorch is installed
if ! "$PYTHON_IN_ENV" -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch installation failed! Cannot proceed."
    echo "Please check your internet connection and SSL certificates."
    echo "You may need to manually install PyTorch:"
    echo "  conda activate ${ENV_NAME}"
    echo "  pip install torch==${TORCH_VERSION}+${PYTORCH_CUDA} torchvision==${TORCHVISION_VERSION}+${PYTORCH_CUDA} --extra-index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}"
    exit 1
fi

echo "PyTorch installation successful!"
"$PYTHON_IN_ENV" -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# --- Pin NumPy ---
echo ""
echo "Pinning NumPy to ${NUMPY_VERSION} to avoid compatibility issues..."
"$PYTHON_IN_ENV" -m pip install "numpy==${NUMPY_VERSION}"

# --- Install OpenMMLab dependencies ---
echo ""
echo "Installing OpenMMLab dependencies (openmim, mmengine, mmcv)..."
"$PYTHON_IN_ENV" -m pip install --no-cache-dir -U openmim

echo "Installing mmengine ${MMENGINE_VERSION}..."
if ! mim install "mmengine==${MMENGINE_VERSION}"; then
    echo "mim install failed, trying pip..."
    "$PYTHON_IN_ENV" -m pip install --no-cache-dir "mmengine==${MMENGINE_VERSION}"
fi

# --- Install MMCV ---
echo ""
echo "Installing MMCV ${MMCV_VERSION} (compatible with mmdet)..."
# Check if mmcv is already installed
if "$PYTHON_IN_ENV" -c "import mmcv; print(f'MMCV {mmcv.__version__} already installed')" 2>/dev/null; then
    echo "MMCV already installed, skipping..."
else
    # CUDA 12.4 wheels are not published for MMCV; try pip first, then build from source
    if ! mim install "mmcv==${MMCV_VERSION}"; then
        echo "mim install failed, trying pip directly..."
        if ! "$PYTHON_IN_ENV" -m pip install --no-cache-dir "mmcv==${MMCV_VERSION}"; then
            echo "MMCV wheel not available, building from source..."
            if command -v nvcc &> /dev/null; then
                echo "nvcc found, building with CUDA ops..."
                MMCV_WITH_OPS=1 FORCE_CUDA=1 "$PYTHON_IN_ENV" -m pip install --no-cache-dir --no-binary mmcv "mmcv==${MMCV_VERSION}"
            else
                echo "Warning: nvcc not found; installing MMCV without CUDA ops."
                "$PYTHON_IN_ENV" -m pip install --no-cache-dir --no-binary mmcv "mmcv==${MMCV_VERSION}"
            fi
        fi
    fi
fi

# --- Install MMDetection ---
echo ""
echo "Installing MMDetection ${MMDET_VERSION} for person detection..."
if ! mim install "mmdet==${MMDET_VERSION}"; then
    echo "mim install failed, trying pip..."
    "$PYTHON_IN_ENV" -m pip install --no-cache-dir "mmdet==${MMDET_VERSION}"
fi

# --- Install chumpy ---
echo ""
echo "Installing chumpy (mmpose dependency)..."
"$PYTHON_IN_ENV" -m pip install --no-cache-dir chumpy --no-build-isolation || \
    "$PYTHON_IN_ENV" -m pip install --no-cache-dir chumpy || \
    echo "Warning: chumpy installation failed (non-critical)"

# --- Install MMPose ---
echo ""
echo "Installing MMPose ${MMPOSE_VERSION}..."
if ! mim install "mmpose==${MMPOSE_VERSION}"; then
    echo "mim install failed, trying pip..."
    "$PYTHON_IN_ENV" -m pip install --no-cache-dir "mmpose==${MMPOSE_VERSION}"
fi

# --- Install FiftyOne ---
echo ""
echo "Installing FiftyOne..."
# Pin NumPy and opencv-python-headless versions to avoid conflicts
"$PYTHON_IN_ENV" -m pip install --no-cache-dir fiftyone "numpy<2" "opencv-python-headless==4.9.0.80" || \
    echo "Warning: FiftyOne installation failed (optional)"

# --- Install FiftyOne plugins ---
echo ""
echo "Installing FiftyOne plugins..."
# Install transformers package required for VitPose plugin
echo "Installing transformers for VitPose plugin..."
"$PYTHON_IN_ENV" -m pip install --no-cache-dir transformers || \
    echo "Warning: transformers installation failed"

# Use the fiftyone command from the conda environment
FIFTYONE_CMD="$CONDA_DIR/envs/$ENV_NAME/bin/fiftyone"
if [ -f "$FIFTYONE_CMD" ]; then
    # Install VitPose plugin
    echo "Installing VitPose plugin..."
    "$FIFTYONE_CMD" plugins download https://github.com/harpreetsahota204/vitpose-plugin || \
        echo "Warning: VitPose plugin download failed (non-critical)"

    # Install other FiftyOne plugins
    "$FIFTYONE_CMD" plugins download https://github.com/voxel51/fiftyone-plugins || \
        echo "Warning: FiftyOne plugins download failed (non-critical)"

    echo "Verifying VitPose plugin installation..."
    "$FIFTYONE_CMD" plugins list | grep -i vitpose && \
        echo "✓ VitPose plugin installed successfully!" || \
        echo "⚠ VitPose plugin may not be installed correctly"
else
    echo "Warning: FiftyOne command not found at $FIFTYONE_CMD, skipping plugins"
fi

# --- Install COCO tools ---
echo ""
echo "Installing COCO tools..."
"$PYTHON_IN_ENV" -m pip install --no-cache-dir pycocotools
"$PYTHON_IN_ENV" -m pip install --no-cache-dir xtcocotools || \
    echo "Warning: xtcocotools installation failed (non-critical)"

# --- Install additional packages ---
echo ""
echo "Installing additional packages (opencv, pillow, matplotlib)..."
"$PYTHON_IN_ENV" -m pip install --no-cache-dir \
    "scipy==1.13.1" \
    "opencv-python==4.9.0.80" "opencv-python-headless==4.9.0.80" \
    pillow matplotlib pandas tqdm

# Reinstall numpy to ensure it stays at the correct version
"$PYTHON_IN_ENV" -m pip install --force-reinstall --no-deps "numpy==${NUMPY_VERSION}"

# --- Verify installations ---
echo ""
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="

# Check PyTorch
if "$PYTHON_IN_ENV" -c "import torch; print(f'✓ PyTorch version: {torch.__version__}'); print(f'✓ CUDA build: {torch.version.cuda}'); print(f'✓ CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    :
else
    echo "✗ PyTorch installation failed!"
fi

# Check MMPose
if "$PYTHON_IN_ENV" -c "import mmpose; print(f'✓ MMPose version: {mmpose.__version__}')" 2>/dev/null; then
    :
else
    echo "✗ MMPose installation failed!"
fi

# Check MMDetection
if "$PYTHON_IN_ENV" -c "import mmdet; print(f'✓ MMDetection version: {mmdet.__version__}')" 2>/dev/null; then
    :
else
    echo "✗ MMDetection installation failed!"
fi

# Check MMCV
if "$PYTHON_IN_ENV" -c "import mmcv; print(f'✓ MMCV version: {mmcv.__version__}')" 2>/dev/null; then
    :
else
    echo "✗ MMCV installation failed!"
fi

# Check FiftyOne
if "$PYTHON_IN_ENV" -c "import fiftyone as fo; print(f'✓ FiftyOne version: {fo.__version__}')" 2>/dev/null; then
    :
else
    echo "⚠ FiftyOne installation failed (optional)"
fi

# --- Cleanup ---
echo ""
echo "Cleaning up caches..."
conda clean -afy
"$PYTHON_IN_ENV" -m pip cache purge

echo ""
echo "=========================================="
echo "Setup completed!"
echo "=========================================="
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "IMPORTANT: To use FiftyOne and the VitPose plugin:"
echo "  1. Always activate the conda environment first:"
echo "     conda activate ${ENV_NAME}"
echo "  2. Launch FiftyOne from the activated environment:"
echo "     ./launch_fiftyone.py"
echo ""
echo "The VitPose plugin will only appear in the FiftyOne UI"
echo "when launched from the conda environment."
echo ""
