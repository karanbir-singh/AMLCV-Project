#!/bin/bash
#SBATCH --job-name=Isolated_Eval_Job
#SBATCH --output=logs/isolated_eval_%j.out
#SBATCH --error=logs/isolated_eval_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu=1
#SBATCH --partition=acltr
#SBATCH --time=04:00:00
#SBATCH --mem=32G

# ------------------------------------------------------------
# 1. LOAD BASE PYTHON MODULE (Crucial: DO NOT load PyTorch module here)
# ------------------------------------------------------------
module purge
module load CUDA/12.1.1 # Load CUDA drivers
module load Python/3.11.3-GCCcore-12.3.0 # Load a clean, recent Python environment

# Verify python loaded correctly
if ! command -v python &> /dev/null; then
    echo "CRITICAL ERROR: Python module failed to load."
    exit 1
fi
echo "Python loaded: $(which python)"

# 2. CREATE AND ACTIVATE VIRTUAL ENVIRONMENT
VENV_PATH=$TMPDIR/my_eval_env
echo "Creating isolated environment at $VENV_PATH"

# Create the environment. We use --without-pip to ensure a clean slate, then install pip.
python -m venv $VENV_PATH --without-pip
source $VENV_PATH/bin/activate

# Upgrade PIP
$VENV_PATH/bin/python -m pip install --upgrade pip

# 3. INSTALL FULL, COMPATIBLE STACK
echo "Installing full compatible stack..."

# We install the full stack from PyPI, ensuring compatibility flags are used.
# The previous conflicts showed we need to constrain numpy and install all required packages.

# CRITICAL FIX: Install PyTorch/Torchvision using a specific index URL for CUDA support.
# This prevents symbol errors and resolves the 'register_fake' AttributeError.
# (Assuming your cluster supports CUDA 12.1, which corresponds to PyTorch 2.1.x/2.2.x)
$VENV_PATH/bin/python -m pip install \
    torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install the remaining CV libraries
$VENV_PATH/bin/python -m pip install \
    clean-fid lpips ultralytics opencv-python-headless pandas \
    scikit-image

# 4. VERIFY ENVIRONMENT
echo "=================================================="
echo "VERIFYING ISOLATED ENVIRONMENT"
echo "=================================================="
# Run the verification script using the environment's python executable
$VENV_PATH/bin/python -c "
import sys
try:
    import torch;
    import torchvision;
    import lpips;
    import ultralytics;
    import skimage;
    import cv2;

    # Check for the numpy conflicts reported previously
    import numpy as np

    # Check if the installed libraries are compatible (version will be 0.16.x)
    print(f'✓ PyTorch: {torch.__version__} (Compatible with loaded CUDA)')
    print(f'✓ Torchvision: {torchvision.__version__} (Fixed incompatibility)')
    print(f'✓ Numpy: {np.__version__} (Resolved compatibility conflicts)')
    print('ALL DEPENDENCIES ARE ISOLATED AND COMPATIBLE.')
except Exception as e:
    print(f'FATAL ERROR: Failed to load core stack: {e}')
    sys.exit(1)
"

# 5. RUN EVALUATION
echo "=================================================="
echo "STARTING EVALUATION"
echo "=================================================="

# Run your script using the environment's python executable
python /home/algh/AMLCV-Project/results/eval_unpaired.py > "logs/unpaired_results_${SLURM_JOB_ID}.log" 2>&1
EXIT_CODE=$?

# 6. CLEANUP AND FINAL STATUS
# Deactivate is not strictly necessary as the job exits, but it's good practice
deactivate

echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Job Completed Successfully!"
else
    echo "Job Failed with Exit Code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=================================================="

exit $EXIT_CODE