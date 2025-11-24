#!/bin/bash
#SBATCH --job-name=Setup_Env
#SBATCH --output=logs/setup_env_${SLURM_JOB_ID}.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu=1
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --partition=acltr

echo "=================================================="
echo "CREATING COMPATIBLE VIRTUAL ENVIRONMENT"
echo "=================================================="

# 1. Clean start - remove existing environment
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi

# 2. Load Modules
module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0

# 3. Create fresh environment
echo "Creating new virtual environment..."
python -m venv venv
source venv/bin/activate

# 4. Upgrade pip and setuptools first
python -m pip install --upgrade pip setuptools wheel

# 5. Install NumPy 1.x FIRST (critical for compatibility)
echo "Installing NumPy 1.x for compatibility..."
python -m pip install "numpy<2.0"

# 6. Install PyTorch with compatible dependencies
echo "Installing PyTorch with CUDA 12.1..."
python -m pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# 7. Install other packages with version pins for stability
echo "Installing computer vision packages..."
python -m pip install \
    opencv-python-headless==4.8.1.78 \
    scikit-image==0.21.0 \
    clean-fid==0.6.6 \
    lpips==0.1.4 \
    ultralytics==8.0.186 \
    pandas==2.0.3 \
    scipy==1.11.1 \
    pillow==10.0.0 \
    tqdm==4.65.0 \
    matplotlib==3.7.1 \
    seaborn==0.12.2

# 8. VERIFY INSTALLATION WITH BETTER ERROR HANDLING
echo "=================================================="
echo "VERIFYING INSTALLATION"
echo "=================================================="

python -c "
import sys
try:
    # Test NumPy first (most critical)
    import numpy as np
    print(f'✓ NumPy {np.__version__}')
    assert np.__version__.startswith('1.'), f'NumPy version {np.__version__} may cause compatibility issues'

    # Core ML stack
    import torch
    import torchvision
    print(f'✓ PyTorch {torch.__version__}')
    print(f'✓ Torchvision {torchvision.__version__}')
    print(f'✓ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✓ GPU: {torch.cuda.get_device_name(0)}')

    # Image processing
    import cv2
    from skimage.metrics import structural_similarity as ssim
    print(f'✓ OpenCV {cv2.__version__}')
    print('✓ scikit-image')

    # Evaluation metrics
    import lpips
    import cleanfid
    print('✓ LPIPS and clean-fid')

    # Other essentials
    import ultralytics
    import pandas as pd
    print(f'✓ Ultralytics {ultralytics.__version__}')
    print(f'✓ pandas {pd.__version__}')

    print('')
    print('🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!')
    print('Environment is ready for evaluation scripts.')

except Exception as e:
    print(f'❌ ERROR: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo "=================================================="
echo "SETUP COMPLETE!"
echo "Environment location: $(pwd)/venv"
echo "=================================================="