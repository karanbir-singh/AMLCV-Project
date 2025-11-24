#!/bin/bash
#SBATCH --job-name=Setup_Env
#SBATCH --output=logs/setup_env_${SLURM_JOB_ID}.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu=1
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --partition=acltr

echo "=================================================="
echo "CREATING PERSISTENT VIRTUAL ENVIRONMENT"
echo "=================================================="

# 1. Load Modules - Use the SAME Python that PyTorch module uses
module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0 # This matches PyTorch/2.1.2-foss-2023a

# 2. Create Persistent Environment
echo "Creating virtual environment in $(pwd)/venv"
python -m venv venv

# 3. Activate and Upgrade pip
source venv/bin/activate
python -m pip install --upgrade pip

# 4. Install Libraries with Compatible Versions
echo "Installing libraries with compatible versions..."

# Install PyTorch with CUDA 12.1
echo "Installing PyTorch..."
python -m pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Install computer vision and evaluation packages
echo "Installing computer vision packages..."
python -m pip install \
    opencv-python-headless \
    scikit-image \
    clean-fid \
    lpips \
    ultralytics \
    pandas \
    scipy \
    pillow \
    tqdm \
    matplotlib \
    seaborn

python -m pip install "numpy<2.0"

# 5. VERIFY INSTALLATION
echo "=================================================="
echo "VERIFYING INSTALLATION"
echo "=================================================="

python -c "
import sys
try:
    # Core ML stack
    import torch
    import torchvision
    print(f'✓ PyTorch {torch.__version__}')
    print(f'✓ Torchvision {torchvision.__version__}')
    print(f'✓ CUDA available: {torch.cuda.is_available()}')

    # Image processing
    import cv2
    from skimage.metrics import structural_similarity as ssim
    print('✓ OpenCV and scikit-image')

    # Evaluation metrics
    import lpips
    import cleanfid
    print('✓ LPIPS and clean-fid')

    # Other essentials
    import ultralytics
    import pandas
    print('✓ Ultralytics and pandas')

    print('')
    print('🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!')
    print('You can now activate this environment with:')
    print('  source venv/bin/activate')

except Exception as e:
    print(f'❌ ERROR: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo "=================================================="
echo "SETUP COMPLETE!"
echo "Environment location: $(pwd)/venv"
echo "To use this environment in future jobs, add to your SLURM scripts:"
echo "  source $(pwd)/venv/bin/activate"
echo "=================================================="