#!/bin/bash
#SBATCH --job-name=CycleGAN_Eval
#SBATCH --output=logs/eval_metrics_%j.out
#SBATCH --error=logs/eval_metrics_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu=1
#SBATCH --partition=acltr
#SBATCH --time=04:00:00
#SBATCH --mem=32G

mkdir -p logs

echo "=================================================="
echo "CycleGAN Evaluation Job"
echo "=================================================="

# 1. LOAD MODULES
# These provide the heavy drivers (CUDA, PyTorch) so we don't need to download them
module purge
module load CUDA/12.1.1
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# 2. INSTALL ONLY MISSING LIBRARIES
echo "Installing/Updating evaluation libraries..."

# We use --user to install in your home folder.
# We DO NOT include 'torch' or 'torchvision' here because the modules provide them.
pip install --user --upgrade clean-fid lpips ultralytics opencv-python-headless pandas

# 3. VERIFY EVERYTHING IS THERE
echo "Verifying imports..."
python -c "
import sys
try:
    import torch; print(f'✓ PyTorch: {torch.__version__}')
    import torchvision; print(f'✓ Torchvision: {torchvision.__version__}')
    import cv2; print(f'✓ OpenCV: {cv2.__version__}')
    import cleanfid; print('✓ clean-fid (FID) available')
    import lpips; print('✓ LPIPS available')
    import ultralytics; print('✓ Ultralytics (YOLO) available')
    print('ALL SYSTEMS GO')
except Exception as e:
    print(f'ERROR: {e}')
    # We do NOT exit here, we try to run anyway just in case
"

echo "=================================================="
echo "RUNNING EVALUATION 2: Object Consist., Brightness, LPIPS"
echo "=================================================="

python eval_unpaired.py > "logs/unpaired_results_${SLURM_JOB_ID}.log" 2>&1
echo "Paired Metrics Run Finished. Check logs/paired_results_${SLURM_JOB_ID}.log"

echo "=================================================="
echo "Job Complete"