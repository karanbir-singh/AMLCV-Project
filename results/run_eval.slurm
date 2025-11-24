#!/bin/bash
#SBATCH --job-name=Eval_Job
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu=1
#SBATCH --partition=acltr
#SBATCH --time=24:00:00
#SBATCH --mem=32G

# ------------------------------------------------------------
# 1. LOAD BASE PYTHON MODULE (Crucial: DO NOT load PyTorch module here)
# ------------------------------------------------------------
module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0


pip uninstall -y torch torchvision triton numpy clean-fid lpips opencv-python opencv-python-headless scikit-image

# Install compatible versions
pip install --user numpy==1.24.3
pip install --user torch==2.1.2 torchvision==0.16.2
pip install --user clean-fid==0.1.35
pip install --user lpips==0.1.4
pip install --user opencv-python-headless==4.8.1.78
pip install --user scikit-image==0.21.0
pip install --user ultralytics
pip install --user pandas

# Verify installation
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torchvision; print('torchvision:', torchvision.__version__)"
python -c "import lpips; print('LPIPS: OK')"
python -c "import cv2; print('OpenCV: OK')"
echo "Start Time: $(date)"

echo "=================================================="
echo "STARTING EVALUATION"
echo "=================================================="
python eval_unpaired.py > "./logs/eval_results_${SLURM_JOB_ID}.log" 2>&1


EXIT_CODE=$?




echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Job Completed Successfully!"
else
    echo "Job Failed with Exit Code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=================================================="

exit $EXIT_CODE