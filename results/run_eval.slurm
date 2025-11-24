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
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0
module load Anaconda3/2025.06-1

# Verify python loaded correctly
if ! command -v python &> /dev/null; then
    echo "CRITICAL ERROR: Python module failed to load."
    exit 1
fi
echo "Python loaded: $(which python)"

if ! conda info --envs | grep -q "amlcv"; then
    echo "Creating conda environment..."
    conda create -y -n amlcv python=3.11
fi

conda activate amlcv

echo "Installing packages in conda environment..."

pip install scikit-image opencv-python-headless pandas clean-fid lpips ultralytics

# 4. VERIFY
echo "Verifying installation..."
python -c "
import sys
try:
    import torch; print(f'PyTorch: {torch.__version__}')
    import torchvision; print(f'Torchvision: {torchvision.__version__}')
    from skimage.metrics import structural_similarity as ssim; print('scikit-image OK')
    import lpips; print('LPIPS OK')
    print('ALL IMPORTS WORKING')
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
"


# 5. RUN EVALUATION

echo "Start Time: $(date)"

echo "=================================================="
echo "STARTING EVALUATION"
echo "=================================================="

# Run your script using the environment's python executable
python eval_unpaired.py > "logs/unpaired_results_${SLURM_JOB_ID}.log" 2>&1
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