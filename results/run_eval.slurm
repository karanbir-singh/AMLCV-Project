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


# 2. ACTIVATE YOUR PERSISTENT ENVIRONMENT
source venv/bin/activate

pip install cv2

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