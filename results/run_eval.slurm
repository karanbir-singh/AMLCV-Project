#!/bin/bash
#SBATCH --job-name=Eval_Job
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu=1
#SBATCH --partition=acltr
#SBATCH --time=5:00:00
#SBATCH --mem=32G

# ------------------------------------------------------------
# 1. LOAD BASE PYTHON MODULE (Crucial: DO NOT load PyTorch module here)
# ------------------------------------------------------------
module purge
module load CUDA/12.1.1

# 2. ACTIVATE YOUR PERSISTENT ENVIRONMENT
source /home/algh/AMLCV-Project/venv/bin/activate

# 3. VERIFY ENVIRONMENT
echo "Using Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 5. RUN EVALUATION

echo "Start Time: $(date)"

echo "=================================================="
echo "STARTING EVALUATION"
echo "=================================================="

cd /home/algh/AMLCV-Project/results
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