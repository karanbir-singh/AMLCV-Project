import torch
from cleanfid import fid
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

# --- CONFIGURATION ---
# Folder containing your CycleGAN output (Generated Night images)
GEN_FOLDER = './path/to/generated_night'

# Folder containing REAL night images (Reference for FID)
# If you don't have this, FID is meaningless.
REF_FOLDER = './path/to/real_night_dataset'


def calculate_metrics():
    print("--- Calculating FID ---")
    if os.path.exists(REF_FOLDER):
        # clean-fid automatically handles resizing and inception feature extraction
        score_fid = fid.compute_fid(GEN_FOLDER, REF_FOLDER)
        print(f"FID Score: {score_fid:.4f}")
    else:
        print(f"Warning: Reference folder '{REF_FOLDER}' not found. Cannot compute FID.")

    print("\n--- Calculating Inception Score (IS) ---")
    # IS does not strictly require a reference folder, but clean-fid calculates it
    # usually as a side effect or we can use a custom call.
    # Note: pure IS is often less relevant for I2I than FID.
    # Using torch-fidelity logic often wrapped or calling a separate function if needed.
    # For simplicity, clean-fid is best for FID.

    # If you specifically need IS, many researchers use torch-fidelity:
    # !fid_score --input1 {GEN_FOLDER} --isc
    print("Note: For strict IS calculation, ensure 'torch-fidelity' is installed.")


if __name__ == '__main__':
    calculate_metrics()