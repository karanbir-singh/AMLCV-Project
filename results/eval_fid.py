"""
Interpreting Results:
FID Score:

< 10: Excellent quality
10-50: Good quality
50-100: Moderate quality
> 100: Poor quality

Inception Score:

> 10: Excellent
5-10: Good
3-5: Moderate
< 3: Poor

KID Score:

< 0.01: Excellent
0.01-0.05: Good
0.05-0.10: Moderate
> 0.10: Poor
"""

import os
from torch_fidelity import calculate_metrics

# Folder containing your CycleGAN output (Generated Day images)
GEN_FOLDER = 'day_flat_generated_output'

# Folder containing real day images
REF_FOLDER = 'flat_reference_input'


def calculate_all_metrics():

    # Check folders exist
    if not os.path.exists(GEN_FOLDER):
        print(f"ERROR: Generated folder '{GEN_FOLDER}' not found!")
        return

    if not os.path.exists(REF_FOLDER):
        print(f"ERROR: Reference folder '{REF_FOLDER}' not found!")
        return

    # Count images
    gen_images = len([f for f in os.listdir(GEN_FOLDER)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    ref_images = len([f for f in os.listdir(REF_FOLDER)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"\nGenerated images: {gen_images}")
    print(f"Reference images: {ref_images}")

    if gen_images == 0 or ref_images == 0:
        print("ERROR: No images found in one or both folders!")
        return

    print("\n" + "=" * 60)
    print("Computing metrics (this may take a few minutes)...")
    print("=" * 60)

    # Use torch-fidelity for ALL metrics (avoids Windows multiprocessing issues)
    print("\nCalculating FID, IS, and KID using torch-fidelity...")
    try:
        metrics = calculate_metrics(
            input1=GEN_FOLDER,
            input2=REF_FOLDER,
            cuda=False,
            isc=True,  # Inception Score
            fid=True,  # FID Score
            kid=True,  # Kernel Inception Distance
            verbose=False,
            batch_size=1  # Reduce batch size for stability
        )

        # Extract all metrics
        fid_score = metrics['frechet_inception_distance']
        is_mean = metrics['inception_score_mean']
        is_std = metrics['inception_score_std']
        kid_mean = metrics['kernel_inception_distance_mean']
        kid_std = metrics['kernel_inception_distance_std']

        print(f"✓ FID Score: {fid_score:.4f}")
        print(f"✓ Inception Score: {is_mean:.4f} ± {is_std:.4f}")
        print(f"✓ KID Score: {kid_mean:.6f} ± {kid_std:.6f}")

    except Exception as e:
        print(f"✗ Metric calculation failed: {e}")
        print("\nTrying alternative approach...")
        return calculate_metrics_alternative()

    # Print interpretation (same as before)
    print_interpretation(fid_score, is_mean, is_std, kid_mean, kid_std)

    return {
        'fid': float(fid_score),
        'is_mean': float(is_mean),
        'is_std': float(is_std),
        'kid_mean': float(kid_mean),
        'kid_std': float(kid_std)
    }


def calculate_metrics_alternative():
    """Alternative calculation if main method fails"""
    try:
        # Force CPU and single processing
        metrics = calculate_metrics(
            input1=GEN_FOLDER,
            input2=REF_FOLDER,
            cuda=False,
            isc=True,
            fid=True,
            kid=True,
            verbose=False,
            batch_size=1,
            rng_seed=42
        )
        return metrics
    except Exception as e:
        print(f"✗ Alternative method also failed: {e}")
        return None

if __name__ == '__main__':
    results = calculate_all_metrics()

    # Save results
    if results:
        import json

        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(" Results saved to evaluation_results.json")