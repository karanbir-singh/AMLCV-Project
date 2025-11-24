import os
import cv2
import numpy as np
import torch
import lpips
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim

# --- CONFIGURATION ---
DAY_PATH = 'flat_reference_input'
NIGHT_PATH = 'flat_generated_output'
OUTPUT_CSV = 'evaluation_results.csv'
SUMMARY_CSV = 'summary_stats.csv'

# Load Models
print("Loading models...")
lpips_fn = lpips.LPIPS(net='alex').cuda() if torch.cuda.is_available() else lpips.LPIPS(net='alex')
yolo_model = YOLO('yolov8n.pt')


def load_image_tensor(path):
    """Load image for LPIPS (Tensor, -1 to 1 normalized)"""
    img = Image.open(path).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return tf(img).unsqueeze(0)


def calculate_object_consistency(img_day_path, img_night_path):
    """
    Returns object consistency score (0.0 to 1.0).
    Measures how many day objects are preserved in night image.
    """
    res_day = yolo_model(img_day_path, verbose=False)[0]
    res_night = yolo_model(img_night_path, verbose=False)[0]

    boxes_day = res_day.boxes.data.cpu().numpy()
    boxes_night = res_night.boxes.data.cpu().numpy()

    # Handle edge cases
    if len(boxes_day) == 0 and len(boxes_night) == 0:
        return 1.0  # Both empty = consistent
    if len(boxes_day) == 0:
        return 1.0  # No objects to preserve
    if len(boxes_night) == 0:
        return 0.0  # Lost all objects

    matched_count = 0

    for d_box in boxes_day:
        d_cls = d_box[5]
        d_coords = d_box[:4]
        best_iou = 0

        for n_box in boxes_night:
            n_cls = n_box[5]
            n_coords = n_box[:4]

            if d_cls == n_cls:
                # Calculate IoU
                xA = max(d_coords[0], n_coords[0])
                yA = max(d_coords[1], n_coords[1])
                xB = min(d_coords[2], n_coords[2])
                yB = min(d_coords[3], n_coords[3])

                interArea = max(0, xB - xA) * max(0, yB - yA)
                boxAArea = (d_coords[2] - d_coords[0]) * (d_coords[3] - d_coords[1])
                boxBArea = (n_coords[2] - n_coords[0]) * (n_coords[3] - n_coords[1])
                iou = interArea / float(boxAArea + boxBArea - interArea + 1e-8)

                best_iou = max(best_iou, iou)

        if best_iou > 0.3:
            matched_count += 1

    return matched_count / len(boxes_day)


def calculate_ssim_score(img_day, img_night):
    """Calculate structural similarity (geometry preservation)"""
    gray_day = cv2.cvtColor(img_day, cv2.COLOR_BGR2GRAY)
    gray_night = cv2.cvtColor(img_night, cv2.COLOR_BGR2GRAY)

    if gray_day.shape != gray_night.shape:
        gray_night = cv2.resize(gray_night, (gray_day.shape[1], gray_day.shape[0]))

    return ssim(gray_day, gray_night)


def calculate_night_characteristics(img_day, img_night):
    """Analyze if image has proper night characteristics"""

    # LAB color space
    lab_day = cv2.cvtColor(img_day, cv2.COLOR_BGR2LAB)
    lab_night = cv2.cvtColor(img_night, cv2.COLOR_BGR2LAB)

    # HSV color space
    hsv_day = cv2.cvtColor(img_day, cv2.COLOR_BGR2HSV)
    hsv_night = cv2.cvtColor(img_night, cv2.COLOR_BGR2HSV)

    metrics = {
        # Brightness reduction (should be 30-80 for good night effect)
        'brightness_reduction': np.mean(lab_day[:, :, 0]) - np.mean(lab_night[:, :, 0]),

        # Blue shift (positive = more blue, typical for night)
        'blue_shift': np.mean(lab_day[:, :, 2]) - np.mean(lab_night[:, :, 2]),

        # Saturation reduction (night should be less saturated)
        'saturation_reduction': np.mean(hsv_day[:, :, 1]) - np.mean(hsv_night[:, :, 1])
    }

    return metrics


def calculate_edge_preservation(img_day, img_night):
    gray_day = cv2.cvtColor(img_day, cv2.COLOR_BGR2GRAY)
    gray_night = cv2.cvtColor(img_night, cv2.COLOR_BGR2GRAY)

    if gray_day.shape != gray_night.shape:
        gray_night = cv2.resize(gray_night, (gray_day.shape[1], gray_day.shape[0]))

    gray_day_norm = cv2.normalize(gray_day, None, 0, 255, cv2.NORM_MINMAX)
    gray_night_norm = cv2.normalize(gray_night, None, 0, 255, cv2.NORM_MINMAX)

    # Use same thresholds for both
    edges_day = cv2.Canny(gray_day_norm, 100, 200)
    edges_night = cv2.Canny(gray_night_norm, 100, 200)

    intersection = np.logical_and(edges_day, edges_night).sum()
    union = np.logical_or(edges_day, edges_night).sum()

    return intersection / (union + 1e-8)

def interpret_results(row):
    """Automatically flag potential issues"""
    issues = []

    if row['object_consistency'] < 0.5:
        issues.append("Lost_objects")
    if row['brightness_reduction'] < 20:
        issues.append("Too_bright")
    elif row['brightness_reduction'] > 100:
        issues.append("Too_dark")
    if row['ssim'] < 0.6:
        issues.append("Structure_degraded")
    if row['lpips'] < 0.3:
        issues.append("Minimal_transformation")
    elif row['lpips'] > 0.8:
        issues.append("Over_transformation")
    if row['edge_preservation'] < 0.3:
        issues.append("Details_lost")
    if row['blue_shift'] < 0:
        issues.append("Wrong_color_temp")

    return '; '.join(issues) if issues else 'OK'


def main():
    results = []
    errors = []
    print("Starting evaluation...")

    day_files = sorted([f for f in os.listdir(DAY_PATH)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for i, fname in enumerate(day_files, 1):
        print(f"Processing {i}/{len(day_files)}: {fname}")

        try:
            day_full_path = os.path.join(DAY_PATH, fname)
            night_full_path = os.path.join(NIGHT_PATH, fname)

            if not os.path.exists(night_full_path):
                errors.append(f"{fname}: No matching output")
                continue

            # Load images
            cv_day = cv2.imread(day_full_path)
            cv_night = cv2.imread(night_full_path)

            if cv_day is None or cv_night is None:
                errors.append(f"{fname}: Failed to load images")
                continue

            # 1. LPIPS (transformation magnitude)
            t_day = load_image_tensor(day_full_path)
            t_night = load_image_tensor(night_full_path)
            if torch.cuda.is_available():
                t_day, t_night = t_day.cuda(), t_night.cuda()
            lpips_val = lpips_fn(t_day, t_night).item()

            # 2. Object Consistency
            obj_score = calculate_object_consistency(day_full_path, night_full_path)

            # 3. SSIM (structure preservation)
            ssim_val = calculate_ssim_score(cv_day, cv_night)

            # 4. Night characteristics
            night_chars = calculate_night_characteristics(cv_day, cv_night)

            # 5. Edge preservation
            edge_pres = calculate_edge_preservation(cv_day, cv_night)

            result = {
                'image_name': fname,
                'object_consistency': obj_score,
                'ssim': ssim_val,
                'brightness_reduction': night_chars['brightness_reduction'],
                'blue_shift': night_chars['blue_shift'],
                'saturation_reduction': night_chars['saturation_reduction'],
                'edge_preservation': edge_pres,
                'lpips': lpips_val
            }

            # Add automatic issue detection
            result['issues'] = interpret_results(result)

            results.append(result)

            print(f"  Obj: {obj_score:.2f} | SSIM: {ssim_val:.2f} | "
                  f"Bright↓: {night_chars['brightness_reduction']:.1f} | "
                  f"LPIPS: {lpips_val:.3f}")

        except Exception as e:
            errors.append(f"{fname}: {str(e)}")
            print(f"  ERROR: {e}")

    if not results:
        print("No results to save!")
        return

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save detailed results
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Detailed results saved to {OUTPUT_CSV}")

    # Calculate and save summary statistics
    numeric_cols = ['object_consistency', 'ssim', 'brightness_reduction',
                    'blue_shift', 'saturation_reduction', 'edge_preservation', 'lpips']

    summary = pd.DataFrame({
        'metric': numeric_cols,
        'mean': [df[col].mean() for col in numeric_cols],
        'std': [df[col].std() for col in numeric_cols],
        'min': [df[col].min() for col in numeric_cols],
        'max': [df[col].max() for col in numeric_cols],
        'median': [df[col].median() for col in numeric_cols]
    })

    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"✓ Summary statistics saved to {SUMMARY_CSV}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for _, row in summary.iterrows():
        print(f"{row['metric']:25s}: {row['mean']:6.3f} ± {row['std']:5.3f} "
              f"(range: {row['min']:.3f} - {row['max']:.3f})")

    print("\n" + "=" * 60)
    print("QUALITY INTERPRETATION")
    print("=" * 60)
    avg_obj = df['object_consistency'].mean()
    avg_bright = df['brightness_reduction'].mean()
    avg_ssim = df['ssim'].mean()
    avg_lpips = df['lpips'].mean()
    avg_edge = df['edge_preservation'].mean()

    print(f"Object Consistency ({avg_obj:.3f}):",
          "✓ Excellent" if avg_obj >= 0.7 else "✓ Good" if avg_obj >= 0.5 else "✗ Poor")
    print(f"Brightness Reduction ({avg_bright:.1f}):",
          "✓ Good night effect" if 30 <= avg_bright <= 80 else "⚠ Check manually")
    print(f"SSIM ({avg_ssim:.3f}):",
          "✓ Good structure" if avg_ssim >= 0.7 else "⚠ Moderate changes" if avg_ssim >= 0.5 else "✗ Structure issues")
    print(f"LPIPS ({avg_lpips:.3f}):",
          "✓ Reasonable transformation" if 0.3 <= avg_lpips <= 0.8 else "⚠ Check manually")
    print(f"Edge Preservation ({avg_edge:.3f}):",
          "✓ Details preserved" if avg_edge >= 0.5 else "⚠ Some detail loss" if avg_edge >= 0.3 else "✗ Significant blur")

    # Show problematic images
    problem_images = df[df['issues'] != 'OK']
    if len(problem_images) > 0:
        print("\n" + "=" * 60)
        print(f"FLAGGED IMAGES ({len(problem_images)}/{len(df)})")
        print("=" * 60)
        for _, row in problem_images.head(10).iterrows():
            print(f"{row['image_name']:30s}: {row['issues']}")
        if len(problem_images) > 10:
            print(f"... and {len(problem_images) - 10} more (see {OUTPUT_CSV})")
    else:
        print("\n✓ No major issues detected!")

    # Show errors
    if errors:
        print("\n" + "=" * 60)
        print(f"ERRORS ({len(errors)})")
        print("=" * 60)
        for err in errors[:10]:
            print(f"  {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == '__main__':
    main()