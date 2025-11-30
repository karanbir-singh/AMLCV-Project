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
NIGHT_PATH = 'flat_generated_output'  # Original night images
DAY_PATH = 'day_flat_generated_output'  # Generated day images
OUTPUT_CSV = 'night_to_day_results.csv'
SUMMARY_CSV = 'night_to_day_summary.csv'

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


def calculate_object_consistency(img_night_path, img_day_path):
    """
    Returns object consistency score (0.0 to 1.0).
    Measures how many night objects are preserved in day image.
    For night→day, we expect IMPROVED detection (more objects visible).
    """
    res_night = yolo_model(img_night_path, verbose=False)[0]
    res_day = yolo_model(img_day_path, verbose=False)[0]

    boxes_night = res_night.boxes.data.cpu().numpy()
    boxes_day = res_day.boxes.data.cpu().numpy()

    # Handle edge cases
    if len(boxes_night) == 0 and len(boxes_day) == 0:
        return 1.0  # Both empty = consistent
    if len(boxes_night) == 0:
        # No objects in night, but objects appeared in day = GOOD for night→day!
        return 1.0  # Consider this successful illumination
    if len(boxes_day) == 0:
        return 0.0  # Lost all objects (bad)

    matched_count = 0

    for n_box in boxes_night:
        n_cls = n_box[5]
        n_coords = n_box[:4]
        best_iou = 0

        for d_box in boxes_day:
            d_cls = d_box[5]
            d_coords = d_box[:4]

            if n_cls == d_cls:
                # Calculate IoU
                xA = max(n_coords[0], d_coords[0])
                yA = max(n_coords[1], d_coords[1])
                xB = min(n_coords[2], d_coords[2])
                yB = min(n_coords[3], d_coords[3])

                interArea = max(0, xB - xA) * max(0, yB - yA)
                boxAArea = (n_coords[2] - n_coords[0]) * (n_coords[3] - n_coords[1])
                boxBArea = (d_coords[2] - d_coords[0]) * (d_coords[3] - d_coords[1])
                iou = interArea / float(boxAArea + boxBArea - interArea + 1e-8)

                best_iou = max(best_iou, iou)

        if best_iou > 0.3:
            matched_count += 1

    base_score = matched_count / len(boxes_night)

    # BONUS: If more objects detected in day than night (good for night→day)
    if len(boxes_day) > len(boxes_night):
        improvement_bonus = min(0.2, (len(boxes_day) - len(boxes_night)) / len(boxes_night) * 0.1)
        return min(1.0, base_score + improvement_bonus)

    return base_score


def calculate_object_visibility_improvement(img_night_path, img_day_path):
    """
    NEW METRIC: How many MORE objects are visible in day vs night?
    Returns improvement ratio (can be >1.0 if many new objects appear)
    """
    res_night = yolo_model(img_night_path, verbose=False)[0]
    res_day = yolo_model(img_day_path, verbose=False)[0]

    boxes_night = res_night.boxes.data.cpu().numpy()
    boxes_day = res_day.boxes.data.cpu().numpy()

    if len(boxes_night) == 0:
        return len(boxes_day) / 1.0  # Any objects in day is improvement

    improvement = len(boxes_day) / len(boxes_night)
    return improvement


def calculate_ssim_score(img_night, img_day):
    """Calculate structural similarity (geometry preservation)"""
    gray_night = cv2.cvtColor(img_night, cv2.COLOR_BGR2GRAY)
    gray_day = cv2.cvtColor(img_day, cv2.COLOR_BGR2GRAY)

    if gray_night.shape != gray_day.shape:
        gray_day = cv2.resize(gray_day, (gray_night.shape[1], gray_night.shape[0]))

    return ssim(gray_night, gray_day)


def calculate_day_characteristics(img_night, img_day):
    """
    Analyze if image has proper DAY characteristics.
    For night→day, we want:
    - BRIGHTNESS INCREASE (opposite of night)
    - WARM/NEUTRAL colors (not blue)
    - SATURATION INCREASE (colors more vivid)
    """
    # LAB color space
    lab_night = cv2.cvtColor(img_night, cv2.COLOR_BGR2LAB)
    lab_day = cv2.cvtColor(img_day, cv2.COLOR_BGR2LAB)

    # HSV color space
    hsv_night = cv2.cvtColor(img_night, cv2.COLOR_BGR2HSV)
    hsv_day = cv2.cvtColor(img_day, cv2.COLOR_BGR2HSV)

    metrics = {
        # Brightness INCREASE (should be 30-80 for good day effect)
        # INVERTED from day→night!
        'brightness_increase': np.mean(lab_day[:, :, 0]) - np.mean(lab_night[:, :, 0]),

        # Warm shift (negative = warmer, positive = cooler)
        # For day: should be NEGATIVE (warmer than night)
        'warm_shift': np.mean(lab_night[:, :, 2]) - np.mean(lab_day[:, :, 2]),

        # Saturation INCREASE (day should be more saturated than night)
        # INVERTED from day→night!
        'saturation_increase': np.mean(hsv_day[:, :, 1]) - np.mean(hsv_night[:, :, 1])
    }

    return metrics


def calculate_edge_preservation(img_night, img_day):
    """Edge preservation - same as before"""
    gray_night = cv2.cvtColor(img_night, cv2.COLOR_BGR2GRAY)
    gray_day = cv2.cvtColor(img_day, cv2.COLOR_BGR2GRAY)

    if gray_night.shape != gray_day.shape:
        gray_day = cv2.resize(gray_day, (gray_night.shape[1], gray_night.shape[0]))

    gray_night_norm = cv2.normalize(gray_night, None, 0, 255, cv2.NORM_MINMAX)
    gray_day_norm = cv2.normalize(gray_day, None, 0, 255, cv2.NORM_MINMAX)

    edges_night = cv2.Canny(gray_night_norm, 100, 200)
    edges_day = cv2.Canny(gray_day_norm, 100, 200)

    intersection = np.logical_and(edges_night, edges_day).sum()
    union = np.logical_or(edges_night, edges_day).sum()

    return intersection / (union + 1e-8)


def calculate_shadow_removal(img_night, img_day):
    """
    NEW METRIC: Assess shadow removal quality.
    Measures local brightness variance - day should be more uniform.
    """
    gray_night = cv2.cvtColor(img_night, cv2.COLOR_BGR2GRAY)
    gray_day = cv2.cvtColor(img_day, cv2.COLOR_BGR2GRAY)

    if gray_night.shape != gray_day.shape:
        gray_day = cv2.resize(gray_day, (gray_night.shape[1], gray_night.shape[0]))

    # Calculate local standard deviation (measure of shadow intensity)
    def local_std(img, ksize=15):
        mean = cv2.blur(img.astype(float), (ksize, ksize))
        sqr_mean = cv2.blur((img.astype(float) ** 2), (ksize, ksize))
        return np.sqrt(np.maximum(sqr_mean - mean ** 2, 0))

    std_night = local_std(gray_night)
    std_day = local_std(gray_day)

    # Shadow removal score: how much the variance decreased
    avg_std_night = np.mean(std_night)
    avg_std_day = np.mean(std_day)

    if avg_std_night == 0:
        return 1.0

    reduction = (avg_std_night - avg_std_day) / avg_std_night
    return max(0.0, min(1.0, reduction))


def calculate_overexposure_score(img_day):
    """
    NEW METRIC: Check for overexposure (too bright/washed out).
    Returns score 0-1 where 1 = no overexposure, 0 = severe overexposure.
    """
    # Convert to LAB
    lab = cv2.cvtColor(img_day, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # Check percentage of pixels that are very bright (>240 in 0-255 scale)
    overexposed_pixels = np.sum(l_channel > 240)
    total_pixels = l_channel.size
    overexposure_ratio = overexposed_pixels / total_pixels

    # Also check for washed-out colors (low saturation in bright areas)
    hsv = cv2.cvtColor(img_day, cv2.COLOR_BGR2HSV)
    bright_areas = l_channel > 200
    if np.sum(bright_areas) > 0:
        avg_saturation_bright = np.mean(hsv[:, :, 1][bright_areas])
        saturation_penalty = max(0, (50 - avg_saturation_bright) / 50)  # Penalty if sat < 50
    else:
        saturation_penalty = 0

    # Combined overexposure score
    overexposure_score = 1.0 - (overexposure_ratio * 2 + saturation_penalty * 0.5)
    return max(0.0, min(1.0, overexposure_score))


def interpret_results(row):
    """
    Automatically flag potential issues for NIGHT→DAY transformation.
    Different thresholds than day→night!
    """
    issues = []

    # Object consistency
    if row['object_consistency'] < 0.5:
        issues.append("Lost_objects")

    # Brightness increase (opposite of day→night)
    if row['brightness_increase'] < 20:
        issues.append("Insufficient_brightening")
    elif row['brightness_increase'] > 120:
        issues.append("Over_brightened")

    # Structure preservation
    if row['ssim'] < 0.6:
        issues.append("Structure_degraded")

    # Transformation magnitude
    if row['lpips'] < 0.3:
        issues.append("Minimal_transformation")
    elif row['lpips'] > 0.8:
        issues.append("Over_transformation")

    # Edge preservation
    if row['edge_preservation'] < 0.3:
        issues.append("Details_lost")

    # Color temperature (for day, should be warm/neutral, not blue)
    if row['warm_shift'] < 0:  # Negative means it got COOLER (wrong for day)
        issues.append("Wrong_color_temp")

    # Saturation increase (day should be more saturated)
    if row['saturation_increase'] < 0:
        issues.append("Undersaturated")

    # Overexposure check
    if row['overexposure_score'] < 0.6:
        issues.append("Overexposed")

    # Shadow removal
    if row['shadow_removal'] < 0.2:
        issues.append("Shadows_remain")

    return '; '.join(issues) if issues else 'OK'


def main():
    results = []
    errors = []
    print("Starting night→day evaluation...")

    night_files = sorted([f for f in os.listdir(NIGHT_PATH)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for i, fname in enumerate(night_files, 1):
        print(f"Processing {i}/{len(night_files)}: {fname}")

        try:
            night_full_path = os.path.join(NIGHT_PATH, fname)
            day_full_path = os.path.join(DAY_PATH, fname)

            if not os.path.exists(day_full_path):
                errors.append(f"{fname}: No matching output")
                continue

            # Load images
            cv_night = cv2.imread(night_full_path)
            cv_day = cv2.imread(day_full_path)

            if cv_night is None or cv_day is None:
                errors.append(f"{fname}: Failed to load images")
                continue

            # 1. LPIPS (transformation magnitude)
            t_night = load_image_tensor(night_full_path)
            t_day = load_image_tensor(day_full_path)
            if torch.cuda.is_available():
                t_night, t_day = t_night.cuda(), t_day.cuda()
            lpips_val = lpips_fn(t_night, t_day).item()

            # 2. Object Consistency
            obj_score = calculate_object_consistency(night_full_path, day_full_path)

            # 3. Object Visibility Improvement (NEW)
            obj_improvement = calculate_object_visibility_improvement(night_full_path, day_full_path)

            # 4. SSIM (structure preservation)
            ssim_val = calculate_ssim_score(cv_night, cv_day)

            # 5. Day characteristics
            day_chars = calculate_day_characteristics(cv_night, cv_day)

            # 6. Edge preservation
            edge_pres = calculate_edge_preservation(cv_night, cv_day)

            # 7. Shadow removal (NEW)
            shadow_removal = calculate_shadow_removal(cv_night, cv_day)

            # 8. Overexposure check (NEW)
            overexposure = calculate_overexposure_score(cv_day)

            result = {
                'image_name': fname,
                'object_consistency': obj_score,
                'object_visibility_improvement': obj_improvement,
                'ssim': ssim_val,
                'brightness_increase': day_chars['brightness_increase'],
                'warm_shift': day_chars['warm_shift'],
                'saturation_increase': day_chars['saturation_increase'],
                'edge_preservation': edge_pres,
                'shadow_removal': shadow_removal,
                'overexposure_score': overexposure,
                'lpips': lpips_val
            }

            # Add automatic issue detection
            result['issues'] = interpret_results(result)

            results.append(result)

            print(f"  Obj: {obj_score:.2f} (+{obj_improvement:.2f}x) | SSIM: {ssim_val:.2f} | "
                  f"Bright↑: {day_chars['brightness_increase']:.1f} | "
                  f"Overexp: {overexposure:.2f} | LPIPS: {lpips_val:.3f}")

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
    numeric_cols = ['object_consistency', 'object_visibility_improvement', 'ssim',
                    'brightness_increase', 'warm_shift', 'saturation_increase',
                    'edge_preservation', 'shadow_removal', 'overexposure_score', 'lpips']

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
    print("NIGHT→DAY EVALUATION SUMMARY")
    print("=" * 60)
    for _, row in summary.iterrows():
        print(f"{row['metric']:30s}: {row['mean']:6.3f} ± {row['std']:5.3f} "
              f"(range: {row['min']:.3f} - {row['max']:.3f})")

    print("\n" + "=" * 60)
    print("QUALITY INTERPRETATION")
    print("=" * 60)

    avg_obj = df['object_consistency'].mean()
    avg_obj_imp = df['object_visibility_improvement'].mean()
    avg_bright = df['brightness_increase'].mean()
    avg_ssim = df['ssim'].mean()
    avg_lpips = df['lpips'].mean()
    avg_edge = df['edge_preservation'].mean()
    avg_shadow = df['shadow_removal'].mean()
    avg_overexp = df['overexposure_score'].mean()

    print(f"Object Consistency ({avg_obj:.3f}):",
          "✓ Excellent" if avg_obj >= 0.7 else "✓ Good" if avg_obj >= 0.5 else "✗ Poor")
    print(f"Object Visibility (+{avg_obj_imp:.2f}x):",
          "✓ Significant improvement" if avg_obj_imp >= 1.2 else "✓ Some improvement" if avg_obj_imp >= 1.0 else "✗ Objects lost")
    print(f"Brightness Increase ({avg_bright:.1f}):",
          "✓ Good day effect" if 30 <= avg_bright <= 100 else "⚠ Check manually")
    print(f"SSIM ({avg_ssim:.3f}):",
          "✓ Good structure" if avg_ssim >= 0.7 else "⚠ Moderate changes" if avg_ssim >= 0.5 else "✗ Structure issues")
    print(f"LPIPS ({avg_lpips:.3f}):",
          "✓ Reasonable transformation" if 0.3 <= avg_lpips <= 0.8 else "⚠ Check manually")
    print(f"Edge Preservation ({avg_edge:.3f}):",
          "✓ Details preserved" if avg_edge >= 0.5 else "⚠ Some detail loss" if avg_edge >= 0.3 else "✗ Significant blur")
    print(f"Shadow Removal ({avg_shadow:.3f}):",
          "✓ Good shadow removal" if avg_shadow >= 0.5 else "⚠ Some shadows remain" if avg_shadow >= 0.3 else "✗ Poor shadow removal")
    print(f"Overexposure Control ({avg_overexp:.3f}):",
          "✓ Well balanced" if avg_overexp >= 0.8 else "⚠ Some overexposure" if avg_overexp >= 0.6 else "✗ Significant overexposure")

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