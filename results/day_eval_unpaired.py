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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_image_tensor(path):
    """Load image for LPIPS (Tensor, -1 to 1 normalized)"""
    try:
        img = Image.open(path).convert('RGB')
        tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return tf(img).unsqueeze(0)
    except Exception as e:
        print(f"  ERROR in load_image_tensor: {e}")
        return None


def calculate_object_consistency(img_night_path, img_day_path):
    """Returns object consistency score (0.0 to 1.0)"""
    try:
        res_night = yolo_model(img_night_path, verbose=False)[0]
        res_day = yolo_model(img_day_path, verbose=False)[0]

        boxes_night = res_night.boxes.data.cpu().numpy()
        boxes_day = res_day.boxes.data.cpu().numpy()

        # Handle edge cases
        if len(boxes_night) == 0 and len(boxes_day) == 0:
            return 1.0
        if len(boxes_night) == 0:
            return 1.0
        if len(boxes_day) == 0:
            return 0.0

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

        # Bonus for more objects in day
        if len(boxes_day) > len(boxes_night):
            improvement_bonus = min(0.2, (len(boxes_day) - len(boxes_night)) / len(boxes_night) * 0.1)
            return min(1.0, base_score + improvement_bonus)

        return base_score
    except Exception as e:
        print(f"    ERROR in object_consistency: {e}")
        return 0.0


def calculate_object_visibility_improvement(img_night_path, img_day_path):
    """How many MORE objects are visible in day vs night?"""
    try:
        res_night = yolo_model(img_night_path, verbose=False)[0]
        res_day = yolo_model(img_day_path, verbose=False)[0]

        boxes_night = res_night.boxes.data.cpu().numpy()
        boxes_day = res_day.boxes.data.cpu().numpy()

        if len(boxes_night) == 0:
            return len(boxes_day) / 1.0
        return len(boxes_day) / len(boxes_night)
    except Exception as e:
        print(f"    ERROR in object_visibility: {e}")
        return 1.0


def calculate_ssim_score(img_night, img_day):
    """Calculate structural similarity"""
    try:
        gray_night = cv2.cvtColor(img_night, cv2.COLOR_BGR2GRAY)
        gray_day = cv2.cvtColor(img_day, cv2.COLOR_BGR2GRAY)

        if gray_night.shape != gray_day.shape:
            gray_day = cv2.resize(gray_day, (gray_night.shape[1], gray_night.shape[0]))

        return ssim(gray_night, gray_day)
    except Exception as e:
        print(f"    ERROR in SSIM: {e}")
        return 0.0


def calculate_day_characteristics(img_night, img_day):
    """Analyze if image has proper DAY characteristics"""
    try:
        lab_night = cv2.cvtColor(img_night, cv2.COLOR_BGR2LAB)
        lab_day = cv2.cvtColor(img_day, cv2.COLOR_BGR2LAB)

        hsv_night = cv2.cvtColor(img_night, cv2.COLOR_BGR2HSV)
        hsv_day = cv2.cvtColor(img_day, cv2.COLOR_BGR2HSV)

        metrics = {
            'brightness_increase': np.mean(lab_day[:, :, 0]) - np.mean(lab_night[:, :, 0]),
            'warm_shift': np.mean(lab_night[:, :, 2]) - np.mean(lab_day[:, :, 2]),
            'saturation_increase': np.mean(hsv_day[:, :, 1]) - np.mean(hsv_night[:, :, 1])
        }
        return metrics
    except Exception as e:
        print(f"    ERROR in day_characteristics: {e}")
        return {'brightness_increase': 0, 'warm_shift': 0, 'saturation_increase': 0}


def calculate_edge_preservation(img_night, img_day):
    """Edge preservation"""
    try:
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
    except Exception as e:
        print(f"    ERROR in edge_preservation: {e}")
        return 0.0


def calculate_shadow_removal(img_night, img_day):
    """Assess shadow removal quality"""
    try:
        gray_night = cv2.cvtColor(img_night, cv2.COLOR_BGR2GRAY)
        gray_day = cv2.cvtColor(img_day, cv2.COLOR_BGR2GRAY)

        if gray_night.shape != gray_day.shape:
            gray_day = cv2.resize(gray_day, (gray_night.shape[1], gray_night.shape[0]))

        def local_std(img, ksize=15):
            mean = cv2.blur(img.astype(float), (ksize, ksize))
            sqr_mean = cv2.blur((img.astype(float) ** 2), (ksize, ksize))
            return np.sqrt(np.maximum(sqr_mean - mean ** 2, 0))

        std_night = local_std(gray_night)
        std_day = local_std(gray_day)

        avg_std_night = np.mean(std_night)
        avg_std_day = np.mean(std_day)

        if avg_std_night == 0:
            return 1.0

        reduction = (avg_std_night - avg_std_day) / avg_std_night
        return max(0.0, min(1.0, reduction))
    except Exception as e:
        print(f"    ERROR in shadow_removal: {e}")
        return 0.0


def calculate_overexposure_score(img_day):
    """Check for overexposure"""
    try:
        lab = cv2.cvtColor(img_day, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        overexposed_pixels = np.sum(l_channel > 240)
        total_pixels = l_channel.size
        overexposure_ratio = overexposed_pixels / total_pixels

        hsv = cv2.cvtColor(img_day, cv2.COLOR_BGR2HSV)
        bright_areas = l_channel > 200
        if np.sum(bright_areas) > 0:
            avg_saturation_bright = np.mean(hsv[:, :, 1][bright_areas])
            saturation_penalty = max(0, (50 - avg_saturation_bright) / 50)
        else:
            saturation_penalty = 0

        overexposure_score = 1.0 - (overexposure_ratio * 2 + saturation_penalty * 0.5)
        return max(0.0, min(1.0, overexposure_score))
    except Exception as e:
        print(f"    ERROR in overexposure: {e}")
        return 0.0


def interpret_results(row):
    """Automatically flag potential issues"""
    issues = []
    if row['object_consistency'] < 0.5:
        issues.append("Lost_objects")
    if row['brightness_increase'] < 20:
        issues.append("Insufficient_brightening")
    elif row['brightness_increase'] > 120:
        issues.append("Over_brightened")
    if row['ssim'] < 0.6:
        issues.append("Structure_degraded")
    if row['lpips'] < 0.3:
        issues.append("Minimal_transformation")
    elif row['lpips'] > 0.8:
        issues.append("Over_transformation")
    if row['edge_preservation'] < 0.3:
        issues.append("Details_lost")
    if row['warm_shift'] < 0:
        issues.append("Wrong_color_temp")
    if row['saturation_increase'] < 0:
        issues.append("Undersaturated")
    if row['overexposure_score'] < 0.6:
        issues.append("Overexposed")
    if row['shadow_removal'] < 0.2:
        issues.append("Shadows_remain")
    return '; '.join(issues) if issues else 'OK'


def get_all_image_files(directory):
    """Get all image files from directory"""
    return sorted([f for f in os.listdir(directory)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])


def create_filename_mapping():
    """
    Create a mapping between night and day filenames.
    Since they have different names, we'll match them by:
    1. Same numeric patterns
    2. Same city names
    3. Same sequence
    """
    night_files = get_all_image_files(NIGHT_PATH)
    day_files = get_all_image_files(DAY_PATH)

    print(f"Found {len(night_files)} night files and {len(day_files)} day files")

    # If counts match, assume they're in the same order
    if len(night_files) == len(day_files):
        print("✓ File counts match - using sequential pairing")
        return list(zip(night_files, day_files))

    # Otherwise, try to match by numeric patterns
    mapping = []
    used_day_files = set()

    for night_file in night_files:
        best_match = None
        best_score = 0

        # Extract numeric parts from night filename
        night_parts = night_file.split('_')
        night_nums = [p for p in night_parts if p.isdigit()]

        for day_file in day_files:
            if day_file in used_day_files:
                continue

            # Extract numeric parts from day filename
            day_parts = day_file.split('_')
            day_nums = [p for p in day_parts if p.isdigit()]

            # Calculate matching score
            score = 0
            if night_nums and day_nums:
                # Check if any numbers match
                common_nums = set(night_nums) & set(day_nums)
                score = len(common_nums)

            # Also check city name
            night_city = night_parts[0] if night_parts else ""
            day_city = day_parts[0] if day_parts else ""
            if night_city and day_city and night_city in day_city or day_city in night_city:
                score += 1

            if score > best_score:
                best_score = score
                best_match = day_file

        if best_match and best_score > 0:
            mapping.append((night_file, best_match))
            used_day_files.add(best_match)
            print(f"  Matched: {night_file} → {best_match} (score: {best_score})")
        else:
            print(f"  ❌ No match found for: {night_file}")

    return mapping


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    print("=" * 60)
    print("NIGHT→DAY EVALUATION")
    print("=" * 60)

    # Create filename mapping
    print("\nCreating filename mapping...")
    file_mapping = create_filename_mapping()

    if not file_mapping:
        print("❌ No files could be matched! Check your folders.")
        return

    print(f"✓ Successfully mapped {len(file_mapping)} file pairs")

    results = []
    errors = []

    processed_count = 0
    for i, (night_filename, day_filename) in enumerate(file_mapping, 1):
        print(f"\n[{i}/{len(file_mapping)}] Processing pair:")
        print(f"  Night: {night_filename}")
        print(f"  Day:   {day_filename}")

        try:
            night_full_path = os.path.join(NIGHT_PATH, night_filename)
            day_full_path = os.path.join(DAY_PATH, day_filename)

            # Load images
            cv_night = cv2.imread(night_full_path)
            cv_day = cv2.imread(day_full_path)

            if cv_night is None:
                print(f"  ❌ Failed to load night image")
                errors.append(f"{night_filename}: Failed to load night image")
                continue

            if cv_day is None:
                print(f"  ❌ Failed to load day image")
                errors.append(f"{day_filename}: Failed to load day image")
                continue

            print(f"  ✓ Images loaded - Night: {cv_night.shape}, Day: {cv_day.shape}")

            # 1. LPIPS (transformation magnitude)
            t_night = load_image_tensor(night_full_path)
            t_day = load_image_tensor(day_full_path)

            if t_night is None or t_day is None:
                errors.append(f"{night_filename}: Failed to load tensors")
                continue

            if torch.cuda.is_available():
                t_night, t_day = t_night.cuda(), t_day.cuda()
            lpips_val = lpips_fn(t_night, t_day).item()

            # 2. Object Consistency
            obj_score = calculate_object_consistency(night_full_path, day_full_path)

            # 3. Object Visibility Improvement
            obj_improvement = calculate_object_visibility_improvement(night_full_path, day_full_path)

            # 4. SSIM (structure preservation)
            ssim_val = calculate_ssim_score(cv_night, cv_day)

            # 5. Day characteristics
            day_chars = calculate_day_characteristics(cv_night, cv_day)

            # 6. Edge preservation
            edge_pres = calculate_edge_preservation(cv_night, cv_day)

            # 7. Shadow removal
            shadow_removal = calculate_shadow_removal(cv_night, cv_day)

            # 8. Overexposure check
            overexposure = calculate_overexposure_score(cv_day)

            # Create result
            result = {
                'night_image_name': night_filename,
                'day_image_name': day_filename,
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
            processed_count += 1

            print(f"  ✅ SUCCESS - Obj: {obj_score:.2f} (+{obj_improvement:.2f}x) | "
                  f"SSIM: {ssim_val:.2f} | Bright↑: {day_chars['brightness_increase']:.1f}")

        except Exception as e:
            errors.append(f"{night_filename}: {str(e)}")
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n" + "=" * 60)
    print(f"PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Successfully processed: {processed_count}/{len(file_mapping)} image pairs")
    print(f"Errors: {len(errors)}")

    if not results:
        print("❌ No results to save!")
        return

    # Create DataFrame and save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Detailed results saved to {OUTPUT_CSV}")

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
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for _, row in summary.iterrows():
        print(f"{row['metric']:30s}: {row['mean']:6.3f} ± {row['std']:5.3f}")

    # Show errors if any
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for err in errors[:10]:
            print(f"  {err}")


if __name__ == '__main__':
    main()