import os
import cv2
import numpy as np
import torch
import lpips
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms

# --- CONFIGURATION ---
DAY_PATH = './path/to/input_day'
NIGHT_PATH = './path/to/generated_night'
OUTPUT_CSV = 'evaluation_results.csv'

# Load Models
print("Loading models...")
# LPIPS (AlexNet is faster/standard)
lpips_fn = lpips.LPIPS(net='alex').cuda() if torch.cuda.is_available() else lpips.LPIPS(net='alex')

# YOLO for Object Consistency (using a pre-trained model)
yolo_model = YOLO('yolov8n.pt')  # 'n' is nano (fast), use 'yolov8x.pt' for higher accuracy


def load_image_tensor(path):
    """Load image for LPIPS (Tensor, -1 to 1 normalized)"""
    img = Image.open(path).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return tf(img).unsqueeze(0)


def calculate_brightness_change(img_day, img_night):
    """
    Returns the reduction in brightness. Positive value = Night is darker.
    """
    # Convert to HSV, extract V (Value/Brightness) channel
    hsv_day = cv2.cvtColor(img_day, cv2.COLOR_BGR2HSV)
    hsv_night = cv2.cvtColor(img_night, cv2.COLOR_BGR2HSV)

    avg_day = np.mean(hsv_day[:, :, 2])
    avg_night = np.mean(hsv_night[:, :, 2])

    return avg_day - avg_night


def calculate_object_consistency(img_day_path, img_night_path):
    """
    Returns a consistency score (0.0 to 1.0).
    Ratio of objects detected in Day that are still present in Night.
    """
    # Run inference
    res_day = yolo_model(img_day_path, verbose=False)[0]
    res_night = yolo_model(img_night_path, verbose=False)[0]

    # Extract boxes (cls, x1, y1, x2, y2)
    # We focus on common classes: Car (2), Person (0), Bus (5), Truck (7), Traffic Light (9)
    # You can filter classes if you want.

    boxes_day = res_day.boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
    boxes_night = res_night.boxes.data.cpu().numpy()

    if len(boxes_day) == 0:
        return 1.0  # No objects to lose, so consistency is perfect

    matched_count = 0

    for d_box in boxes_day:
        d_cls = d_box[5]
        d_coords = d_box[:4]

        best_iou = 0

        # Check against all night boxes
        for n_box in boxes_night:
            n_cls = n_box[5]
            n_coords = n_box[:4]

            # Must be same class to be a "consistent" object
            if d_cls == n_cls:
                # Calculate IoU
                xA = max(d_coords[0], n_coords[0])
                yA = max(d_coords[1], n_coords[1])
                xB = min(d_coords[2], n_coords[2])
                yB = min(d_coords[3], n_coords[3])

                interArea = max(0, xB - xA) * max(0, yB - yA)
                boxAArea = (d_coords[2] - d_coords[0]) * (d_coords[3] - d_coords[1])
                boxBArea = (n_coords[2] - n_coords[0]) * (n_coords[3] - n_coords[1])
                iou = interArea / float(boxAArea + boxBArea - interArea)

                if iou > best_iou:
                    best_iou = iou

        # Threshold: If IoU > 0.3, we consider the object "preserved"
        # Note: Threshold is low because translation might shift pixels slightly
        if best_iou > 0.3:
            matched_count += 1

    return matched_count / len(boxes_day)


def main():
    results = []
    print("Starting evaluation...")

    # Iterate over input folder
    for fname in os.listdir(DAY_PATH):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        day_full_path = os.path.join(DAY_PATH, fname)

        # Handle filename matching.
        # CycleGAN often outputs: input="A.jpg", output="A_fake_B.png"
        # ADJUST THIS LOGIC based on your actual filenames
        night_fname = fname  # Assuming identical names for this example
        night_full_path = os.path.join(NIGHT_PATH, night_fname)

        if not os.path.exists(night_full_path):
            print(f"Skipping {fname}, corresponding output not found.")
            continue

        # 1. LPIPS
        t_day = load_image_tensor(day_full_path)
        t_night = load_image_tensor(night_full_path)
        if torch.cuda.is_available():
            t_day, t_night = t_day.cuda(), t_night.cuda()

        lpips_val = lpips_fn(t_day, t_night).item()

        # Load cv2 images for other metrics
        cv_day = cv2.imread(day_full_path)
        cv_night = cv2.imread(night_full_path)

        # 2. Brightness Reduction
        bright_diff = calculate_brightness_change(cv_day, cv_night)

        # 3. Object Consistency
        obj_score = calculate_object_consistency(day_full_path, night_full_path)

        print(f"{fname} | Obj: {obj_score:.2f} | BrightDiff: {bright_diff:.2f} | LPIPS: {lpips_val:.3f}")

        results.append({
            'name': fname,
            'object_consistency': obj_score,
            'brightness_reduction': bright_diff,
            'lpips': lpips_val
        })

    # Calculate Averages
    avg_obj = np.mean([r['object_consistency'] for r in results])
    avg_bright = np.mean([r['brightness_reduction'] for r in results])
    avg_lpips = np.mean([r['lpips'] for r in results])

    print("\n--- FINAL AVERAGES ---")
    print(f"Object Consistency: {avg_obj:.4f} (Higher is better)")
    print(f"Brightness Reduction: {avg_bright:.4f} (Positive is better)")
    print(f"LPIPS: {avg_lpips:.4f} (Lower is usually 'closer', but expect high val for Day->Night)")


if __name__ == '__main__':
    main()