# augmentation.py
import os
import cv2
import csv
import random
import numpy as np

INPUT_ROOT = "processed"
OUTPUT_ROOT = "augmented"
ANNOTATIONS = "augmented_annotations.csv"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Create CSV header
with open(ANNOTATIONS, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["subject", "label", "original_file", "augmented_file"])

# --- Augmentation functions ---

def random_rotation(img):
    """Rotate image randomly between -15° and +15°"""
    angle = random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def random_brightness(img):
    """Randomly adjust brightness and contrast"""
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-30, 30)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def horizontal_flip(img):
    """Flip image horizontally"""
    return cv2.flip(img, 1)

def add_noise(img):
    """Add Gaussian noise to image"""
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# List of augmentation operations
AUGMENTATIONS = [
    random_rotation,
    random_brightness,
    horizontal_flip,
    add_noise
]

# Loop through dataset
for subject in os.listdir(INPUT_ROOT):
    for label in os.listdir(os.path.join(INPUT_ROOT, subject)):
        input_dir = os.path.join(INPUT_ROOT, subject, label)
        output_dir = os.path.join(OUTPUT_ROOT, subject, label)
        os.makedirs(output_dir, exist_ok=True)

        for img_file in os.listdir(input_dir):
            img = cv2.imread(os.path.join(input_dir, img_file))
            if img is None:
                continue

            # Generate 10 augmented versions per image
            for i in range(10):
                aug_img = random.choice(AUGMENTATIONS)(img)
                new_name = f"{img_file.split('.')[0]}_aug{i}.jpg"
                cv2.imwrite(os.path.join(output_dir, new_name), aug_img)

                # Save augmentation metadata
                with open(ANNOTATIONS, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([subject, label, img_file, new_name])

        print(f"✅ Augmented {subject} {label}")

print("🎉 Data augmentation completed successfully")
