# augmentation.py
import os
import cv2
import csv
import random
import numpy as np
from glob import glob

# Input / Output
input_folder = "preaugmentation"
output_folder = "augmented"
annotations_file = "augmented_annotations.csv"

os.makedirs(output_folder, exist_ok=True)

# Create CSV header
with open(annotations_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["class_label", "original_filename", "augmented_filename"])

# --- Augmentation functions ---

def random_rotation(image):
    angle = random.uniform(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def horizontal_flip(image):
    return cv2.flip(image, 1)

def random_brightness_contrast(image):
    alpha = random.uniform(0.7, 1.3)  # contrast
    beta = random.uniform(-40, 40)    # brightness
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def random_zoom_translation(image):
    h, w = image.shape[:2]
    zoom_factor = random.uniform(1.0, 1.2)  # up to 20% zoom
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)

    resized = cv2.resize(image, (new_w, new_h))
    x_start = random.randint(0, max(new_w - w, 0))
    y_start = random.randint(0, max(new_h - h, 0))
    cropped = resized[y_start:y_start+h, x_start:x_start+w]

    if cropped.shape[0] < h or cropped.shape[1] < w:
        cropped = cv2.resize(cropped, (w, h))

    return cropped

def add_noise(image):
    noise = np.random.normal(0, 20, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy

def augment_image(img):
    transforms = [
        random_rotation,
        horizontal_flip,
        random_brightness_contrast,
        random_zoom_translation,
        add_noise
    ]
    # Randomly pick one transform per augmentation
    return random.choice(transforms)(img)

# --- Process each class folder ---
for class_folder in os.listdir(input_folder):
    class_path = os.path.join(input_folder, class_folder)
    if not os.path.isdir(class_path):
        continue

    # Output subfolder
    output_class_folder = os.path.join(output_folder, class_folder)
    os.makedirs(output_class_folder, exist_ok=True)

    # Process all images
    images = glob(os.path.join(class_path, "*.jpg")) + glob(os.path.join(class_path, "*.png"))

    for img_path in images:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_path}")
            continue

        # Generate 50 augmented images
        for i in range(50):
            aug_img = augment_image(img)
            aug_filename = f"{os.path.splitext(img_name)[0]}_aug{i+1:03d}.jpg"
            aug_path = os.path.join(output_class_folder, aug_filename)

            cv2.imwrite(aug_path, aug_img)

            # Save annotation
            with open(annotations_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([class_folder, img_name, aug_filename])

    print(f"✅ Finished class {class_folder}: {len(images)*50} augmented images saved to {output_class_folder}")

print("\n🎉 Augmentation completed!")
print(f"➡ Augmented dataset saved to '{output_folder}/'")
print(f"➡ Metadata saved to '{annotations_file}'")
