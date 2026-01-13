# preprocess_pipeline.py
import os
import cv2
import csv

CROP_ROOT = "crops"
OUT_ROOT = "processed"
ANNOTATIONS = "preprocessed_annotations.csv"

os.makedirs(OUT_ROOT, exist_ok=True)

# Create CSV header
with open(ANNOTATIONS, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["subject", "label", "filename"])

# Loop through subjects and gesture labels
for subject in os.listdir(CROP_ROOT):
    for label in os.listdir(os.path.join(CROP_ROOT, subject)):
        input_dir = os.path.join(CROP_ROOT, subject, label)
        output_dir = os.path.join(OUT_ROOT, subject, label)
        os.makedirs(output_dir, exist_ok=True)

        for img_file in os.listdir(input_dir):
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Resize image to match ResNet input
            img = cv2.resize(img, (224, 224))

            # Apply Gaussian blur for noise reduction
            img = cv2.GaussianBlur(img, (5, 5), 0)

            # Normalize pixel values to [0, 1] then convert back to uint8
            img = (img.astype("float32") / 255.0 * 255).astype("uint8")

            # Save processed image
            cv2.imwrite(os.path.join(output_dir, img_file), img)

            # Save metadata
            with open(ANNOTATIONS, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([subject, label, img_file])

        print(f"✅ Preprocessed {subject} {label}")

print("🎉 Preprocessing completed successfully")
