# preprocess_pipeline.py
import os
import cv2
import pandas as pd
import csv
import glob

# Files / folders
annotations_file = "cropped_annotations.csv"        # created by extract_crop_hands.py
crops_folder = "crops"                           # root folder where crops are stored
output_folder = "processed"                      # where processed images will be saved
preprocessed_csv = "preprocessed_annotations.csv"

os.makedirs(output_folder, exist_ok=True)

# Check the annotations CSV exists
if not os.path.exists(annotations_file):
    print(f"ERROR: '{annotations_file}' not found in current folder ({os.getcwd()}).")
    print("Make sure you run extract_crop_hands.py first and that cropped_annotations.csv is in this folder.")
    raise SystemExit(1)

# Load annotations
annotations = pd.read_csv(annotations_file)

# Prepare (overwrite) the preprocessed CSV with header
with open(preprocessed_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["video_name", "frame_index", "hand_index", "preprocessed_filename", "label"])

processed_count = 0
skipped_count = 0

# --- Group annotations by video_name ---
groups = annotations.groupby("video_name")

for video_name, group in groups:
    saved_count = 0

    # Output folder for this video
    video_out = os.path.join(output_folder, video_name if video_name else "unknown")
    os.makedirs(video_out, exist_ok=True)

    for idx, row in group.iterrows():
        frame_index = int(row.get("frame_index", 0))
        hand_index = int(row.get("hand_index", 0))
        label = str(row.get("label", "")).strip()
        crop_filename = str(row.get("crop_filename", "")).strip()

        # Try to find the crop path
        possible_paths = []
        if os.path.isabs(crop_filename) and os.path.exists(crop_filename):
            crop_path = crop_filename
        else:
            if video_name:
                possible_paths.append(os.path.join(crops_folder, video_name, crop_filename))
            possible_paths.append(os.path.join(crops_folder, crop_filename))
            possible_paths.append(crop_filename)
            matches = glob.glob(os.path.join(crops_folder, "**", os.path.basename(crop_filename)), recursive=True)
            possible_paths.extend(matches)

            crop_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    crop_path = p
                    break

        if not crop_path or not os.path.exists(crop_path):
            print(f"WARNING: crop file not found for row {idx}: tried {len(possible_paths)} locations.")
            skipped_count += 1
            continue

        img = cv2.imread(crop_path)
        if img is None:
            print(f"WARNING: OpenCV failed to read image at '{crop_path}' (row {idx}). Skipping.")
            skipped_count += 1
            continue

        # --- Preprocessing pipeline ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        roi = blurred
        resized = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
        normalized = (resized.astype("float32") / 255.0)

        # Save processed image
        out_filename = os.path.join(video_out, os.path.basename(crop_path))
        cv2.imwrite(out_filename, (normalized * 255).astype("uint8"))

        # Append metadata
        with open(preprocessed_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([video_name, frame_index, hand_index, out_filename, label])

        processed_count += 1
        saved_count += 1

    # ✅ Print summary right after finishing this folder
    print(f"✅ Finished {video_name if video_name else 'unknown'}: {saved_count} images saved to {video_out}")

# ✅ Final summary
print("\n🎉 Preprocessing finished.")
print(f"Processed: {processed_count} images")
print(f"Skipped (missing / unreadable): {skipped_count}")
print(f"Preprocessed images saved under: '{output_folder}/'")
print(f"Preprocessed metadata written to: '{preprocessed_csv}'")
