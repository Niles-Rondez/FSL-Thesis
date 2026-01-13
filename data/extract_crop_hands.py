# extract_crop_hands.py
import os
import cv2
import math
import csv
import mediapipe as mp

RAW_ROOT = "raw"
CROP_ROOT = "crops"
ANNOTATIONS = "cropped_annotations.csv"

os.makedirs(CROP_ROOT, exist_ok=True)

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands

# Create CSV header
with open(ANNOTATIONS, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["subject", "label", "frame_index", "filename"])


def rotate_image(image, angle):
    """
    Rotates the image by a given angle.
    This helps standardize hand orientation (important for CNN learning).
    """
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


# Use MediaPipe in video tracking mode
with mp_hands.Hands(
    static_image_mode=False,      # Video mode (better tracking)
    max_num_hands=1,              # Only one hand per frame
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    # Loop through subjects (H01–H10)
    for subject in sorted(os.listdir(RAW_ROOT)):
        subject_path = os.path.join(RAW_ROOT, subject)
        if not os.path.isdir(subject_path):
            continue

        # Loop through each video of the subject
        for video_file in os.listdir(subject_path):
            if not video_file.lower().endswith((".mp4", ".mov")):
                continue

            # Extract label from filename (e.g., H01_0-9.mp4 → 0-9)
            label = video_file.split("_", 1)[1].split(".")[0]

            # Create output directory: data/crops/H01/0-9/
            output_dir = os.path.join(CROP_ROOT, subject, label)
            os.makedirs(output_dir, exist_ok=True)

            cap = cv2.VideoCapture(os.path.join(subject_path, video_file))
            frame_index = 0
            saved_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_index += 1

                # Convert BGR (OpenCV) to RGB (MediaPipe requirement)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                # Skip frames with no detected hands
                if not results.multi_hand_landmarks:
                    continue

                h, w, _ = frame.shape
                landmarks = results.multi_hand_landmarks[0]

                # Get bounding box from landmarks
                x_coords = [lm.x * w for lm in landmarks.landmark]
                y_coords = [lm.y * h for lm in landmarks.landmark]

                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))

                # Add margin so fingers are not clipped
                margin = 40
                x1 = max(x1 - margin, 0)
                y1 = max(y1 - margin, 0)
                x2 = min(x2 + margin, w)
                y2 = min(y2 + margin, h)

                # Crop the hand region (ROI)
                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue

                # Rotate hand to a consistent upright orientation
                wrist = landmarks.landmark[0]
                middle = landmarks.landmark[9]
                dx = middle.x - wrist.x
                dy = middle.y - wrist.y
                angle = math.degrees(math.atan2(dy, dx))
                cropped = rotate_image(cropped, -90 - angle)

                # Save cropped image
                filename = f"{subject}_{label}_{saved_count:05d}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), cropped)

                # Save metadata
                with open(ANNOTATIONS, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([subject, label, frame_index, filename])

                saved_count += 1

            cap.release()
            print(f"✅ {subject} {label}: {saved_count} hand crops saved")

print("🎉 Hand extraction completed successfully")
