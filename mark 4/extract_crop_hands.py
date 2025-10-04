import cv2
import os
import csv
import mediapipe as mp

# Input / Output folders
video_folder = "videos"
output_folder = "crops"
os.makedirs(output_folder, exist_ok=True)

# CSV for crop metadata
crop_csv = "cropped_annotations.csv"
with open(crop_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["video_name", "frame_index", "hand_index", "crop_filename", "label"])

# Mediapipe setup
mp_hands = mp.solutions.hands

# Process each video
for video_file in os.listdir(video_folder):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]

        # Derive label from video name (everything after first underscore if present)
        label = video_name.split("_", 1)[-1] if "_" in video_name else video_name

        # Create subfolder for this video’s crops
        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        ) as hands:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                h, w, _ = frame.shape

                # Convert frame to RGB for MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        # Get bounding box
                        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                        x_min, x_max = int(min(x_coords)), int(max(x_coords))
                        y_min, y_max = int(min(y_coords)), int(max(y_coords))

                        # Add margin (larger than 224x224 to allow resizing later)
                        margin = 40
                        x_min = max(x_min - margin, 0)
                        y_min = max(y_min - margin, 0)
                        x_max = min(x_max + margin, w)
                        y_max = min(y_max + margin, h)

                        # Crop hand
                        cropped = frame[y_min:y_max, x_min:x_max]

                        if cropped.size == 0:
                            continue  # skip invalid crops

                        # New filename format: VideoName_####.jpg
                        crop_filename = f"{video_name}_{saved_count+1:04d}.jpg"
                        save_path = os.path.join(video_output_folder, crop_filename)

                        # Save crop
                        cv2.imwrite(save_path, cropped)
                        saved_count += 1

                        # Save metadata with label
                        with open(crop_csv, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([video_name, frame_count, hand_index, crop_filename, label])

                        # Show preview
                        cv2.imshow("Hand Crop", cropped)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        print(f"✅ Finished {video_file}: {saved_count} crops saved to {video_output_folder}")

cv2.destroyAllWindows()
