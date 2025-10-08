import os
import cv2
import csv
import math
import mediapipe as mp

# Input / Output folders
video_folder = "videos"
output_folder = "crops"
os.makedirs(output_folder, exist_ok=True)

# CSV for crop metadata (matches your filename in the posted script)
crop_csv = "cropped_annotations.csv"
with open(crop_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["video_name", "frame_index", "hand_index", "crop_filename", "label"])

# Mediapipe setup
mp_hands = mp.solutions.hands

# Helper: rotate image by angle_degs around center (keeps same size)
def rotate_image(img, angle_degs):
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_degs, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

# Process each video
for video_file in sorted(os.listdir(video_folder)):
    if not video_file.lower().endswith(".mp4"):
        continue

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

    # Use tracking confidence to improve multi-hand detection across frames
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            h_frame, w_frame, _ = frame.shape

            # Convert frame to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if not results.multi_hand_landmarks:
                # no hands found in this frame
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # results.multi_hand_landmarks and results.multi_handedness correspond index-wise
            handedness_list = results.multi_handedness if results.multi_handedness else [None] * len(results.multi_hand_landmarks)

            # iterate through each detected hand (this will capture both hands)
            for hand_index, (hand_landmarks, hand_handedness) in enumerate(zip(results.multi_hand_landmarks, handedness_list)):
                # compute landmark pixel coordinates
                x_coords = [lm.x * w_frame for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h_frame for lm in hand_landmarks.landmark]

                # bounding box from landmarks
                x_min = int(min(x_coords))
                x_max = int(max(x_coords))
                y_min = int(min(y_coords))
                y_max = int(max(y_coords))

                # Add generous margin so crop > 224x224 and to avoid cutting fingers
                margin = 40
                x_min = max(x_min - margin, 0)
                y_min = max(y_min - margin, 0)
                x_max = min(x_max + margin, w_frame)
                y_max = min(y_max + margin, h_frame)

                # Make the crop square (centered)
                box_w = x_max - x_min
                box_h = y_max - y_min
                box_size = max(box_w, box_h)
                # ensure some minimum minimum (e.g., 250) if you want always >224
                MIN_BOX = 250
                if box_size < MIN_BOX:
                    box_size = MIN_BOX

                # center of original bbox
                cx = x_min + box_w // 2
                cy = y_min + box_h // 2

                # compute new square coordinates around center
                x1 = int(max(cx - box_size // 2, 0))
                y1 = int(max(cy - box_size // 2, 0))
                x2 = int(min(x1 + box_size, w_frame))
                y2 = int(min(y1 + box_size, h_frame))

                # adjust if crop went out of bounds on the right/bottom
                if x2 - x1 < box_size:
                    x1 = max(x2 - box_size, 0)
                if y2 - y1 < box_size:
                    y1 = max(y2 - box_size, 0)

                # final crop
                cropped = frame[y1:y2, x1:x2]

                if cropped.size == 0:
                    continue  # skip invalid crops

                # --- orientation correction ---
                # landmarks: 0 = wrist, 9 = middle_finger_mcp (MediaPipe indexing)
                try:
                    wrist = hand_landmarks.landmark[0]
                    middle_mcp = hand_landmarks.landmark[9]
                    wrist_px = (wrist.x * w_frame - x1, wrist.y * h_frame - y1)
                    middle_px = (middle_mcp.x * w_frame - x1, middle_mcp.y * h_frame - y1)
                    vx = middle_px[0] - wrist_px[0]
                    vy = middle_px[1] - wrist_px[1]
                    # compute current angle (degrees)
                    theta_deg = math.degrees(math.atan2(vy, vx))
                    # compute rotation to make the finger vector point upward (-90 deg)
                    rot_deg = -90.0 - theta_deg
                    # rotate cropped image by rot_deg
                    cropped = rotate_image(cropped, rot_deg)
                except Exception:
                    # If anything goes wrong with orientation logic, keep the unrotated crop
                    pass

                # Save crop filename in format: VideoName_####.jpg
                saved_count += 1
                crop_filename = f"{video_name}_{saved_count:04d}.jpg"
                save_path = os.path.join(video_output_folder, crop_filename)

                # Save crop
                cv2.imwrite(save_path, cropped)

                # Save metadata with label (note: we write just filename so preprocess can find it under crops/<video_name>/)
                with open(crop_csv, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([video_name, frame_count, hand_index, crop_filename, label])

                # Optional preview (comment out to speed up)
                cv2.imshow("Hand Crop", cropped)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    print(f"✅ Finished {video_file}: {saved_count} crops saved to {video_output_folder}")

cv2.destroyAllWindows()
