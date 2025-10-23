import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import mediapipe as mp
import numpy as np

# ==== CONFIG ====
MODEL_PATH = "resnet50_best.pth"  # change to your model file
USE_MODEL = "resnet50"            # or "resnet50"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes 0–9, A–Y excluding J
CLASS_NAMES = [str(i) for i in range(10)] + [chr(c) for c in range(65, 91) if chr(c) != 'J' and chr(c) != 'Z']
print(f"Loaded {len(CLASS_NAMES)} classes: {CLASS_NAMES}")

# ==== MODEL LOAD ====
if USE_MODEL == "resnet50":
    model = models.resnet50(weights=None)
elif USE_MODEL == "resnet101":
    model = models.resnet101(weights=None)
else:
    raise ValueError("Invalid model type! Use 'resnet50' or 'resnet101'.")

model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
model.load_state_dict(state_dict)

model.eval().to(DEVICE)
print(f"✅ Loaded {USE_MODEL} model on {DEVICE}")

# ==== TRANSFORMS ====
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== MEDIAPIPE HAND DETECTION ====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# ==== LIVE CAMERA LOOP ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("🎥 Press 'q' to quit")
pred_text = ""
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box of hand
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
            ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)

            # Add small margin
            margin = 40
            xmin = max(0, xmin - margin)
            ymin = max(0, ymin - margin)
            xmax = min(w, xmax + margin)
            ymax = min(h, ymax + margin)

            # Crop the hand region
            hand_img = frame[ymin:ymax, xmin:xmax]
            if hand_img.size == 0:
                continue

            # Preprocess
            input_tensor = transform(hand_img).unsqueeze(0).to(DEVICE)

            # Predict
            with torch.no_grad():
                outputs = model(input_tensor)
                _, preds = torch.max(outputs, 1)
                pred_text = CLASS_NAMES[preds.item()]

            # Draw box and prediction
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{pred_text}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    else:
        pred_text = ""

    cv2.imshow("FSL Live Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
