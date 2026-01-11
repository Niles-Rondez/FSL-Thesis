import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import mediapipe as mp
import numpy as np
from collections import deque
import time

# ==== CONFIG ====
MODEL_PATH = "models/final/resnet50_final.pth"  # Change to your model file
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREDICTIONS_PER_SECOND = 3  # How many predictions per second
SMOOTHING_WINDOW = 5  # Number of predictions to smooth over

# ==== LOAD MODEL ====
print(f"Loading model from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
classes = checkpoint.get("classes")
if classes is None:
    raise RuntimeError("Checkpoint does not contain 'classes' key.")

# Determine model architecture from checkpoint or path
if "resnet101" in MODEL_PATH.lower():
    model = models.resnet101(weights=None)
    print("Using ResNet101 architecture")
else:
    model = models.resnet50(weights=None)
    print("Using ResNet50 architecture")

model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["model_state"])
model.eval().to(DEVICE)
print(f"✅ Loaded model with {len(classes)} classes on {DEVICE}")
print(f"Classes: {classes}")

# ==== TRANSFORMS ====
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== MEDIAPIPE HAND DETECTION ====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ==== PREDICTION SMOOTHING ====
prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

def get_smoothed_prediction():
    """Return most common prediction from buffer"""
    if not prediction_buffer:
        return None
    return max(set(prediction_buffer), key=prediction_buffer.count)

def predict_sign(hand_img):
    """Run inference on hand image"""
    try:
        input_tensor = transform(hand_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][pred].item()
            return classes[pred.item()], confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0

# ==== LIVE CAMERA LOOP ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("🎥 Press 'q' to quit")
print(f"Running {PREDICTIONS_PER_SECOND} predictions per second with {SMOOTHING_WINDOW}-frame smoothing")

last_prediction_time = 0
prediction_interval = 1.0 / PREDICTIONS_PER_SECOND
current_prediction = ""
current_confidence = 0.0
smoothed_prediction = ""

frame_count = 0
fps_start_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    current_time = time.time()

    # Calculate FPS
    frame_count += 1
    if current_time - fps_start_time >= 1.0:
        fps = frame_count / (current_time - fps_start_time)
        frame_count = 0
        fps_start_time = current_time

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Display info panel
    cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Prediction: {smoothed_prediction}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {current_confidence:.2%}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box with margin
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
            ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)

            # Add margin (20% of bbox size)
            width = xmax - xmin
            height = ymax - ymin
            margin_x = int(width * 0.2)
            margin_y = int(height * 0.2)
            
            xmin = max(0, xmin - margin_x)
            ymin = max(0, ymin - margin_y)
            xmax = min(w, xmax + margin_x)
            ymax = min(h, ymax + margin_y)

            # Crop hand region
            hand_img = frame[ymin:ymax, xmin:xmax]
            
            if hand_img.size > 0:
                # Run prediction at specified rate
                if current_time - last_prediction_time >= prediction_interval:
                    pred, conf = predict_sign(hand_img)
                    if pred:
                        current_prediction = pred
                        current_confidence = conf
                        prediction_buffer.append(pred)
                        smoothed_prediction = get_smoothed_prediction()
                    last_prediction_time = current_time

                # Draw bounding box
                color = (0, 255, 0) if current_confidence > 0.8 else (0, 255, 255)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
                
                # Draw large prediction on hand
                text_size = cv2.getTextSize(smoothed_prediction, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
                text_x = xmin + (xmax - xmin - text_size[0]) // 2
                text_y = ymin - 20
                
                cv2.putText(frame, smoothed_prediction, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
    else:
        # No hand detected
        if current_time - last_prediction_time > 1.0:
            prediction_buffer.clear()
            smoothed_prediction = ""
            current_prediction = ""
            current_confidence = 0.0

    cv2.imshow("FSL Live Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
print("✅ Camera released and windows closed")