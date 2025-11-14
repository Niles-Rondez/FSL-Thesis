import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import base64
from io import BytesIO
from device_helper import get_device

UPLOAD_FOLDER = "uploads"
ALLOWED = {"png", "jpg", "jpeg"}
MODEL_MAP = {"resnet50": "resnet50_best.pth", "resnet101": "resnet101_best.pth"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device, device_type = get_device()

# For Flask with DirectML, always use CPU for inference
inference_device = torch.device("cpu") if device_type == "directml" else device
print(f"[INFO] Flask inference will use: {inference_device}")

# Cache for loaded models
model_cache = {}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

def load_model(choice):
    """Load model with caching"""
    if choice in model_cache:
        return model_cache[choice]
    
    ckpt = torch.load(MODEL_MAP[choice], map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    
    if choice == "resnet101":
        model = models.resnet101(weights=None)
    else:
        model = models.resnet50(weights=None)
    
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    
    # Keep model on CPU for DirectML compatibility with Flask
    model.to(inference_device)
    model.eval()
    
    print(f"[INFO] Model {choice} loaded on {inference_device}")
    
    model_cache[choice] = (model, classes)
    return model, classes

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle file upload prediction"""
    try:
        model_choice = request.form.get("model_choice", "resnet50")
        file = request.files.get("file")
        
        if not file or file.filename == "":
            return jsonify({"error": "No file provided"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Use PNG, JPG, or JPEG"}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        
        # Load model and predict
        model, classes = load_model(model_choice)
        img = Image.open(filepath).convert("RGB")
        x = transform(img).unsqueeze(0).to(inference_device)
        
        with torch.no_grad():
            out = model(x)
            probs = torch.nn.functional.softmax(out, dim=1)[0]
            pred = out.argmax(dim=1).item()
            confidence = probs[pred].item()
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probs, min(3, len(classes)))
        top_predictions = [
            {"class": classes[idx], "confidence": prob.item()}
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        return jsonify({
            "prediction": classes[pred],
            "confidence": confidence,
            "top_predictions": top_predictions,
            "model": model_choice
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_webcam", methods=["POST"])
def predict_webcam():
    """Handle webcam frame prediction with hand detection"""
    try:
        model_choice = request.form.get("model_choice", "resnet50")
        image_data = request.form.get("image")
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode base64 image
        image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400
        
        # Import mediapipe here to detect hands
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.6
        )
        
        # Convert to RGB for MediaPipe
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        bbox = None
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get bounding box
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
            
            bbox = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
            
            # Crop hand region
            hand_img = frame[ymin:ymax, xmin:xmax]
            
            if hand_img.size == 0:
                hands.close()
                return jsonify({"error": "Hand region too small", "bbox": bbox}), 400
            
            # Convert to PIL and predict
            hand_img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(hand_img_rgb)
            
            model, classes = load_model(model_choice)
            x = transform(pil_img).unsqueeze(0).to(inference_device)
            
            with torch.no_grad():
                out = model(x)
                probs = torch.nn.functional.softmax(out, dim=1)[0]
                pred = out.argmax(dim=1).item()
                confidence = probs[pred].item()

                top_probs, top_indices = torch.topk(probs, min(3, len(classes)))
                top_predictions = [
                    {"class": classes[idx], "confidence": prob.item()}
                    for prob, idx in zip(top_probs, top_indices)
                ]
                            
            hands.close()
            
            return jsonify({
                "prediction": classes[pred],
                "confidence": confidence,
                "top_predictions": top_predictions,
                "model": model_choice,
                "bbox": bbox,
                "hand_detected": True
            })
        else:
            hands.close()
            return jsonify({
                "error": "No hand detected",
                "hand_detected": False,
                "bbox": None
            }), 200
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/models", methods=["GET"])
def get_models():
    """Return available models"""
    available_models = []
    for model_name, path in MODEL_MAP.items():
        if os.path.exists(path):
            available_models.append(model_name)
    return jsonify({"models": available_models})

if __name__ == "__main__":
    print(f"Device type: {device_type}")
    print(f"Inference device: {inference_device}")
    print(f"Available models: {list(MODEL_MAP.keys())}")
    app.run(debug=True, host="0.0.0.0", port=5000)