import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms, models
from PIL import Image
from device_helper import get_device

UPLOAD_FOLDER = "uploads"
ALLOWED = {"png","jpg","jpeg"}
MODEL_MAP = {"resnet50":"resnet50_best.pth", "resnet101":"resnet101_best.pth"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device, device_type = get_device()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED

def load_model(choice):
    ckpt = torch.load(MODEL_MAP[choice], map_location="cpu")
    classes = ckpt["classes"]
    if choice == "resnet101":
        model = models.resnet101(pretrained=False)
    else:
        model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, classes

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        model_choice = request.form.get("model_choice","resnet50")
        file = request.files.get("file")
        if not file or file.filename == "":
            return "No file", 400
        filename = secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)
        model, classes = load_model(model_choice)
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            pred = out.argmax(dim=1).item()
        return render_template("result.html", label=classes[pred], model=model_choice)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
