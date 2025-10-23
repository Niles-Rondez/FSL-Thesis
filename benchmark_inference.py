import time, argparse, os, numpy as np
import torch, psutil
from torchvision import transforms, models
from PIL import Image
from device_helper import get_device

parser = argparse.ArgumentParser()
parser.add_argument("--model_ckpt", required=True)
parser.add_argument("--image_dir", default="data/splits/test")
parser.add_argument("--n_runs", type=int, default=200)
args = parser.parse_args()

device, device_type = get_device()

ckpt = torch.load(args.model_ckpt, map_location="cpu")
classes = ckpt.get("classes")
if "resnet101" in args.model_ckpt:
    model = models.resnet101(pretrained=False)
else:
    model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["model_state"])
model = model.to(device)
model.eval()

image_paths = []
for root,_,files in os.walk(args.image_dir):
    for f in files:
        if f.lower().endswith(('.jpg','.jpeg','.png')):
            image_paths.append(os.path.join(root,f))
if not image_paths:
    raise SystemExit("No images found")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# warmup
for i in range(10):
    im = Image.open(image_paths[i % len(image_paths)]).convert("RGB")
    _ = model(transform(im).unsqueeze(0).to(device))

times = []
proc = psutil.Process()
for i in range(args.n_runs):
    im = Image.open(image_paths[i % len(image_paths)]).convert("RGB")
    x = transform(im).unsqueeze(0).to(device)
    t0 = time.time()
    with torch.no_grad():
        _ = model(x)
    # torch.cuda.synchronize() # not valid with DirectML; ignore
    t1 = time.time()
    times.append(t1 - t0)

times = np.array(times)
print("Mean latency (ms):", times.mean()*1000)
print("Median latency (ms):", np.median(times)*1000)
print("Std latency (ms):", times.std()*1000)
print("Throughput (images/sec):", 1.0 / times.mean())
mem = proc.memory_info()
print("Process RSS (MB):", mem.rss / (1024*1024))
if device_type == "cuda":
    print("CUDA mem allocated (MB):", torch.cuda.memory_allocated()/(1024*1024))
