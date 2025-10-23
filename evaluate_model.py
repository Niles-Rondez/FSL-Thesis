# evaluate_model.py
import os
import argparse
import csv
import torch
import numpy as np
from torchvision import transforms, datasets, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# device helper (paste or import your existing helper)
def get_device():
    try:
        import torch_directml
        dev = torch_directml.device()
        print("[DEVICE] Using DirectML device:", dev)
        return dev, "directml"
    except Exception:
        if torch.cuda.is_available():
            print("[DEVICE] Using CUDA device")
            return torch.device("cuda"), "cuda"
        else:
            print("[DEVICE] Using CPU")
            return torch.device("cpu"), "cpu"


def evaluate(ckpt_path, test_dir, out_csv, cm_png, batch_size, num_workers):
    device, device_type = get_device()

    # NOTE: map_location="cpu" to safely load checkpoint; we'll move model to device later
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt.get("classes")
    if classes is None:
        raise RuntimeError("Checkpoint does not contain 'classes' key.")

    # Build model skeleton
    if "resnet101" in ckpt_path.lower():
        model = models.resnet101(pretrained=False)
    else:
        model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_data = datasets.ImageFolder(test_dir, transform=transform)

    pin_memory = True if device_type == "cuda" else False
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            # Move to device. If device is DirectML device object, we still use .to(device)
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    # Metrics
    report = classification_report(y_true, y_pred, target_names=test_data.classes, output_dict=True)

    # Save CSV
    with open(out_csv, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for cls in test_data.classes:
            m = report[cls]
            writer.writerow([cls, m["precision"], m["recall"], m["f1-score"], m["support"]])
        writer.writerow(["accuracy", report["accuracy"], "", "", sum([report[c]["support"] for c in test_data.classes])])
    print("[INFO] Classification report saved to", out_csv)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", annot=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(cm_png)
    print("[INFO] Confusion matrix saved to", cm_png)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", required=True, help="path to saved checkpoint (pth)")
    parser.add_argument("--test_dir", default="data/processed/test")
    parser.add_argument("--out_csv", default="test_classification_report.csv")
    parser.add_argument("--cm_png", default="confusion_matrix.png")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of DataLoader workers. On Windows, set to 0 or use the __main__ guard (default 0).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # run evaluation
    evaluate(args.model_ckpt, args.test_dir, args.out_csv, args.cm_png, args.batch_size, args.num_workers)
