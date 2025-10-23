import os, argparse, random, csv
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score
from device_helper import get_device


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet50", "resnet101"], default="resnet50")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--processed_dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device, device_type = get_device()

    # ====== TRANSFORMS ======
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ====== DATASETS ======
    train_dir = os.path.join(args.processed_dir, "train")
    val_dir = os.path.join(args.processed_dir, "val")

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=val_transform)
    num_classes = len(train_data.classes)

    pin_memory = device_type == "cuda"

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=pin_memory)  # ⚠ safer on Windows
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=pin_memory)

    print("[INFO] Device type:", device_type)
    print("[INFO] Classes:", train_data.classes)

    # ====== MODEL ======
    if args.model == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
    else:
        model = models.resnet101(weights="IMAGENET1K_V1")

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if args.freeze:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)

    best_val_acc = 0.0
    logfile = f"training_log_{args.model}.csv"
    with open(logfile, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    # ====== TRAINING LOOP ======
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses, y_true, y_pred = [], [], []
        for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.detach().cpu().numpy())

        train_loss = np.mean(train_losses)
        train_acc = accuracy_score(y_true, y_pred)

        # ====== VALIDATION ======
        model.eval()
        val_losses, y_true, y_pred = [], [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                preds = outputs.argmax(dim=1).detach().cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(labels.detach().cpu().numpy())

        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(y_true, y_pred)
        scheduler.step(val_acc)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

        with open(logfile, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, optimizer.param_groups[0]["lr"]])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = f"{args.model}_best.pth"
            torch.save({
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch,
                "classes": train_data.classes
            }, ckpt)
            print("[INFO] Saved best model:", ckpt)

    print("[INFO] Training finished. Best val acc:", best_val_acc)


if __name__ == "__main__":
    main()
