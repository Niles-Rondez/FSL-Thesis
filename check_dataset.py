import os, csv, hashlib
from PIL import Image

DATA_DIR = "data/augmented"
REPORT_CSV = "dataset_integrity_report.csv"

def sha1_of_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(8192)
            if not b: break
            h.update(b)
    return h.hexdigest()

def main():
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    report_rows = []
    seen_hashes = {}
    total_images = 0
    for cls in classes:
        cls_dir = os.path.join(DATA_DIR, cls)
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        total_images += len(imgs)
        unreadable = []
        duplicates = []
        for fn in imgs:
            path = os.path.join(cls_dir, fn)
            try:
                Image.open(path).verify()
            except Exception:
                unreadable.append(fn)
                continue
            h = sha1_of_file(path)
            if h in seen_hashes:
                duplicates.append(f"{fn}=={seen_hashes[h]}")
            else:
                seen_hashes[h] = os.path.join(cls, fn)
        report_rows.append({
            "class": cls,
            "count": len(imgs),
            "unreadable": ";".join(unreadable),
            "duplicates": ";".join(duplicates)
        })

    with open(REPORT_CSV, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=["class","count","unreadable","duplicates"])
        writer.writeheader()
        for r in report_rows:
            writer.writerow(r)
    print(f"Checked {len(classes)} classes, {total_images} images total. Report: {REPORT_CSV}")

if __name__ == "__main__":
    main()
