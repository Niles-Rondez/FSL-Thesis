import os, random, csv
from PIL import Image
import matplotlib.pyplot as plt

TEST_DIR = "data/splits/test"
OUT_CSV = "validator_responses.csv"
SAMPLE_PER_CLASS = 5

def sample_images():
    samples = []
    classes = sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR,d))])
    for cls in classes:
        cls_dir = os.path.join(TEST_DIR, cls)
        imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        samples.extend(random.sample(imgs, min(len(imgs), SAMPLE_PER_CLASS)))
    return samples

def main():
    samples = sample_images()
    with open(OUT_CSV, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path","predicted_label","validator_correct","notes"])
        for path in samples:
            img = Image.open(path)
            plt.imshow(img); plt.axis("off"); plt.show()
            predicted = os.path.basename(os.path.dirname(path))
            resp = input(f"Predicted label '{predicted}'. Correct? (y/n): ")
            notes = input("notes (optional): ")
            writer.writerow([path, predicted, resp.lower() in ("y","yes"), notes])
    print("Saved validator responses to", OUT_CSV)

if __name__ == "__main__":
    main()
