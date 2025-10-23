import os, shutil, argparse, random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--src", default="data/augmented")
parser.add_argument("--out", default="data/splits")
parser.add_argument("--train", type=float, default=0.7)
parser.add_argument("--val", type=float, default=0.15)
parser.add_argument("--test", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

assert abs(args.train + args.val + args.test - 1.0) < 1e-6

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main():
    if os.path.exists(os.path.join(args.out, "train")):
        print("[INFO] Already split. Delete data/splits to re-split.")
        return

    classes = sorted([d for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d))])
    for cls in tqdm(classes, desc="Classes"):
        cls_src = os.path.join(args.src, cls)
        imgs = [f for f in os.listdir(cls_src) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        imgs_full = [os.path.join(cls_src, f) for f in imgs]
        train_imgs, rest = train_test_split(imgs_full, train_size=args.train, random_state=args.seed, shuffle=True)
        val_ratio = args.val / (args.val + args.test)
        val_imgs, test_imgs = train_test_split(rest, train_size=val_ratio, random_state=args.seed, shuffle=True)

        for split_name, arr in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            out_dir = os.path.join(args.out, split_name, cls)
            ensure_dir(out_dir)
            for src_path in arr:
                shutil.copy(src_path, os.path.join(out_dir, os.path.basename(src_path)))
    print("[INFO] Split done under", args.out)

if __name__ == "__main__":
    main()
