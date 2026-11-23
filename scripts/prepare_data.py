"""Organizer-only: build the competition data from a raw clone of the BCCD dataset.

    git clone https://github.com/Shenggan/BCCD_Dataset.git /path/to/BCCD_Dataset
    python scripts/prepare_data.py --src /path/to/BCCD_Dataset

Outputs
-------
data/public/                         (committed)
    train/images/<image_id>.jpg
    val/images/<image_id>.jpg
    test/images/<image_id>.jpg
    train_annotations.csv            image_id,label,xmin,ymin,xmax,ymax
    val_annotations.csv              image_id,label,xmin,ymin,xmax,ymax
    {train,val,test}_images.csv      image_id,width,height   (test: NO labels anywhere public)
    sample_submission.csv            image_id,label,score,xmin,ymin,xmax,ymax
data/private/                        (git-ignored, NEVER commit)
    test_labels.csv                  hidden ground truth used by CI
    id_mapping.csv                   image_id -> original BCCD filename

Images are renamed to opaque ids (shuffled with a fixed seed) and re-encoded, so the
public test images cannot be matched to the original BCCD filenames by name or file hash.

Noise: every split gets the same random photometric corruptions (1-2 per image out of
gaussian noise, salt & pepper, blur, heavy JPEG compression, colour/brightness/contrast jitter;
~10% of images stay clean). Corruptions are pixel-only, so the boxes stay exact. The per-image
corruption is logged in data/private/id_mapping.csv (organizer only).
The official BCCD split (ImageSets/Main/{train,val,test}.txt) is used.
"""
import argparse
import csv
import io
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

CLASSES = ["RBC", "WBC", "Platelets"]
SPLITS = ["train", "val", "test"]


def read_split(src: Path, split: str) -> list[str]:
    path = src / "BCCD" / "ImageSets" / "Main" / f"{split}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def gaussian_noise(im: Image.Image, rng: np.random.Generator) -> Image.Image:
    sigma = rng.uniform(8, 25)
    a = np.asarray(im, dtype=np.float32) + rng.normal(0, sigma, (im.height, im.width, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def salt_pepper(im: Image.Image, rng: np.random.Generator) -> Image.Image:
    a = np.asarray(im).copy()
    p = rng.uniform(0.005, 0.02)
    m = rng.random(a.shape[:2])
    a[m < p / 2] = 0
    a[m > 1 - p / 2] = 255
    return Image.fromarray(a)


def blur(im: Image.Image, rng: np.random.Generator) -> Image.Image:
    return im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.8, 2.2)))


def jpeg_artifacts(im: Image.Image, rng: np.random.Generator) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=int(rng.integers(10, 35)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def color_jitter(im: Image.Image, rng: np.random.Generator) -> Image.Image:
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.65, 1.35))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.65, 1.35))
    a = np.asarray(im, dtype=np.float32) * rng.uniform(0.85, 1.15, 3)  # per-channel stain/colour shift
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "salt_pepper": salt_pepper,
    "blur": blur,
    "jpeg": jpeg_artifacts,
    "color_jitter": color_jitter,
}


def corrupt(im: Image.Image, rng: np.random.Generator, clean_prob: float) -> tuple[Image.Image, str]:
    """Apply 1-2 random photometric corruptions. Geometry is untouched, so boxes stay valid."""
    if rng.random() < clean_prob:
        return im, "clean"
    names = list(rng.choice(list(CORRUPTIONS), size=int(rng.integers(1, 3)), replace=False))
    for n in names:
        im = CORRUPTIONS[n](im, rng)
    return im, "+".join(names)


def parse_boxes(xml_path: Path, width: int, height: int) -> tuple[list[dict], int]:
    """Return cleaned boxes (clipped to the image, degenerate boxes dropped) and the drop count."""
    boxes, dropped = [], 0
    for obj in ET.parse(xml_path).getroot().iter("object"):
        label = obj.find("name").text.strip()
        if label not in CLASSES:
            raise ValueError(f"Unknown class {label!r} in {xml_path}")
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = sorted((min(max(x1, 0), width), min(max(x2, 0), width)))
        y1, y2 = sorted((min(max(y1, 0), height), min(max(y2, 0), height)))
        if x2 - x1 < 1 or y2 - y1 < 1:
            dropped += 1
            continue
        boxes.append({"label": label, "xmin": int(x1), "ymin": int(y1), "xmax": int(x2), "ymax": int(y2)})
    return boxes, dropped


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="Path to a clone of Shenggan/BCCD_Dataset")
    ap.add_argument("--out", default=Path("data"), type=Path, help="Output data directory (default: data)")
    ap.add_argument("--seed", default=2026, type=int, help="Seed for the id shuffle")
    ap.add_argument("--quality", default=95, type=int, help="JPEG re-encode quality")
    ap.add_argument("--clean-prob", default=0.1, type=float,
                    help="Fraction of images left without noise (default 0.1)")
    ap.add_argument("--no-noise", action="store_true", help="Disable image corruptions")
    args = ap.parse_args()
    noise_rng = np.random.default_rng(args.seed)

    public, private = args.out / "public", args.out / "private"
    for split in SPLITS:
        shutil.rmtree(public / split, ignore_errors=True)
    private.mkdir(parents=True, exist_ok=True)

    names = {split: read_split(args.src, split) for split in SPLITS}
    all_names = [n for split in SPLITS for n in names[split]]
    if len(set(all_names)) != len(all_names):
        raise ValueError("Official BCCD splits overlap")

    shuffled = all_names[:]
    random.Random(args.seed).shuffle(shuffled)
    new_id = {name: f"bccd_{i:04d}" for i, name in enumerate(shuffled, start=1)}

    ann_fields = ["image_id", "label", "xmin", "ymin", "xmax", "ymax"]
    mapping, total_dropped = [], 0
    for split in SPLITS:
        ann_rows, img_rows = [], []
        for name in sorted(names[split], key=lambda n: new_id[n]):
            iid = new_id[name]
            src_img = args.src / "BCCD" / "JPEGImages" / f"{name}.jpg"
            dst_img = public / split / "images" / f"{iid}.jpg"
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src_img) as im:
                im = im.convert("RGB")
                width, height = im.size
                noise = "clean"
                if not args.no_noise:
                    im, noise = corrupt(im, noise_rng, args.clean_prob)
                im.save(dst_img, "JPEG", quality=args.quality)  # re-encode: strips metadata, changes file hash

            boxes, dropped = parse_boxes(args.src / "BCCD" / "Annotations" / f"{name}.xml", width, height)
            total_dropped += dropped
            ann_rows += [{"image_id": iid, **b} for b in boxes]
            img_rows.append({"image_id": iid, "width": width, "height": height})
            mapping.append({"image_id": iid, "split": split, "original": f"{name}.jpg", "noise": noise})

        write_csv(public / f"{split}_images.csv", ["image_id", "width", "height"], img_rows)
        if split == "test":
            write_csv(private / "test_labels.csv", ann_fields, ann_rows)
            sample = [
                {"image_id": r["image_id"], "label": lab, "score": 0.5,
                 "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 200}
                for r in img_rows for lab in CLASSES
            ]
            write_csv(public / "sample_submission.csv",
                      ["image_id", "label", "score", "xmin", "ymin", "xmax", "ymax"], sample)
        else:
            write_csv(public / f"{split}_annotations.csv", ann_fields, ann_rows)

        counts = {c: sum(r["label"] == c for r in ann_rows) for c in CLASSES}
        print(f"{split:5s}: {len(img_rows):3d} images, {len(ann_rows):5d} boxes {counts}")

    write_csv(private / "id_mapping.csv", ["image_id", "split", "original", "noise"], mapping)
    kinds = {}
    for m in mapping:
        for k in m["noise"].split("+"):
            kinds[k] = kinds.get(k, 0) + 1
    print("Noise applied (images per corruption):", dict(sorted(kinds.items())))
    print(f"Dropped {total_dropped} degenerate box(es) after clipping.")
    print(f"Hidden labels -> {private / 'test_labels.csv'} (do NOT commit)")


if __name__ == "__main__":
    main()
