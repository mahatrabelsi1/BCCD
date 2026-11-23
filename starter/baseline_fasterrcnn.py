"""Starter baseline: torchvision Faster R-CNN (ResNet-50 FPN, COCO-pretrained), fine-tuned on BCCD.

    pip install torch torchvision pandas numpy pillow
    python starter/baseline_fasterrcnn.py --epochs 20            # train on train, report val mAP@0.5
    python starter/baseline_fasterrcnn.py --epochs 20 --train-on-val   # final: train on train+val

Writes:
    starter/out/val_predictions.csv   (scored locally with the official metric)
    starter/out/predictions.csv       (test set, ready for encryption)

Runs on CPU (slow: roughly minutes per epoch) or a CUDA GPU (fast). Free Colab/Kaggle GPUs work fine.
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scoring"))
from metrics import CLASSES, SUBMISSION_COLUMNS, evaluate  # noqa: E402

DATA = ROOT / "data" / "public"
LABEL_TO_ID = {c: i + 1 for i, c in enumerate(CLASSES)}  # 0 = background
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


class BCCD(torch.utils.data.Dataset):
    def __init__(self, split: str, train: bool):
        self.split, self.train = split, train
        self.ids = pd.read_csv(DATA / f"{split}_images.csv", dtype={"image_id": str})["image_id"].tolist()
        ann_path = DATA / f"{split}_annotations.csv"
        self.ann = (pd.read_csv(ann_path, dtype={"image_id": str}).groupby("image_id")
                    if ann_path.exists() else None)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        iid = self.ids[i]
        img = to_tensor(Image.open(DATA / self.split / "images" / f"{iid}.jpg").convert("RGB"))
        target = {"image_id": iid}
        if self.ann is not None:
            a = self.ann.get_group(iid) if iid in self.ann.groups else pd.DataFrame(columns=["label"])
            boxes = torch.tensor(a[["xmin", "ymin", "xmax", "ymax"]].to_numpy(np.float32)).reshape(-1, 4)
            labels = torch.tensor([LABEL_TO_ID[x] for x in a["label"]], dtype=torch.int64)
            if self.train:
                _, h, w = img.shape
                if random.random() < 0.5:
                    img = img.flip(-1)
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                if random.random() < 0.5:
                    img = img.flip(-2)
                    boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
            target.update(boxes=boxes, labels=labels)
        return img, target


def collate(batch):
    return tuple(zip(*batch))


def build_model(score_thresh: float) -> torch.nn.Module:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT", box_score_thresh=score_thresh, box_detections_per_img=300,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASSES) + 1)
    return model


@torch.no_grad()
def predict(model, split: str, device) -> pd.DataFrame:
    model.eval()
    loader = torch.utils.data.DataLoader(BCCD(split, train=False), batch_size=4, collate_fn=collate)
    rows = []
    for imgs, targets in loader:
        outs = model([im.to(device) for im in imgs])
        for t, o in zip(targets, outs):
            for box, lab, s in zip(o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(), o["scores"].cpu().numpy()):
                rows.append([t["image_id"], ID_TO_LABEL[int(lab)], float(s), *map(float, box)])
    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--score-thresh", type=float, default=0.01)
    ap.add_argument("--train-on-val", action="store_true", help="Also train on val (for the final submission)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "starter" / "out")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_sets = [BCCD("train", train=True)] + ([BCCD("val", train=True)] if args.train_on_val else [])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(train_sets), batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=min(4, os.cpu_count() or 1),
    )

    model = build_model(args.score_thresh).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for imgs, targets in loader:
            imgs = [im.to(device) for im in imgs]
            targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
            loss = sum(model(imgs, targets).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        print(f"epoch {epoch:3d}/{args.epochs}  loss {total / len(loader):.4f}")

    args.out.mkdir(parents=True, exist_ok=True)

    val_pred = predict(model, "val", device)
    val_pred.to_csv(args.out / "val_predictions.csv", index=False)
    gt = pd.read_csv(DATA / "val_annotations.csv", dtype={"image_id": str})
    res = evaluate(val_pred, gt)
    note = " (trained on val, optimistic!)" if args.train_on_val else ""
    print(f"VAL mAP@0.5 = {res['map50']:.4f}{note}  " + "  ".join(f"{c}={res['ap_' + c]:.4f}" for c in CLASSES))

    test_pred = predict(model, "test", device)
    test_pred.to_csv(args.out / "predictions.csv", index=False)
    print(f"Test predictions: {args.out / 'predictions.csv'} ({len(test_pred)} boxes)")
    print("Next: python scoring/validate_submission.py starter/out/predictions.csv")


if __name__ == "__main__":
    main()
