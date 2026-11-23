"""Score a predictions CSV locally.

Participants (validation split, labels are public):
    python scoring/score_submission.py --pred val_predictions.csv --split val

Organizers (hidden test split, needs data/private/test_labels.csv):
    python scoring/score_submission.py --pred predictions.csv --split test
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import CLASSES, SubmissionError, evaluate, load_submission  # noqa: E402

PUBLIC = "data/public"
LABELS = {
    "train": f"{PUBLIC}/train_annotations.csv",
    "val": f"{PUBLIC}/val_annotations.csv",
    "test": "data/private/test_labels.csv",
}


def score_file(pred_path: str, split: str = "test", labels_path: str | None = None) -> dict:
    images = pd.read_csv(f"{PUBLIC}/{split}_images.csv", dtype={"image_id": str})
    gt = pd.read_csv(labels_path or LABELS[split], dtype={"image_id": str})
    pred = load_submission(pred_path, images)
    return evaluate(pred, gt)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute mAP@0.5 for a predictions CSV.")
    ap.add_argument("--pred", required=True, help="Path to predictions CSV")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--labels", default=None, help="Override ground-truth CSV path")
    args = ap.parse_args()

    try:
        res = score_file(args.pred, args.split, args.labels)
    except SubmissionError as e:
        print(f"Invalid submission: {e}")
        sys.exit(1)

    print(f"Split: {args.split}  |  file: {args.pred}")
    for c in CLASSES:
        print(f"  AP@0.5 {c:<9s} {res[f'ap_{c}']:.4f}")
    print(f"  mAP@0.5         {res['map50']:.4f}")


if __name__ == "__main__":
    main()
