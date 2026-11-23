"""Check a test-set predictions CSV for format errors before encrypting it (no labels needed).

    python scoring/validate_submission.py predictions.csv
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import SubmissionError, load_submission  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scoring/validate_submission.py <predictions.csv>")
        sys.exit(2)

    images = pd.read_csv("data/public/test_images.csv", dtype={"image_id": str})
    try:
        pred = load_submission(sys.argv[1], images)
    except SubmissionError as e:
        print(f"Invalid submission: {e}")
        sys.exit(1)

    covered = pred["image_id"].nunique()
    print(f"OK: {len(pred)} boxes over {covered}/{len(images)} test images.")
    if covered < len(images):
        print("Warning: some test images have no predictions (allowed, but costs recall).")


if __name__ == "__main__":
    main()
