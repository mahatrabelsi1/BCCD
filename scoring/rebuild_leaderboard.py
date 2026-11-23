"""Re-score every archived encrypted submission on main and rebuild the leaderboard.

Original submission timestamps and GitHub users are preserved. Needs the private key and
hidden labels in data/private/ (restored from secrets in the "Rebuild Leaderboard" workflow).

    python scoring/rebuild_leaderboard.py
"""
import glob
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "encryption"))

import leaderboard  # noqa: E402
from decrypt import decrypt_file  # noqa: E402
from score_submission import score_file  # noqa: E402
from validate_metadata import load_metadata  # noqa: E402

KEY = "data/private/submission_private_key.pem"


def main() -> None:
    old = leaderboard.load()
    new = leaderboard.load().iloc[0:0]
    enc_paths = sorted(glob.glob("submissions/inbox/*/*/predictions.csv.enc"))
    for enc in enc_paths:
        run_dir = os.path.dirname(enc)
        try:
            meta = load_metadata(os.path.join(run_dir, "metadata.json"), expect_dir=True)
            with tempfile.TemporaryDirectory() as tmp:
                plain = Path(tmp) / "predictions.csv"
                decrypt_file(Path(enc), Path(KEY), plain)
                scores = score_file(str(plain), split="test")
        except Exception as e:  # noqa: BLE001
            print(f"SKIP {run_dir}: {type(e).__name__}: {e}")
            continue

        prev = old[(old["team"] == meta["team"]) & (old["run_id"] == meta["run_id"])]
        user_file = Path(run_dir) / "github_user.txt"
        row = {
            **meta,
            "github_user": user_file.read_text().strip() if user_file.exists() else "",
            "submitted_at": prev["submitted_at"].iloc[0] if not prev.empty else "",
            **{k: f"{v:.6f}" for k, v in scores.items()},
        }
        new = leaderboard.upsert(new, row)
        print(f"OK   {run_dir}: mAP@0.5={scores['map50']:.4f}")

    leaderboard.save(new)
    print(f"Leaderboard rebuilt with {len(new)} run(s).")


if __name__ == "__main__":
    main()
