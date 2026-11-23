"""CI entrypoint: validate, decrypt, score one submission and update the leaderboard.

Run from the root of the trusted (main-branch) checkout. The PR checkout is treated as DATA only:
nothing from it is executed, and every value from it is passed through files / argv, never
interpolated into shell or workflow expressions.

    python scoring/ci_score.py --pr-root ../repo_pr --run-dir submissions/inbox/<team>/<run_id> \
        --key data/private/submission_private_key.pem --github-user <login> --result result.json

Writes a JSON result {"ok": bool, ...} to --result and exits non-zero on failure.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "encryption"))

import leaderboard  # noqa: E402
from decrypt import decrypt_file  # noqa: E402
from metrics import SubmissionError  # noqa: E402
from score_submission import score_file  # noqa: E402
from validate_metadata import MetadataError, load_metadata  # noqa: E402

MAX_SUBMISSIONS = int(os.environ.get("MAX_SUBMISSIONS_PER_TEAM", "1"))
MAX_ENC_BYTES = 5 * 1024 * 1024
MAX_META_BYTES = 10 * 1024


class Rejected(Exception):
    """A submission problem whose message is safe to show publicly."""


def safe_file(pr_root: Path, rel: str, max_bytes: int) -> Path:
    path = pr_root / rel
    if path.is_symlink() or not path.is_file():
        raise Rejected(f"{rel} must be a regular file (no symlinks)")
    if not path.resolve().is_relative_to(pr_root.resolve()):
        raise Rejected(f"{rel} resolves outside the repository")
    if path.stat().st_size > max_bytes:
        raise Rejected(f"{rel} is larger than {max_bytes // 1024} KB")
    return path


def run(args) -> dict:
    pr_root = Path(args.pr_root)
    run_dir = args.run_dir.strip("/")
    parts = run_dir.split("/")
    if len(parts) != 4 or parts[:2] != ["submissions", "inbox"]:
        raise Rejected("Submission must live in submissions/inbox/<team>/<run_id>/")

    meta_path = safe_file(pr_root, f"{run_dir}/metadata.json", MAX_META_BYTES)
    enc_path = safe_file(pr_root, f"{run_dir}/predictions.csv.enc", MAX_ENC_BYTES)

    try:
        meta = load_metadata(str(meta_path), expect_dir=True)
    except MetadataError as e:
        raise Rejected(str(e)) from None

    lb = leaderboard.load()
    already = leaderboard.count_existing(lb, meta["team"], args.github_user)
    if already >= MAX_SUBMISSIONS:
        raise Rejected(
            f"Team '{meta['team']}' / user '{args.github_user}' already has {already} scored submission(s); "
            f"the limit is {MAX_SUBMISSIONS}."
        )

    with tempfile.TemporaryDirectory() as tmp:
        plain = Path(tmp) / "predictions.csv"
        try:
            decrypt_file(enc_path, Path(args.key), plain)
        except Exception:  # noqa: BLE001
            raise Rejected("Could not decrypt predictions.csv.enc — encrypt it with encryption/public_key.pem "
                           "using encryption/encrypt.py") from None
        try:
            scores = score_file(str(plain), split="test")
        except SubmissionError as e:
            raise Rejected(str(e)) from None

    row = {**meta, "github_user": args.github_user, **{k: f"{v:.6f}" for k, v in scores.items()}}
    lb = leaderboard.upsert(lb, row)
    leaderboard.save(lb)

    # Archive the (still encrypted) submission on main so the leaderboard can be rebuilt later.
    archive = Path(run_dir)
    archive.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(enc_path, archive / "predictions.csv.enc")
    shutil.copyfile(meta_path, archive / "metadata.json")
    (archive / "github_user.txt").write_text(args.github_user + "\n", encoding="utf-8")

    ranked = leaderboard.ranked(lb)
    me = ranked[(ranked["team"] == meta["team"]) & (ranked["run_id"] == meta["run_id"])].iloc[0]
    return {"ok": True, **meta, **scores, "rank": int(me["rank"]), "total": len(ranked)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr-root", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--github-user", default="")
    ap.add_argument("--result", required=True)
    args = ap.parse_args()

    try:
        result = run(args)
        code = 0
    except Rejected as e:
        result, code = {"ok": False, "error": str(e)}, 1
    except Exception:  # noqa: BLE001 - never echo internals (could include paths/labels) to the PR
        traceback.print_exc()
        result, code = {"ok": False, "error": "Internal scoring error — please contact the organizers."}, 1

    Path(args.result).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    sys.exit(code)


if __name__ == "__main__":
    main()
