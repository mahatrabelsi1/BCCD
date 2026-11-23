"""Organizer-only: split private files into base64 chunks for GitHub Actions secrets.

GitHub secrets are limited to 48 KB each, so each private file is base64-encoded and split
across up to 8 numbered secrets, which the workflows concatenate back together:

    TEST_LABELS_B64_01..08           <- data/private/test_labels.csv
    SUBMISSION_PRIVATE_KEY_B64_01..08 <- data/private/submission_private_key.pem

    python scripts/make_secrets.py            # writes chunk files to data/private/secrets/
    python scripts/make_secrets.py --set      # also uploads them with the GitHub CLI (gh secret set)
"""
import argparse
import base64
import subprocess
from pathlib import Path

CHUNK = 40_000
MAX_CHUNKS = 8
FILES = {
    "TEST_LABELS_B64": Path("data/private/test_labels.csv"),
    "SUBMISSION_PRIVATE_KEY_B64": Path("data/private/submission_private_key.pem"),
}
OUT = Path("data/private/secrets")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="store_true", help="Upload with `gh secret set` (needs gh auth + repo access)")
    ap.add_argument("--repo", default=None, help="owner/repo for gh (default: current repo)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for prefix, path in FILES.items():
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        chunks = [b64[i:i + CHUNK] for i in range(0, len(b64), CHUNK)]
        if len(chunks) > MAX_CHUNKS:
            raise SystemExit(f"{path} is too large for {MAX_CHUNKS} secrets")
        for i in range(1, MAX_CHUNKS + 1):
            name = f"{prefix}_{i:02d}"
            value = chunks[i - 1] if i <= len(chunks) else ""
            if value:
                (OUT / f"{name}.txt").write_text(value, encoding="ascii")
                print(f"{name}: {len(value)} chars -> {OUT / (name + '.txt')}")
            if args.set and value:
                cmd = ["gh", "secret", "set", name] + (["--repo", args.repo] if args.repo else [])
                subprocess.run(cmd, input=value, text=True, check=True)
        print(f"{prefix}: {len(chunks)} chunk(s) used; leave the remaining numbered secrets unset.\n")


if __name__ == "__main__":
    main()
