"""Validate a submission's metadata.json.

    python scoring/validate_metadata.py submissions/inbox/<team>/<run_id>/metadata.json
"""
import json
import os
import re
import sys

ALLOWED_AUTHOR_TYPES = {"human", "llm", "hybrid"}
REQUIRED_KEYS = ["team", "run_id", "author_type", "model"]
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MAX_MODEL_LEN = 120
MAX_NOTES_LEN = 500


class MetadataError(ValueError):
    pass


def clean_text(value, max_len: int) -> str:
    text = " ".join(str(value or "").split())  # collapse newlines / tabs / repeated spaces
    return text[:max_len]


def load_metadata(path: str, expect_dir: bool = True) -> dict:
    """Load + validate metadata. If expect_dir, team/run_id must match the folder names."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        raise MetadataError("metadata.json is not valid UTF-8 JSON") from None
    if not isinstance(data, dict):
        raise MetadataError("metadata.json must be a JSON object")

    missing = [k for k in REQUIRED_KEYS if not str(data.get(k, "")).strip()]
    if missing:
        raise MetadataError(f"metadata.json missing required field(s): {', '.join(missing)}")

    for k in ("team", "run_id"):
        if not isinstance(data[k], str) or not NAME_RE.match(data[k].strip()):
            raise MetadataError(
                f"'{k}' must be 1-64 chars of letters, digits, '_', '-', '.' (starting with a letter/digit)"
            )

    author_type = str(data["author_type"]).strip().lower()
    if author_type not in ALLOWED_AUTHOR_TYPES:
        raise MetadataError(f"invalid author_type {author_type!r}; allowed: {sorted(ALLOWED_AUTHOR_TYPES)}")

    meta = {
        "team": data["team"].strip(),
        "run_id": data["run_id"].strip(),
        "author_type": author_type,
        "model": clean_text(data["model"], MAX_MODEL_LEN),
        "notes": clean_text(data.get("notes", ""), MAX_NOTES_LEN),
    }

    if expect_dir:
        parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
        if len(parts) >= 3 and (parts[-3], parts[-2]) != (meta["team"], meta["run_id"]):
            raise MetadataError(
                f"Folder must be submissions/inbox/{meta['team']}/{meta['run_id']}/ "
                f"to match metadata (got {parts[-3]}/{parts[-2]})"
            )
    return meta


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scoring/validate_metadata.py <metadata.json>")
        sys.exit(2)
    try:
        meta = load_metadata(sys.argv[1])
    except MetadataError as e:
        print(f"Invalid metadata: {e}")
        sys.exit(1)
    print("metadata.json is valid:", meta)


if __name__ == "__main__":
    main()
