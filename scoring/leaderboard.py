"""Leaderboard storage: leaderboard/leaderboard.csv (source of truth), leaderboard.md, docs/leaderboard.csv."""
import os
import shutil
from datetime import datetime, timezone

import pandas as pd

LB_CSV = "leaderboard/leaderboard.csv"
LB_MD = "leaderboard/leaderboard.md"
DOCS_CSV = "docs/leaderboard.csv"
COLUMNS = [
    "team", "run_id", "github_user", "author_type", "model", "notes",
    "map50", "ap_RBC", "ap_WBC", "ap_Platelets", "submitted_at",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load() -> pd.DataFrame:
    if not os.path.exists(LB_CSV):
        return pd.DataFrame(columns=COLUMNS)
    lb = pd.read_csv(LB_CSV, dtype=str, keep_default_na=False)
    for c in COLUMNS:
        if c not in lb.columns:
            lb[c] = ""
    return lb[COLUMNS]


def count_existing(lb: pd.DataFrame, team: str, github_user: str) -> int:
    same_team = lb["team"].str.lower() == team.lower()
    same_user = (lb["github_user"].str.lower() == github_user.lower()) if github_user else False
    return int((same_team | same_user).sum())


def upsert(lb: pd.DataFrame, row: dict) -> pd.DataFrame:
    """Replace any row with the same (team, run_id), keeping its original timestamp if not given."""
    same = (lb["team"] == row["team"]) & (lb["run_id"] == row["run_id"])
    if not row.get("submitted_at"):
        row["submitted_at"] = lb.loc[same, "submitted_at"].iloc[0] if same.any() else now_utc()
    lb = lb[~same]
    return pd.concat([lb, pd.DataFrame([{c: row.get(c, "") for c in COLUMNS}])], ignore_index=True)


def ranked(lb: pd.DataFrame) -> pd.DataFrame:
    """Sort best-first (ties broken by earliest submission) with competition ranks (1,2,2,4)."""
    df = lb.copy()
    df["_score"] = pd.to_numeric(df["map50"], errors="coerce").round(6)
    df = df.sort_values(["_score", "submitted_at"], ascending=[False, True]).reset_index(drop=True)
    df["rank"] = df["_score"].rank(method="min", ascending=False).astype("Int64")
    return df


def _fmt(x) -> str:
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "—"


def _md_cell(s: str) -> str:
    return str(s).replace("|", "\\|")


def save(lb: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(LB_CSV), exist_ok=True)
    df = ranked(lb)
    df[COLUMNS].to_csv(LB_CSV, index=False)

    lines = [
        "# Leaderboard — BCCD Blood Cell Detection\n\n",
        "Metric: **mAP@0.5** (mean of per-class AP over RBC, WBC, Platelets). Higher is better.\n\n",
        "| Rank | Team | Run | Author | Model | mAP@0.5 | AP RBC | AP WBC | AP Platelets | Submitted at |\n",
        "|---:|---|---|---|---|---:|---:|---:|---:|---|\n",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['rank']} | {_md_cell(r['team'])} | {_md_cell(r['run_id'])} | {r['author_type']} | "
            f"{_md_cell(r['model'])} | **{_fmt(r['map50'])}** | {_fmt(r['ap_RBC'])} | {_fmt(r['ap_WBC'])} | "
            f"{_fmt(r['ap_Platelets'])} | {r['submitted_at']} |\n"
        )
    if df.empty:
        lines.append("| — | *No submissions yet* | | | | | | | | |\n")
    with open(LB_MD, "w", encoding="utf-8") as f:
        f.writelines(lines)

    os.makedirs(os.path.dirname(DOCS_CSV), exist_ok=True)
    shutil.copyfile(LB_CSV, DOCS_CSV)
