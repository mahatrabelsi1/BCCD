# Organizer guide

This file is for the people running the competition. Participants only need the [README](README.md).

## How scoring works

```text
participant PR ──► .github/workflows/score_submission.yml
                    1. checks the PR only adds submissions/inbox/<team>/<run_id>/{predictions.csv.enc, metadata.json}
                    2. restores hidden labels + private key from repository secrets
                    3. scoring/ci_score.py: validate → decrypt → score mAP@0.5 → update leaderboard
                    4. commits leaderboard + archived (still encrypted) submission to main
                    5. comments the score on the PR and closes it (or comments the error and leaves it open)
```

The website (`docs/index.html`, GitHub Pages from `main` / `/docs`) reads `docs/leaderboard.csv`, which the bot
keeps in sync with `leaderboard/leaderboard.csv`.

## Setup (already done for mahatrabelsi1/BCCD)

Hidden labels (`data/private/test_labels.csv`) and the private key (`data/private/submission_private_key.pem`)
are git-ignored and must **never** be committed. **Keep a backup of `data/private/`**: without the private key,
submissions cannot be decrypted.

1. Build the data and a key pair (only needed to reset the competition):
   ```bash
   git clone https://github.com/Shenggan/BCCD_Dataset.git ../BCCD_Dataset
   python scripts/prepare_data.py --src ../BCCD_Dataset          # add --no-noise for clean images
   python encryption/generate_keys.py
   python scripts/make_secrets.py                                # writes data/private/secrets/*.txt
   ```
2. **Settings → Secrets and variables → Actions → New repository secret**: create `TEST_LABELS_B64_01` and
   `SUBMISSION_PRIVATE_KEY_B64_01`, each with the full content of the matching file in `data/private/secrets/`.
   Larger files may need `_02` … `_08`; `make_secrets.py` tells you how many.
3. **Settings → Actions → General → Workflow permissions** → *Read and write permissions*.
   If `main` is branch-protected, allow GitHub Actions to push to it.
4. **Settings → Pages** → Deploy from a branch → `main` / `/docs`.

## Common tasks

| Task | How |
|---|---|
| Allow more submissions per team | Add `MAX_SUBMISSIONS_PER_TEAM: "3"` to the `env:` of the scoring step in `score_submission.yml` |
| Remove an entry | Delete its row from `leaderboard/leaderboard.csv` and its folder in `submissions/inbox/`, then run **Actions → Rebuild Leaderboard** |
| Re-score everything (e.g. after fixing labels) | Update the secrets, then run **Actions → Rebuild Leaderboard** |
| Score a file locally | `python scoring/score_submission.py --pred file.csv --split test` (needs `data/private/`) |

## Repository layout

```text
.github/workflows/
  score_submission.yml      PR → score → leaderboard → comment → close
  rebuild_leaderboard.yml   manual: re-score every archived submission
data/public/                images + train/val labels + test image list
encryption/                 encrypt.py, decrypt.py, generate_keys.py, public_key.pem
scoring/
  metrics.py                submission validation + mAP@0.5 (single source of truth)
  score_submission.py       local scoring (--split val|test)
  validate_submission.py    format check without labels
  validate_metadata.py      metadata.json check
  leaderboard.py            leaderboard CSV/MD writer
  ci_score.py               CI entrypoint
  rebuild_leaderboard.py    re-score archived submissions
leaderboard/                leaderboard.csv (source of truth), leaderboard.md
docs/                       competition website (GitHub Pages)
starter/                    Faster R-CNN baseline
scripts/                    prepare_data.py (data + noise), make_secrets.py
submissions/inbox/          encrypted submissions archived by the bot
```

## Security notes

- The workflow uses `pull_request_target`, which is needed so fork PRs can use secrets. It runs **only scripts
  from `main`**: the PR is read as data, symlinks are rejected, and participant values never reach a shell or
  a `${{ }}` expression.
- PR comments only report format problems and scores, never labels.
- BCCD is public, so a determined cheater could match test images to the original dataset despite renaming,
  re-encoding and noise. The rules forbid this; double-check suspiciously high scores.
