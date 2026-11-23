# Submissions inbox (encrypted)

A submission PR must add **exactly two files** in one new folder:

```text
submissions/inbox/<team>/<run_id>/predictions.csv.enc
submissions/inbox/<team>/<run_id>/metadata.json
```

`<team>` and `<run_id>` are 1 to 64 characters from letters, digits, `_`, `-` and `.`, starting with a letter or digit,
and they must match the `team` / `run_id` fields in `metadata.json`.

## Plain predictions (before encryption)

```csv
image_id,label,score,xmin,ymin,xmax,ymax
bccd_0002,RBC,0.97,412,88,515,190
```

Check it, then encrypt it:

```bash
python scoring/validate_submission.py predictions.csv
python encryption/encrypt.py predictions.csv encryption/public_key.pem predictions.csv.enc
```

Never commit the plain `predictions.csv`.

## metadata.json

Start from [`metadata_template.json`](metadata_template.json):

```json
{
  "team": "my_team",
  "run_id": "frcnn_v1",
  "author_type": "human",
  "model": "Faster R-CNN R50-FPN",
  "notes": "optional, max 500 chars"
}
```

`author_type` is one of `human`, `llm` or `hybrid`.

Scored submissions are archived here automatically by the bot, still encrypted, so the leaderboard can be rebuilt.
