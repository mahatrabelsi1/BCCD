# BCCD Blood Cell Detection Challenge

A **GitHub-native mini competition** in the style of Kaggle. Participants train an object detector locally on the
[BCCD dataset](https://github.com/Shenggan/BCCD_Dataset), submit **encrypted predictions** through a Pull Request,
and a GitHub Action scores them against hidden test labels and updates a public leaderboard.

**Leaderboard and competition page:** `https://<owner>.github.io/<repo>/` (GitHub Pages from `/docs`)

| | |
|---|---|
| Task | Object detection: RBC, WBC, Platelets |
| Images | 640×480 blood smear microscope images: 205 train · 87 val · 72 test (hidden), **with image noise** |
| Metric | **mAP@0.5**: mean over the 3 classes of AP at IoU ≥ 0.5 (VOC all-point) |
| Submissions | 1 scored submission per team **and** per GitHub account |
| What you submit | `predictions.csv.enc` + `metadata.json`, never code |

---

## Data

```text
data/public/
├── train/images/*.jpg            205 images
├── val/images/*.jpg               87 images
├── test/images/*.jpg              72 images (labels hidden)
├── train_annotations.csv         image_id,label,xmin,ymin,xmax,ymax
├── val_annotations.csv           image_id,label,xmin,ymin,xmax,ymax
├── {train,val,test}_images.csv   image_id,width,height
├── sample_submission.csv
└── LICENSE_BCCD                  MIT licence of the original dataset
```

| Split | Images | RBC | WBC | Platelets |
|---|---:|---:|---:|---:|
| train | 205 | 2,381 | 214 | 209 |
| val | 87 | 967 | 87 | 83 |
| test | 72 | hidden | hidden | hidden |

The split is BCCD's official one. Images were renamed to opaque ids and re-encoded. Coordinates are pixels,
origin at top-left. The original BCCD labels are known to miss some RBCs, so expect some label noise.

### Image noise (robustness challenge)

About 90% of the images in **every split** were corrupted with 1 or 2 random photometric corruptions:

| Corruption | What it looks like |
|---|---|
| Gaussian noise | grainy pixels |
| Salt & pepper | isolated black/white dots (easy to confuse with platelets!) |
| Gaussian blur | soft edges, merged RBC borders |
| Heavy JPEG compression | 8×8 block artifacts, colour banding |
| Colour / brightness / contrast jitter | stain-like colour casts, washed-out or dark images |

Only pixels were changed, never geometry, so every box is still exact. Which corruption was applied to which image
is not published. Denoising, augmentation and robust training are all fair game.

## Evaluation

For each class, all predicted boxes across the test set are sorted by `score`. A box is a true positive if its IoU
with a still-unmatched ground-truth box of the same class and image is ≥ 0.5. AP is the area under the
precision–recall curve (all-point interpolation). **mAP@0.5 = mean(AP_RBC, AP_WBC, AP_Platelets)**, so rare classes
weigh as much as RBCs. Implementation: [`scoring/metrics.py`](scoring/metrics.py).

Run the exact same metric locally on validation:

```bash
pip install -r requirements.txt
python scoring/score_submission.py --pred val_predictions.csv --split val
```

## Submission format

```csv
image_id,label,score,xmin,ymin,xmax,ymax
bccd_0002,RBC,0.97,412,88,515,190
bccd_0002,WBC,0.99,201,150,330,281
```

- one row per box; `image_id` from `data/public/test_images.csv`
- `label` ∈ {`RBC`, `WBC`, `Platelets`}, `score` ∈ [0, 1], `xmax > xmin`, `ymax > ymin`
- at most 300 boxes per image; boxes are clipped to the image

## How to participate

1. **Fork** this repo and clone your fork.
2. **Train** on `train` (+ `val` if you like). The starter trains a Faster R-CNN and writes a valid file:
   ```bash
   pip install torch torchvision          # see pytorch.org for your CUDA version
   python starter/baseline_fasterrcnn.py --epochs 20
   ```
3. **Validate** your test predictions:
   ```bash
   python scoring/validate_submission.py predictions.csv
   ```
4. **Encrypt** them:
   ```bash
   python encryption/encrypt.py predictions.csv encryption/public_key.pem predictions.csv.enc
   ```
5. **Add exactly two files** (copy [`metadata_template.json`](submissions/inbox/metadata_template.json)):
   ```text
   submissions/inbox/<team>/<run_id>/predictions.csv.enc
   submissions/inbox/<team>/<run_id>/metadata.json
   ```
   ```json
   { "team": "my_team", "run_id": "frcnn_v1", "author_type": "human",
     "model": "Faster R-CNN R50-FPN", "notes": "hflip/vflip aug, 20 epochs" }
   ```
   `author_type` ∈ `human` | `llm` | `hybrid`. `team`/`run_id` use letters, digits, `_ - .` and must match the folder names.
6. **Open a Pull Request** to `main`. The bot comments your mAP@0.5, per-class AP and rank, updates the
   leaderboard and closes the PR. If the submission is invalid it comments why; push a fix to the same PR.

## Rules

- One scored submission per team and per GitHub account, so validate on `val` first.
- No external labelled blood-cell data. **Do not use the original BCCD annotations/images to recover test labels.**
  ImageNet/COCO-pretrained backbones are fine; mention them in `notes`.
- No manual labelling of test images.
- Declare `author_type` honestly.
- Organizers may remove entries that break the rules.

---

## Repository layout

```text
.github/workflows/
  score_submission.yml      PR → validate → decrypt → score → commit leaderboard → comment → close
  rebuild_leaderboard.yml   manual: re-score every archived submission
data/public/                public images + labels (train/val), test image list
encryption/                 encrypt.py, decrypt.py, generate_keys.py, public_key.pem
scoring/
  metrics.py                submission validation + mAP@0.5 (single source of truth)
  score_submission.py       local scoring (--split val|test)
  validate_submission.py    format check without labels
  validate_metadata.py      metadata.json check
  leaderboard.py            leaderboard CSV/MD writer
  ci_score.py               CI entrypoint (used by the workflow)
  rebuild_leaderboard.py    re-score archived submissions
leaderboard/                leaderboard.csv (source of truth), leaderboard.md
docs/                       GitHub Pages site (index.html + leaderboard.csv copy)
starter/                    Faster R-CNN baseline
scripts/                    organizer tools: prepare_data.py, make_secrets.py
submissions/inbox/          encrypted submissions archived by the bot
```

## Organizer setup

Hidden labels (`data/private/test_labels.csv`) and the private key (`data/private/submission_private_key.pem`)
are git-ignored and must **never** be committed. CI gets them from repository secrets.

1. *(Already done once; only redo to reset the competition.)* Build the data and a key pair:
   ```bash
   git clone https://github.com/Shenggan/BCCD_Dataset.git ../BCCD_Dataset
   python scripts/prepare_data.py --src ../BCCD_Dataset
   python encryption/generate_keys.py          # writes encryption/public_key.pem + data/private/submission_private_key.pem
   ```
2. Upload the secrets (`TEST_LABELS_B64_01..08`, `SUBMISSION_PRIVATE_KEY_B64_01..08`; one chunk each is enough here):
   ```bash
   python scripts/make_secrets.py --set --repo <owner>/<repo>    # uses the GitHub CLI
   ```
   or paste the files in `data/private/secrets/` into *Settings → Secrets and variables → Actions*.
3. *Settings → Actions → General → Workflow permissions*: **Read and write**.
   If `main` is branch-protected, allow GitHub Actions to push to it.
4. *Settings → Pages*: deploy from branch **`main`**, folder **`/docs`**.
5. Keep `data/private/` backed up somewhere safe. Without the private key, submissions cannot be decrypted.

To allow more than one submission per team, set `MAX_SUBMISSIONS_PER_TEAM` in the workflow env.

### Security notes

- The scoring workflow uses `pull_request_target` (needed for secrets on fork PRs). It runs **only scripts from
  `main`**; the PR checkout is read as data, symlinks are rejected, and participant-supplied values never reach a
  shell or a `${{ }}` expression.
- Error messages posted on PRs are limited to format problems and never reveal labels.
- BCCD is public, so a determined cheater could match test images to the original dataset despite renaming and
  re-encoding. This is enforced by the rules, not by cryptography. Sanity-check suspiciously high scores.

## Credits

Dataset: [BCCD](https://github.com/Shenggan/BCCD_Dataset) by Shenggan et al., MIT licence (`data/public/LICENSE_BCCD`).
Competition structure adapted from
[GNN-MVP-Node-Classification-Under-Noise](https://github.com/mahatrabelsi1/GNN-MVP-Node-Classification-Under-Noise).
