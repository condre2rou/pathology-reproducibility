# Tutorial

## 1. Set up the CPU analysis environment

From this repository's root, use Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py verify
```

On Windows, activate with `.venv\Scripts\activate` instead. The commands below
use the same Python executable as the activated environment.

## 2. Recompute the published aggregate analyses

```bash
python reproduce.py aggregates --out outputs/published_results
```

This recomputes the following from the included task/repetition estimates:

- The balanced-pool benchmark, grouped threshold selection and budget/headroom checks.
- Six policies under random and balanced sampling, with task weighting.
- Source-cohort percentile bootstrap intervals and single-pilot score intervals.
- Source-cohort-held-out meta-models.
- Test-size/gain-margin restrictions, delete-one-cohort summaries and matched
  interval diagnostics from the saved reconstruction records.
- All 16 main/supplementary CSV tables and 13 figure sets.

Inspect `comparison.json`: `all_match` should be `true`. Expected headline values
include balanced-pool score AUROC `0.7931306306306306`, random-pilot score AUROC
`0.46782009376703704`, 3941 eligible random pilots and historical-rule false-stop
rate `0.4778342250615149`.

The command does **not** extract features or refit the per-task models. To only
redraw figures and export tables from stored summaries, use:

```bash
python reproduce.py figures --out outputs/figures_only
```

## 3. Refit a real manuscript task

The local delivery contains TCGA-COAD features in `data/example/`. Verify them:

```bash
python scripts/verify_example.py
python reproduce.py example --out outputs/real_coad
```

The example uses the entire observed 112-participant COAD pool, in its preserved
row order, not just the 40-record subset. It runs:

1. Five complete-pool cross-validation partitions and 30 balanced draws at every
   eligible label budget.
2. Thirty hash-defined test splits; random and balanced 40-record pilots; pilot
   and expanded-pool model fits on common test records.
3. Paired test-gain intervals and 100 record-resampling/refitting attempts on each
   of the first three eligible random pilots.

`example/verification.json` compares these new computations against the exact
COAD task results retained in the paper. All three comparisons should match at
absolute numerical tolerance `1e-10`. The example does not estimate 17-cohort
transfer, and its sampling pool is historically class-balanced. Do not treat its
class proportion as population prevalence.

The additional `tcga_coad_msi_pilot40.npz` contains 40 real records selected with
`RandomState(0)`, including 19 MSI-positive records. It can be passed to the
separate `pathology-pilot` software interface. It is not the sequence of pilots
in the manuscript experiment; those are sampled after each hash-defined split.

## 4. Prepare the example after a Git clone

Feature files are local-only by default. Their upstream terms are documented in
[data.md](data.md); code licensing does not grant data redistribution rights.
If you already have the verified public pool:

```bash
python scripts/prepare_tcga_example.py \
  --pooled-cache /path/to/COAD.npz --out data/example
```

Otherwise, install the H5 reader and download only the 113 required public slide
feature files (approximately 9.08 GB for this example):

```bash
python -m pip install -r requirements-data.txt
python scripts/prepare_tcga_example.py \
  --features-dir data/downloads/tcga --download --out data/example
```

Omit `--download` when the H5 files already exist. The script uses the pinned
source URLs and pools patches within each slide, then slide means equally within
each participant. It does not download WSIs or UNI2-h model weights. Existing
NPZ outputs are not overwritten. Never delete your own files merely to rerun an
example: choose another output path if needed.

## 5. Restore other public tasks

```bash
python scripts/prepare_public_tasks.py --list
python scripts/prepare_public_tasks.py --tasks WSI-COAD WSI-STAD \
  --cache-root /path/to/verified/cache-root --out data/public_prepared
```

For upstream H5 pooling, substitute `--features-root data/downloads/tcga` and add
`--download` only when you intend to retrieve missing files. Use `--tasks all`
for all 85 public tasks. Large feature archives are not included in this folder.
This script checks feature-value hashes. A mismatch is an error requiring review
of upstream revisions or slide selection; it is never silently accepted as exact
reproduction. Public task metadata also specify each binary endpoint and its
preserved evaluated labels. [Source links](data.md) identify the original labels.

## 6. Recompute the complete annotation study

Authors with authorised access can supply the input layout in `docs/data.md`:

```bash
python reproduce.py full --data-root /path/to/authorised/cache-root \
  --workers 4 --out outputs/full_annotation
```

This clears the staged per-task reference/operation/reconstruction outputs, fits
all 134 tasks with the original seeds, refits the diagnostic models, and rebuilds
all derived summaries. The original reference folder is read-only to the runner.
`input_hashes.json`, `logs/`, `status.json` and `comparison.json` document the run.
Files in the output workspace can contain copies of authorised inputs and must
not be published. `outputs/` is Git-ignored.

## 7. Run a supporting experiment

```bash
python reproduce.py list
python -m pip install -r requirements-supporting.txt
python reproduce.py supporting --experiment ki67 \
  --data-root /path/to/authorised/cache-root --out outputs/ki67
python reproduce.py supporting --experiment label-noise \
  --data-root /path/to/authorised/cache-root --out outputs/label_noise
```

The Ki67 example requires the real `ki67_all172.npz` processed archive, which is
not redistributed. The label-noise experiment perturbs **real** grade labels as
reported in Supplementary Figure S3; it does not generate a synthetic image or
feature dataset.

Image/report experiments additionally need the original input paths. When report
extraction is involved, provide `--private-images /path/to/authorised/archive`.
Encoder extraction uses `--hf-cache /path/to/model-cache` with an authorised
Phikon checkpoint. GPU installation and training limitations are described in
[environments.md](environments.md). A public COAD example cannot replace the
hospital images needed for quality, degradation or temporal experiments.

## 8. Troubleshooting and expected differences

- **Output already exists:** use a new `--out`; the runner intentionally refuses
  to overwrite completed work.
- **Missing input:** consult `configs/experiments.json`. Private files are not
  created or replaced with simulated data.
- **Feature checksum mismatch:** verify the original cache or upstream revision.
  Changing dataset membership changes the experiment.
- **GPU training differs:** historical training did not guarantee deterministic
  GPU kernels; one earlier temporal run lacks a complete run record. Compare
  protocols and report new results separately.
- **Missing `owkin/phikon`:** obtain the model through its official model card,
  following the upstream terms, then supply the local cache.
- **Legacy cache needs pickle:** only author-controlled original dictionary NPZ
  caches use `allow_pickle=True`. Do not load an untrusted private cache. Public
  tutorial files and public-task preparation use `allow_pickle=False`.

New scientific outcomes should be reviewed before replacing any reference result.
