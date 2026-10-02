# Experiment coverage

This map covers the current main text and supplement identified by the source
hashes in `configs/experiment_map.json`. No manuscript text or submission
materials are part of this repository. The map distinguishes **rebuilding a
reported estimate**, **refitting its predictive model**, and **recreating its
image representation**. They require different inputs.

## Commands and outputs

Run commands from the repository root with the documented environment. Output
directories must be new; the runner will not overwrite an earlier run.

```bash
# No feature data needed: recompute summaries, uncertainty and plots.
python reproduce.py aggregates --out outputs/aggregate_run

# Refit one complete, real public task, including its sensitivity diagnostic.
python reproduce.py example --out outputs/coad_run

# Refit every annotation task from authorised, already-extracted features.
python reproduce.py full --data-root /path/to/cache-root \
  --workers 4 --out outputs/full_annotation

# Refit a supporting experiment; choose an ID from the table below.
python reproduce.py supporting --experiment task-characterisation \
  --data-root /path/to/cache-root --out outputs/task_characterisation

# For report-derived endpoints, supply the authorised report archive root.
python reproduce.py supporting --experiment histology \
  --data-root /path/to/cache-root \
  --private-images /path/to/authorised/archive --out outputs/subtype
```

Every run writes `status.json`, per-command logs and a numerical comparison.
`figures`, `aggregates` and `full` also write the 16 CSV tables and 13 figure
sets. `example` writes its own fitted-task outputs and verification.
`supporting` compares only the selected experiment's newly computed files; it
does not claim to rerun other estimates staged in its workspace.

The work directory of a run using authorised inputs contains copies of those
inputs. Keep it private. The delivery folder and tracked repository contain no
hospital inputs. No command uploads data.

## Experiment groups

`configs/experiments.json` lists the exact relative input paths and executable
commands. Use `python reproduce.py list` to inspect the same catalogue.

| Experiment ID | Computation and primary output | Model-input requirement | Reported results |
|---|---|---|---|
| `annotation` | `e11_method_audit` finite-pool and single-pilot fitting; `e13_review` grouped meta-models, paired test-gain intervals and record-resampling/refitting. Use `full`, not `supporting`. | 134 ordered feature/label tasks: 49 institutional and 85 public; no raw images needed | Figures 3–5, S6–S7; Tables 1–3, S5, S7, S9, S11–S13 |
| `task-characterisation` | `eval_phikon.py` and `robustness.py`; frozen representation probes with the original repeated fold-mean protocol | Authorised Phikon/SimCLR/ImageNet caches and labels | Figure 2; Tables S1 control, S8 |
| `histology` | `b_histology.py`; serous versus endometrioid classification | Authorised features and original report tables | Figure 2; Table S8 subtype row |
| `additional-tasks` | `validate_scale.py`; LVSI and myoinvasive depth | Authorised features and original report tables | Table S1 |
| `separability` | `validate.py`; nearest-neighbour/separation correlations across 17 configurations | Institutional and public feature caches | Table S2 |
| `active-selection` | `active_select.py`; random, uncertainty and diversity subset sampling | Same mixed caches | Table S2 random-subset row; supporting comparisons |
| `curve-extrapolation` | `learning_curve.py`; asymptote fits to four sample budgets | Same mixed caches | Table S2 |
| `quality` | `quality_strat.py`; measured native-image quality tertiles and repeated probes | Authorised photographs, paths, labels and Phikon features | Table S4 |
| `label-noise` | `e13_review/supporting.py noise`; repeated injected label flips and simulated noise weighting on the real grade task | Authorised grade features and labels | Figure S3 |
| `ki67` | `eval_ki67_172.py`; leave-one-out ridge regression and exploratory binary probes | External 172-identifier Phikon/DAB cache | Supporting Ki67 results |
| `wsi-confounding` | `confound_check.py`; grade-stratified TCGA-UCEC analysis | Public pooled UCEC features and frozen confounder table | Figure S2 |
| `wsi-tasks` | `ucec_wsi_tasks.py`, `train_multi.py`; WSI grade/dMMR, MSI, ER and PR probes | Public pooled features, cohort subset table and cBioPortal grade labels | Figures S4–S5 |
| `cross-cohort` | `expand_pairs.py`; matched endpoint definitions across unpaired cohorts | Authorised reports/features plus TCGA features/extra labels | Table S6 |
| `degradation` | `degradation_study.py`; downsampling, blur, JPEG and illumination perturbations | Authorised 599 photographs, Phikon weights and GPU | Figure 6 |
| `resolution` | `resolution_ladder.py`; seven input-resolution settings | Authorised photographs, trained SimCLR weights and GPU | Figure S1; resolution features feed annotation tasks |
| `temporal-training` | `portable/temporal_runs.py`; full ImageNet/SimCLR and frozen-head runs | Authorised photographs, temporal splits, initial weights and GPU | Table S3 (three rerunnable configurations) |
| `demographics` | `demographics.py`; report availability and aggregate demographic summaries | Authorised reports and alignment tables | Cohort descriptions |
| `extractor-agreement` | `make_e1_summary.py`; agreement of existing extraction outputs | Authorised rule/LLM outputs and label audit summaries | Label availability and extractor-agreement results |

The last group checks saved extraction outputs. It does not rerun an external
language-model service or establish clinical label accuracy. No API credentials
or hospital report text are distributed.

## Figure-to-source map

The figure builder reads the numerical files below. PDF, PNG and SVG outputs are
renamed to `Figure_1`, …, `Figure_6`, `Figure_S1`, …, `Figure_S7`.

| Figure | Source estimate or diagram | Drawing code |
|---|---|---|
| 1 | Workflow schematic; no fitted numerical result | `e12_editorial/figures.py` |
| 2 | `e2_train/runs/tasks_phikon.json`, `robustness_ssl.json`, `robustness_imagenet.json`, `e6_learnability/histology_robust_result.json` | `e12_editorial/figures.py` |
| 3 | `e11_method_audit/outputs/corrected_reference_result.json` | `e11_method_audit/figures.py` |
| 4 | Same reference summary: budget-specific and common-subset estimates | `e12_editorial/figures.py` |
| 5 | `e11_method_audit/outputs/single_budget_result.json` | `e12_editorial/figures.py` |
| 6 | `e10_expansion/degradation_result.json` | `figures/make_figures.py` |
| S1 | `e2_train/runs/resolution_ladder_ssl.json` | `figures/make_figures.py` |
| S2 | `e5_tcga/confound_result.json` | `figures/make_figures.py` |
| S3 | `e13_review/outputs/synthetic_noise.json` | `e13_review/figures.py` |
| S4 | Institutional Phikon and `e5_tcga/ucec_wsi_tasks.json` | `figures/make_figures.py` |
| S5 | Institutional Phikon, `e5_tcga/ucec_wsi_tasks.json`, `e5_tcga/multi_result.json` | `figures/make_figures.py` |
| S6 | Cohort-level finite-pool and natural-pilot summaries | `e12_editorial/figures.py` |
| S7 | Per-task natural-pilot decisions and scores | `e12_editorial/figures.py` |

### Figure S5 correction

The original drawing code associated the two MSI values with the wrong cohort
names. The stored results are **COAD 0.919** and **STAD 0.905**. This release reads
those values directly from `multi_result.json`, fixing the swapped labels. The
underlying model estimates are unchanged. The revised manuscript, its independent
figure files and this software now use the same corrected rendering.

## Table-to-source map

`scripts/export_tables.py` derives each editable CSV directly from numerical
results. It keeps available precision; the manuscript normally rounds to three
decimal places or one percentage decimal. Several CSVs use one row per
condition/representation rather than the manuscript's compact layout.

| Table | Content | Numerical source under `reference/` |
|---|---|---|
| 1 | Source counts, combinations and roles | E1 label summary; E11 task manifest; Ki67 metadata |
| 2 | Descriptive benchmark and grouped validation | E11 corrected reference; E13 grouped meta-model |
| 3 | Six natural-pilot policies | E11 single-budget summary, `natural` |
| S1 | LVSI, myoinvasion and grade control; mean fold AUROC and descriptive fold SD | E6 `validate_scale_result.json`; E2 task probes |
| S2 | Three historical alternative summaries | E6 validation, active-selection and curve results |
| S3 | Image-level temporal AUROC/AUPRC and separate record/image counts | E9 `finetune_*.json`, `temporal_counts.json` |
| S4 | Quality tertiles with sample counts and fold variability | E9 `quality_strat_result.json` |
| S5 | Budget-specific/common-subset results | E11 corrected reference |
| S6 | Task-matched unpaired comparisons | E6 `pairing_table.json` |
| S7 | Six balanced-pilot policies | E11 single-budget summary, `balanced` |
| S8 | Seven endpoints under three representations | E2 task probes and E6 histology result |
| S9 | Finite-pool headroom cutoff sensitivity | E11 corrected reference |
| S10 | Decision accounting for a 100-record pool | Formula example; not an experimental dataset |
| S11 | Test-size and gain-margin sensitivities | E13 `sensitivity.json`, all policy rows exported |
| S12 | Matched pair-resampling versus refitting diagnostic | E13 `sensitivity.json` |
| S13 | Delete-one-cohort ranges | E13 `sensitivity.json` |

Additional exports include `all_task_definitions.csv`, `degradation.csv` and
`key_results.json`. The latter collects abstract/text quantities, meta-model
results, paired-gain uncertainty and supporting Ki67/label/demographic results.
It is not a substitute for their individual fitting scripts.

Table S1 exports descriptive variability across overlapping cross-validation
folds, not confidence intervals. Historical `ci95` fields remain in the original
reference JSON for provenance but are excluded from the revised table export.
Table S3 reports image-level point estimates without an image-iid confidence
interval. Its aggregate count audit uses the retained manifest and current file
availability, not an authenticated run-specific list. Authorised users can rerun
`analysis/portable/temporal_counts.py --manifest /private/manifest.csv
--splits /private/splits.json --out counts.json`; only aggregate counts and input
hashes are written. No clinical inputs are supplied with this repository.

## Features and upstream preparation

The main `full` command starts from frozen feature caches; it does not retrain
the encoders. Public cache restoration and H5 pooling are covered in the
[tutorial](tutorial.md) and [data guide](data.md). Supporting upstream code is
also included:

- `e1_label_engine/align_seq_key.py`, `extract_rules.py` and `freeze_labels.py`
  preserve report alignment and rule-derived label construction.
- `e2_data/make_splits.py` preserves the temporal directory-record split.
- `e2_train/ssl_pretrain.py`, `eval_tasks.py`, `eval_phikon.py` and
  `e6_learnability/extract_res_features.py` contain encoder training/extraction.
- `e5_tcga/prepare_tcga.py`, `prepare_multi.py`, `fetch_labels.py`,
  `fetch_clinical.py` and `fetch_extra_labels.py` retain original public-source
  preparation; the new public manifests freeze the evaluated benchmark labels.
- `e10_expansion/build_cohort_tasks.py` and `build_ovarian_points.py` preserve
  the expanded task construction rules.
- `e10_expansion/expand_ki67.py` constructs the external feature/DAB cache if
  the archive and model are available under appropriate permissions.

`e6_learnability/calibrate.py` contains the original nearest-neighbour and Fisher
helpers imported by the learning-curve script. Unused historical calibration
experiments are not exposed as manuscript results.

These author-stage scripts operate on the staged layout and original schemas.
They are not a general clinical-data importer. Read each script's arguments and
the data guide before use. They may need GPUs, large downloads or image paths
that are not present in a public checkout.

## Verification boundaries

The machine-readable [verification record](verification.json) names checks
actually executed. A complete annotation rerun refits 134 finite-pool tasks,
their natural/balanced pilot repetitions and all 134 refitting-diagnostic task
files. Supporting results present in the same workspace but not selected for
recomputation remain reference inputs; a matching comparison alone does not
mean they were retrained.

No raw-image GPU pipeline was retrained during packaging. In particular:

- The original SimCLR image inventory was not retained, so the original
  pretraining exposure cannot be reconstructed exactly from a current manifest.
- Historical CUDA training has no complete determinism guarantee. Table S3's
  earlier `imagenet_full_PREEXISTING` run has its saved test summary but lacks a
  complete recoverable training history. Its row can be rebuilt, not promised
  to be exactly retrained.
- The temporal evaluator operates on image rows after a directory-record
  train/test split. Exported counts explicitly say `n_image_rows`; they are not
  counts of verified unique persons.
- The paired Ki67 cache supports the reported computation; it does not resolve
  upstream person/specimen linkage or clinical truth.
- Legacy Table S2 methods and their preprocessing are retained as descriptive
  historical analyses. Repackaging does not convert them into independent
  validation estimates.

See [methods](methods.md) for definitions, seeds, units and interval scope.
