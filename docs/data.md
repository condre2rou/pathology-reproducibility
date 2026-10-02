# Data sources and input layout

## Included and excluded material

`reference/` contains cohort/task/repetition summaries, not individual hospital
records. `data/example/` contains a small, processed **public TCGA-COAD** dataset in
the local delivery. These real feature files are Git-ignored until redistribution
terms have been resolved. No hospital image, report, feature vector, record ID or
individual prediction is included.

Public participant IDs in `data/public_tasks.json` are TCGA research identifiers.
They are required to reproduce participant grouping and hash-defined splits.
They do not identify people by name and must not be used for re-identification.

## Upstream downloads

| Resource | Official source | Use and local treatment |
|---|---|---|
| TCGA UNI2-h patch features | [W8Yi feature dataset](https://huggingface.co/datasets/W8Yi/tcga-wsi-uni2h-features) | Large H5 files; download only selected slides. No H5 files or WSIs bundled. |
| TCGA molecular/clinical annotations | [cBioPortal Datahub](https://github.com/cBioPortal/datahub), [API](https://www.cbioportal.org/api/swagger-ui/index.html) | Original study/attribute mappings are in analysis scripts; evaluated public binary labels are frozen in the task manifest. |
| TCGA original data | [Genomic Data Commons](https://portal.gdc.cancer.gov/), [data policies](https://gdc.cancer.gov/access-data/data-access-policies) | Original slides are unnecessary for the cached-feature experiments. |
| UNI2-h | [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) | Upstream encoder and terms; its weights are not needed for H5 pooling. |
| Phikon | [owkin/phikon](https://huggingface.co/owkin/phikon) | Needed to re-extract photograph/Ki67 features. Observed local revision: `057cc0295895c2df3dd7681a89680da6015cbefe`. |
| Paired H&E/IHC archive | [Anonymous192234/KI67](https://huggingface.co/datasets/Anonymous192234/KI67) | `trainA.zip` and `trainB.zip`, about 27 GB locally. Linked only: original permissions and clinical provenance remain unresolved. |
| ImageNet ResNet-50 | [Torchvision model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html) | The original code uses `IMAGENET1K_V1`; do not substitute `DEFAULT`/V2. |

The TCGA feature revision observed in the study is
`35fd1739cc2ad3b2f35396d26b127d791aaec1a4`. The manifest provides pinned resolve
URLs, slide paths, available byte counts, participant order, labels and hashes of
the evaluated feature values. Some original downloads used mutable URLs; a
revision-directory name alone does not authenticate every historical file.
Preparation scripts therefore verify the **pooled feature values**, and stop if
they differ. The current upstream extraction description does not establish the
exact parameters of every historical H5 file.

The Ki67 repository API was reachable during packaging and exposed two archives
at revision `5d8a5ec0a389a729e68885be1f011a35e546b7bf`, with no licence field.
This is an observed current revision, not an authenticated historical analysis
revision. Public accessibility does not establish permission to redistribute the
images. No Ki67 images or feature vectors are included.

## Public task coverage

The benchmark uses 15 TCGA source groups: BRCA, COAD, ESCA, HNSC, KIRC, LGG, LIHC,
LUAD, LUSC, PAAD, PCPG, PRAD, STAD, UCEC and UVM. These contribute 85
task–representation combinations. `data/public_tasks.json` records each endpoint,
the exact evaluated label vector, sample/positive counts and feature dimension.

The task builders retain the original rules: mutation negatives are restricted
to sequenced participants; genomic quantities use within-cohort median splits;
TCGA identifiers are reduced to their 12-character participant prefix. Mean
patch embeddings are first pooled within a slide; slide means receive equal
weight within a participant. One participant contributes one feature row before
folds or test sets are assigned.

`scripts/prepare_public_tasks.py` can restore all 85 tasks from trusted original
public caches, or reconstruct selected tasks from upstream H5s. The retained
slide inventory is not a guarantee of original selection for every task: an H5
reconstruction must match the saved feature-value hash to count as exact. The
small COAD tutorial has additionally been verified by reconstruction from its
113 local H5 files.

The public manifests retain third-party research annotations and identifiers.
They are not relicensed as software. cBioPortal Datahub states its own database
licensing terms; TCGA and the feature encoder have separate terms. See their
linked source pages before redistribution.

## Authorised input layout

`--data-root` points to a directory containing the following relative paths.
It may be the existing author-controlled cache root; no relocation of the source
files is required. The runner copies only specified inputs to the isolated run
workspace. It does not add them to the repository.

```text
e1_label_engine/outputs/
  labels_ec.csv                 Label fields: year, folder, grade, ki67, p53, mmr, er, pr
  labels_ec_patient.csv         Legacy schema; its local IDs are directory records
  image_index.csv               Report alignment; only relevant to authorised audits
  llm_extraction.csv            Saved extractor outputs, if checking agreement
e2_data/
  manifest.csv                 Record ID, image paths, year and task labels
  splits.json                  Temporal record partitions
e2_train/runs/
  feats_phikon.npz              Dictionary cache: feats[record_id] -> vector
  feats_ssl.npz
  feats_imagenet.npz
  simclr_backbone.pt            Only needed for image-level recomputation
e6_learnability/
  res_feats_ssl.npz             Resolution -> record-feature dictionary
e5_tcga/
  pooled_features.npz           UCEC X/Y/pids
  pooled/{COAD,STAD,BRCA_IDC,BRCA_IDC_PR}.npz
  multi_subset.csv
  tcga_confound.csv
  tcga_extra_labels.csv
e10_expansion/
  pooled/*.npz                 Expanded tasks; X/Y/pids
  b_upside_data_v2.csv          Frozen ordered list of 134 combinations
  ki67_all172.npz              Supporting archive cache, if authorised
```

The exact minimum inputs for each supporting experiment are listed in
`configs/experiments.json`. Full annotation recomputation needs the label and
feature caches, not hospital raw images. Photograph quality, degradation,
pretraining and temporal training additionally need the authorised source
images. Subtype/LVSI/myoinvasion extraction needs the source report tables or a
separately validated derived-label adaptation.

Some private manifests contain absolute image paths. On another authorised
machine, map these to the same images before running; do not publish the mapping.
`--private-images` configures the report archive root. Its expected year-specific
schema is preserved in `analysis/e1_label_engine/align_seq_key.py`. Never interpret
a directory record as a verified unique person across years.

Only load legacy pickle-based feature dictionaries from trusted author-controlled
files. Public preparation uses plain, non-pickled arrays with keys `X`, `Y`/`y`
and `pids`/`ids`. Features must be finite, labels binary for classification, and
participant/record rows unique at the final analysis unit.

## What cannot be recovered from links alone

- Hospital access is controlled by the institution; there is no public download.
- The exact historical SimCLR pretraining image inventory was not retained.
- The Ki67 filename identifiers do not establish person/specimen linkage or
  clinical Ki67 ground truth.
- A current cBioPortal download can differ from the frozen evaluated annotation
  vectors. Use the manifest and hashes to identify the study's actual inputs.
- The legacy `wsi-tasks` supporting entry point still retrieves the `GRADE`
  attribute from the cBioPortal API. Its outputs matched the saved estimates in
  the packaging check, but that endpoint is mutable. The 85-task annotation
  benchmark and real COAD tutorial use frozen evaluated labels and do not make
  this request. A future supporting mismatch must be investigated, not accepted
  as the historical result.
- An earlier ImageNet temporal run lacks a complete logged training history.

These constraints are represented in the experiment map rather than hidden by
substituting unrelated public data or simulated observations.
