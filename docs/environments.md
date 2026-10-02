# Environments and resource requirements

## CPU reference analyses and example

Python 3.12 and `requirements.txt` pin NumPy, pandas, SciPy, scikit-learn and
Matplotlib. This is sufficient for figures, aggregate recomputation, the real
COAD example and full cached-feature annotation/refitting runs. The runner
limits each BLAS worker to one thread and accepts up to four annotation workers.

Install `requirements-data.txt` for public H5 pooling. The local COAD NPZ files
are under 1 MB together; reconstructing them requires about 9.08 GB of source H5
files. The full public manifest refers to much larger data, so downloads must be
selected explicitly. Network access is unnecessary for the CPU examples once
dependencies and the processed public sample are present.

## Supporting feature and image experiments

`requirements-supporting.txt` adds image readers, report readers, model loaders
and H5 support. Some original cached-feature scripts import PyTorch even when
features are already present. Install the appropriate PyTorch build separately.
The observed research environment used:

```text
torch 2.11.0+cu128
torchvision 0.26.0+cu128
transformers 5.17.0
Python 3.12
```

For that CUDA build, use the official PyTorch index on a compatible machine:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-supporting.txt
```

Use the [official PyTorch installer](https://pytorch.org/get-started/locally/) to
select a build appropriate for another platform. Changing software/hardware may
change floating-point results; record the resolved environment with your run.
The CPU annotation protocol does not require a GPU. Photograph extraction,
resolution/degradation experiments and Ki67 extraction use the preserved CUDA
paths; sufficient GPU memory and authorised encoder weights are needed.

## What exactness means here

The portable runner preserves numerical algorithms and seeds. It changes output
locations and source fingerprints. Exact-result checks therefore compare numbers
and scientific decision categories, with an absolute tolerance of `1e-10`, rather
than demanding identical code-hash fields or wall-clock times. Reference JSON
files retain their original hashes in the release manifest.

Full image-to-model reproducibility is more limited: the original SimCLR image
inventory was not retained, and historical GPU training did not enable a complete
determinism guarantee. The earlier pre-existing ImageNet run has only its saved
test summary. These results can be inspected and their tables rebuilt, but a
claim of bit-for-bit retraining would exceed the retained evidence.

No dependency installation, download or GPU training is performed by `verify`.
The delivered `docs/verification.json` distinguishes checks actually executed
from available commands that require additional inputs or hardware.
