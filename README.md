# Pathology experiment reproduction

Code, numerical reference results and a real-data tutorial for reproducing the
experiments in *Task feasibility and annotation decisions in low-resource
computational pathology*. This repository corresponds to the current main text
and supplement; their source hashes are recorded in `configs/experiment_map.json`.

The repository includes **18 experiment groups, all 13 cited figures, all 16
tables, the 134-task annotation benchmark and its source-cohort sensitivity
analyses**. It contains no manuscript, Highlights, submission letters, hospital
images, clinical reports, hospital identifiers, hospital features or model weights.

## Choose a reproduction level

| Level | Command | What actually runs | Data needed |
|---|---|---|---|
| Figures and tables | `figures` | Rebuilds figures and CSV tables from supplied estimates | Included aggregate results |
| Aggregate analyses | `aggregates` | Recomputes policies, cohort bootstrap intervals, grouped meta-models and sensitivity summaries | Included task/repetition estimates |
| Real public example | `example` | Refits the TCGA-COAD task: 30 balanced draws per eligible budget, 30 random/balanced pilots and refitting diagnostics | Included local public feature sample |
| Full annotation experiment | `full` | Refits all 134 tasks and paired-gain/refitting sensitivities, then recomputes summaries | Authorised feature/label caches |
| Supporting experiments | `supporting` | Runs the selected task, quality, noise, temporal or other experiment | Inputs listed for that experiment |

Rebuilding a plot or resampling saved estimates is **not** retraining a model.
The public example is one real task from the study, not a substitute for the
134-task benchmark. [Experiment coverage](docs/experiments.md) identifies the
inputs and reproducibility limits of every reported result.

## Quick start

Use Python 3.12 in a new environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py aggregates --out outputs/aggregate_run
```

Open `outputs/aggregate_run/tables/` for all 16 editable CSV tables,
`figures/` for the 13 figure sets, and `comparison.json` for numerical checks.
The source estimates remain unchanged. Each run requires a new output directory.

Refit the bundled real TCGA-COAD example:

```bash
python scripts/verify_example.py
python reproduce.py example --out outputs/coad_run
```

The local delivery includes **112 TCGA-COAD participants, 56 MSI-positive labels
and 1536 features per participant**. A separate 40-record sample is included for
the single-pilot interface. No generated dataset is used in the tutorial.
The feature files are Git-ignored: after cloning a repository without them,
follow the [download and preparation instructions](docs/tutorial.md).

## Full and supporting experiments

```bash
python reproduce.py list
python reproduce.py full --data-root /path/to/authorised/cache-root \
  --workers 4 --out outputs/full_annotation
python reproduce.py supporting --experiment task-characterisation \
  --data-root /path/to/authorised/cache-root --out outputs/task_spectrum
```

The `full` command recomputes the annotation study and refitting diagnostics.
Supporting experiments have separate commands, because several require raw
images, reports or GPU training. They are not silently marked complete by the
`full` command. Install their optional dependencies as described in
[environments](docs/environments.md). Large data and model downloads are explicit,
never part of quick start.

## Repository layout

```text
analysis/       Computation and plotting code, preserving experiment-stage paths
configs/        Experiment commands, input requirements and figure/table mapping
reference/      Aggregate estimates and repetition-level results without record IDs
data/example/   Real public tutorial data, provenance and expected outputs
data/           Public-task membership and download manifests
docs/           English tutorial, methods, data sources and verification report
scripts/        Data preparation, table export, result comparison and integrity checks
tests/          Scientific and release regression tests
reproduce.py    Entry point; uses a fresh output workspace for every run
```

The stage names under `analysis/` preserve imports and provenance; they are not
different software installations. Only scientific scripts are retained.
Chinese strings needed to interpret the original Chinese-language report schema
remain in extraction code; the documentation and new interfaces are in English.

## Data, licensing and interpretation

The 85 public tasks have explicit participant/label manifests and large-data
links. Institutional experiments require authorised access; there is no public
download link for hospital inputs. The Ki67 archive has unresolved upstream
permissions and is linked, not redistributed. See [data sources](docs/data.md).

The original source code and accompanying software documentation are licensed
under the [MIT License](LICENSE). You may use, modify and redistribute the code,
including commercially, subject to the license's copyright and notice requirements.

Third-party code, datasets, derived feature arrays, pretrained models and weights
remain subject to their respective licenses and terms of use. The MIT License
does not grant additional rights to redistribute these materials. See
[licensing details](LICENSING.md). This local delivery has not been uploaded to GitHub.

The historical stopping rule is a research comparator. Its reported 47.8%
false-stop rate is not evidence of a reliable stopping recommendation. The
balanced benchmark is descriptive; conditional resampling intervals do not
represent uncertainty in future annotation gains.

See [tutorial](docs/tutorial.md), [experiment map](docs/experiments.md),
[methods](docs/methods.md), and [verification](docs/verification.json).

## Citation

If you use this software in your research, please cite the accompanying paper:

> Wan Sijie and Teng Da. *Task feasibility and annotation decisions in low-resource
> computational pathology*. Manuscript, 2026.

This is a citation request, not an additional condition of the MIT License.

## Revision verification

This delivery was checked on 29 September 2026. See [revision verification](docs/revision_verification.json) for the executed checks and their scope. Earlier validation records are retained separately.

The license and README were updated on 2 October 2026; see
[licensing update](docs/licensing_update.json). Earlier verification records
describe the licensing status at the time of those checks.
