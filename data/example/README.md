# Real TCGA-COAD example

The local delivery contains a processed public pool of 112 participants (56 MSI
positive, 1536 features) and a 40-record subset with 19 positives. No artificial
observations are generated. `source_manifest.json` identifies participant order,
labels, H5 files, pinned source URLs, pooling rules and expected hashes.

The 40-record subset uses `RandomState(0)` without outcome-based selection. The
manuscript reproduction command instead samples pilots after its hash-defined
train/test splits, preserving the original experiment.

Feature files are excluded from Git by default because upstream redistribution
rights are not supplied by the repository's code licence. See
[data sources](../../docs/data.md) and [tutorial](../../docs/tutorial.md) for
download and reconstruction instructions. `expected_decision.json` is a result
of the real 40-record subset, not a clinical recommendation.

`expected_experiment.json` records the verified comparisons for the complete
COAD benchmark, single-pilot repetitions and refitting/gain diagnostics.
