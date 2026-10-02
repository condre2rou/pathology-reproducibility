# Computational methods

## Units and model fitting

Institutional analysis units are directory records. TCGA units are participants,
with all slides pooled before splitting. The 134 combinations share 17 source
cohorts: 48 institutional EC combinations, one institutional OV combination and
85 public TCGA combinations. A source-cohort sensitivity merges both institutional
cohorts because longitudinal person linkage is unavailable.

The annotation probe is training-fold `StandardScaler` followed by balanced
logistic regression (`C=0.1`, `max_iter=3000`). OOF AUROC uses pooled predictions.
Fold count is at most five, bounded by minority-class size. Supporting task
characterisation retains its separate fold-mean AUROC protocol and original
solver limits. These estimands must not be interchanged.

## Finite-pool benchmark

Budgets are 10, 20, 30, 40, 50, 60 and 80. A budget needs at least twice as many
records and half the budget in each class. Each uses 30 balanced draws with the
original `RandomState` seeds. Complete-pool estimates average five partitions.
Headroom is `A(full) - A(40)`; a value at most 0.05 defines finite-pool saturation.
The descriptive score is `-A(40)`. Its algebraic overlap with the outcome makes
this a descriptive benchmark, not independent prospective prediction.

The historical cutoff is 0.57, selected during earlier overlapping-task work.
Recalibration holds out an entire source cohort and searches 0.400–0.955 in steps
of 0.005, choosing the first maximum of accuracy. Grouped meta-models use a
separate balanced logistic fit with `C=1` and cutoff 0.5, fitted only in the outer
training source groups. The earlier leave-one-combination estimate remains in
the original numerical record but is not the active manuscript's meta-model.

## Single-pilot experiment

For seed 0–29, the first eight SHA-256 bytes of `cohort|analysis_id|seed`, divided
by `2**64`, assign values below 0.25 to testing. Splits are label-independent and
shared by all configurations/endpoints in a cohort. Eligible repetitions need
at least 60 pool records, 10 test records and two per class in both partitions.

Natural sampling draws 40 records using seed `10000 + seed`; balanced sampling
separately draws 20 per class when available. Pilot CV uses the repetition seed.
A 20-record subset uses `30000 + seed`. Pilot OOF label/prediction pairs are
resampled 200 times with `20000 + seed`, keeping fitted models fixed. The
2.5th/97.5th percentiles form a conditional pair-resampling interval.

Pilot and full-pool probes are fitted separately and evaluated on the same test
records. The expanded pool includes the pilot. Gain `G` is expanded minus pilot
test AUROC; `G <= 0.05` is the small-gain outcome. Test and expanded-pool labels
never enter a target pilot's decision. Invalid pilots remain in evaluation as
insufficient evidence; they are not silently deleted.

Policies are always continue, always stop, historical 0.57, held-out-cohort
recalibration, a flat 20-to-40 curve (difference at most 0.02), and the three-way
conditional-interval rule. The latter stops when the upper bound is at most
0.57, continues when the lower bound exceeds 0.57, and otherwise abstains.

## Metrics and intervals

For each retained task with `R` eligible repetitions, each repetition has weight
`1/R`. Thus every task contributes total weight one. Insufficient evidence is
treated as continue for accuracy and cost, and excluded from definitive coverage.

| Measure | Numerator | Denominator |
|---|---|---|
| Precision | Weighted correct stops | Weighted stop calls |
| False-stop rate | Weighted stops with `G > 0.05` | Weighted stop calls |
| Sensitivity | Weighted correct stops | Weighted small-gain repetitions |
| Specificity | Weighted non-stops with `G > 0.05` | Weighted larger-gain repetitions |
| Missed opportunity | Weighted false stops | Weighted larger-gain repetitions |
| Accuracy | Weighted correct binary decisions | All repetition weights |
| Coverage | Weighted explicit stop/continue calls | All repetition weights |
| Annotation savings | Weighted stops times `N_pool - 40` | Weighted `N_pool` |

No-stop precision and false-stop rate are undefined. Evaluator-only test labels
are excluded from cost. Sensitivity subsets recompute each task's weights.

Cohort intervals use 1000 percentile bootstrap samples with seed 20260923,
carrying all tasks and repetitions of each sampled cohort together. Fitted
decisions remain fixed. The matched refitting diagnostic uses the first three
eligible natural pilots per combination and 100 within-class record resamples
with seed `70000 + seed`. Duplicate copies of a record stay together in a fold.
At least 50 valid resamples are required for percentile sensitivity bounds.
Paired test-gain intervals use 200 stratified resamples with seed `60000 + seed`.

## Supporting analyses

Supporting scripts preserve the original data order, task definitions, solver
settings, sample sizes and seeds. Quality tertiles and fold standard deviations
are descriptive. The noise-weighting experiment changes training labels in real
grade data; its simulated flag uses sensitivity 0.70, false-positive probability
0.30, and weights 0.2 versus 1.0. It is not a learned noise detector.

Historical nearest-neighbour/curve comparisons in Table S2 use their original
protocols, including legacy preprocessing choices; they are not corrected
annotation-policy validation results. The Ki67 target is an image-derived DAB
fraction. The conventional Spearman p-value retained in its legacy JSON is not
reported as fitting-aware significance in the current manuscript.

Temporal model evaluation expands each test directory into its image rows.
Training/test splits separate directory records; stored `n`/`pos` fields for
these runs count image rows. The table exporter labels those counts explicitly
and adds separate counts of directory records with usable images from the
retained manifest audit. AUROC/AUPRC are descriptive image-level point estimates;
no image-iid confidence interval is reported. A record-level AUROC cannot be
recovered from these aggregate run summaries. The historical code does not
establish person-level independence.

Table S1 reports mean AUROC and descriptive SD over the 50 overlapping fold
scores (ten seeds, five folds). Resampling those correlated scores does not
establish a confidence interval for generalisation performance. Legacy `ci95`
fields are preserved only in original reference results and excluded from the
revised table.
