# Why `linear_close` Has Worse MAE Than `linear`

## Investigation status

This is the chronological investigation record requested for the MAE gap between
`interpolation_similarity=linear` and `linear_close` in
`cross_space_projection.py`. It is deliberately separate from
`docs/cross_space_projection_formulations.md`, which derives the projector formulas
but does not investigate the observed experiments.

- Started: 2026-08-19
- Production-pipeline changes made by this investigation: **none**
- Diagnostic code: `analysis/linear_close_investigation.py`
- Diagnostic tests: `tests/test_linear_close_investigation.py`
- Current phase: **complete**; final causal explanation and recommendation recorded below

## Question and success criterion

The question is why the two supplied `linear_close` experiments report much worse
headline MAE than their `linear` counterparts. Completion requires both:

1. proving whether the supplied result pairs are genuinely controlled comparisons;
2. isolating the projector behavior under known affine mappings and then under the
   rank, dimensionality, noise, scaling, and refinement conditions of the saved runs.

No change to the main experiment pipeline is justified until the evidence distinguishes
an implementation bug from a statistical/numerical limitation or a configuration
effect.

## Evidence sources and safety

The authoritative sources inspected are:

- current `cross_space_projection.py` implementation and its call path;
- the four supplied aggregate result pickles and aggregate `config_logging.txt` files;
- the launch YAML files named by those logs;
- plain-text per-subtrial `config_logging.txt`, `summary.csv`,
  `refinement_summary.csv`, and reconstruction CSVs;
- projector/refinement checkpoints loaded with PyTorch's `weights_only=True` mode;
- anchor and split CSV files and their byte hashes;
- controlled NumPy/PyTorch diagnostics in the analysis module.

The aggregate pickles contain only standard aggregate values and were read to establish
the reported MAEs and subtrial paths. A normal load of a subtrial pickle attempted to
import project classes whose module initializes multiprocessing state. The sandbox
blocked the resulting socket creation. Because general pickle loading can execute code,
the investigation does **not** bypass that protection; equivalent diagnostics are taken
from safe CSV, safetensors, and weights-only checkpoint artifacts instead.

## Implementation analysis

### `linear`: learned affine map

For anchor matrices \(A\in\mathbb{R}^{K\times D_{old}}\) and
\(B\in\mathbb{R}^{K\times D_{new}}\), `linear` constructs
`torch.nn.Linear(D_old, D_new)`, so its prediction is

\[
\widehat B = A W^T + b.
\]

With the supplied YAMLs it minimizes minibatch mean squared embedding error with AdamW:

\[
\min_{W,b}\frac{1}{K D_{new}}\|A W^T + \mathbf{1}b^T-B\|_F^2,
\]

using learning rate \(10^{-5}\), batch size 64, zero explicit weight decay, 750 epochs,
and no feature normalization. All 100 anchors are training data. A subject-disjoint
half of the target model's `val.csv` selects the best epoch by validation MSE; the
other half is projector test data. Thus finite-step optimization and validation
checkpoint selection are part of the estimator even though no explicit ridge penalty
is configured.

### `linear_close`: truncated-SVD affine least squares

`linear_close` centers the same anchor matrices,

\[
A_c=A-\mathbf{1}\bar a^T,\qquad B_c=B-\mathbf{1}\bar b^T,
\]

and calls `np.linalg.lstsq(A_c, B_c, rcond=1e-5)` in float64. If \(C\) is the
returned coefficient matrix, it materializes exactly the affine map

\[
\widehat B=(A-\bar a)C+\bar b=AC+(\bar b-\bar aC).
\]

The corresponding `nn.Linear` parameters are `weight=C.T` and
`bias=mean(B)-mean(A)@C`. Singular values below \(10^{-5}\) times the largest are
discarded. This is unregularized least squares on the retained singular subspace;
the cutoff is truncation, not ridge shrinkage. There is no optimizer, epoch, or
validation-based model selection.

### Exact similarities and differences

| Property | `linear` | `linear_close` | Consequence |
|---|---|---|---|
| Function family | General affine | General affine | Same representable maps |
| Nominal anchor loss | MSE | MSE / Frobenius least squares | Same nominal data-fit objective |
| Intercept | `nn.Linear.bias` learned | Explicit centering + reconstructed bias | Both include an intercept |
| Explicit regularization | Weight decay from config; zero here | None; only SVD truncation | Equal in these YAMLs only in the narrow explicit-penalty sense |
| Implicit regularization | Initialization, finite AdamW steps, minibatches, validation epoch selection | Minimum-norm retained-subspace solution | Different estimator even with the same nominal loss |
| Normalization | Optional train-anchor coordinate standardization | Same shared preprocessing | Disabled in all four supplied runs |
| Anchor treatment | All anchors train; validation is separate | All anchors solve; validation only evaluated | Same fit rows, different selection use |
| Dimensions | `384 -> 512` in inspected checkpoints | `384 -> 512` | Same dimensions |
| Solver | float32 AdamW, 750 epochs | float64 SVD least squares, then float32 checkpoint | Different spectral filtering |
| Rank limit from 100 anchors | Weight updates live in the rank-at-most-100 raw anchor-row span; initialization remains elsewhere | Centered coefficient rank at most 99 | Severely underdetermined in both; exact OLS interpolates the anchors |
| Refinement | Same two possible modes | Same two possible modes | Supplied refinement hyperparameters differ, but refinement=3 does not replace headline predictions |

The two methods are therefore mathematically equivalent only as an ideal statement:
if learned optimization converged to the same selected OLS solution in a uniquely
determined problem. They are not equivalent estimators in the actual underdetermined,
finite-step, validation-selected regime.

## Supplied experiment comparison

### Aggregate headline results

| Source -> target | `linear_close` MAE | `linear` MAE | Absolute gap |
|---|---:|---:|---:|
| MIntPAIN VideoMAE -> BioVid DFER | 5.952663 | 1.527850 | +4.424814 |
| UNBC VideoMAE -> BioVid DFER | 5.826219 | 0.978494 | +4.847724 |

The MIntPAIN aggregates pool 25 matched model pairs and 15,610 predictions; the UNBC
aggregates pool 25 matched model pairs and 1,000 predictions. Both use seed 42.

### Configuration audit

The aggregate logs and YAMLs show matching:

- the same five source checkpoints and same five target checkpoints in the same
  Cartesian-product order;
- 25 subtrials, seed 42, `num_anchors=100`, `balance_class_random`, anchors from
  target `train`, source evaluation on `test`;
- `weighting_method=none`, loss MSE, batch size 64, 750 projector epochs,
  `normalize_embeddings=false`, and `refinement=3`;
- the same fold/model pairing and deterministic anchor-selection mechanism; the later
  all-subtrial audit byte-matched every corresponding anchor and split file.

The byte matches establish row-manifest identity, not serialized embedding-tensor
identity. Both runs reference the same feature paths, checkpoints, preprocessing, and
deterministic rows, but the per-subtrial safe artifacts do not contain copies of the raw
anchor feature matrices. Tensor equality is therefore strongly implied by the controlled
extraction inputs, not independently hash-verified.

They do **not** match in refinement recipe:

| Parameter | `linear_close` YAML | `linear` YAML |
|---|---:|---:|
| refinement projector LR | `1e-5` | `1e-4` |
| refinement head LR | `1e-4` | `1e-4` |
| `lambda_B` | `0.1` | `1e-4` |
| `lambda_A` | `0.1` | `1e-3` |

This invalidates direct comparison of the saved refinement outcomes. It does **not**
explain the supplied aggregate headline gap: code lines 4484-4537 run both
`linear_only` and `projector_linear` independently when `refinement=3`, and explicitly
keep the headline predictions as the pure pre-refinement projection because there is no
single refined model to select. Per-subtrial `summary.csv` confirms both refinement rows
repeat the same headline projection MAE.

### First matched-subtrial diagnostic (new fold 0, old fold 0; MIntPAIN)

The anchor and projector split CSVs have equal row counts and matching rows. Safe
checkpoint and CSV evidence gives:

| Diagnostic | `linear_close` | `linear` |
|---|---:|---:|
| Anchor count / input / output dimension | `100 / 384 / 512` | `100 / 384 / 512` |
| Projector weight Frobenius norm | 135.6606 | 13.1291 |
| Projector bias norm | 153.4123 | 0.6817 |
| Anchor MSE before joint refinement | `4.08e-12` | 0.061704 |
| Headline source-test MAE | 4.222785 | 1.449255 |
| Source-domain oracle reconstruction, first sample L2 | 44.78 | 7.16 |
| Source-domain oracle reconstruction, first sample cosine | 0.757 | 0.952 |

Interpretation: the closed form is correctly solving its anchor objective—it nearly
interpolates all anchor targets—but that interpolation generalizes very poorly. The
10.3x weight-norm and 225x bias-norm differences are the signature expected when an
underconstrained fit uses unstable anchor directions. Learned `linear` deliberately or
implicitly accepts higher anchor error and produces a far smoother mapping.

## Initial hypothesis ledger

Each entry records the smallest discriminating test and the classification at the point
where the hypothesis was formulated. These preliminary states are intentionally preserved
to show the reasoning path; the completed classifications are in **Final hypothesis
disposition**.

### H1 — The methods solve different function families or nominal objectives

- **Motivation:** A misleading name could hide a different mapping.
- **Diagnostic:** Derive both forward maps and losses from code; test a sufficiently
  determined noiseless affine mapping.
- **Preliminary result:** Both represent `x @ W.T + b` and use anchor embedding MSE in the
  supplied configs. The controlled recovery test passes for closed-form OLS; learned
  convergence comparison is pending at this stage.
- **Status:** **partially rejected**. Function family and nominal loss match; the actual
  estimators differ through optimizer dynamics, rank truncation, and validation selection.
- **Next:** demonstrate equivalence in the ideal determined case and divergence as
  underdetermination/noise are introduced.

### H2 — `linear_close` omits or mishandles the intercept

- **Motivation:** A missing translation can cause a large systematic prediction shift.
- **Diagnostic:** Inspect the bias formula and fit synthetic `Y=XW+b` with a large `b`;
  compare with a forced zero-intercept solve.
- **Result:** Code uses `bias=mean(B)-mean(A)@C`. The controlled test recovers both
  weights and nonzero bias to `1e-10`; the no-intercept control has MSE above 40.
- **Status:** **rejected** as an implementation explanation. Intercept is present and
  algebraically correct. Large real fitted biases remain a symptom of unstable weights,
  not evidence that bias was omitted.
- **Next:** compare saved checkpoint forward maps and bias distributions across all pairs.

### H3 — Feature normalization or centering differs

- **Motivation:** SGD is scale-sensitive and SVD cutoff decisions depend on scaling.
- **Diagnostic:** Compare YAMLs and shared preprocessing branch; run scaling/centering
  controls.
- **Preliminary result:** `normalize_embeddings=false` for both methods in all four YAMLs.
  Both receive the same raw anchors. `linear_close` centers internally to fit its
  intercept; learned `nn.Linear` estimates the equivalent translation jointly.
- **Status:** **rejected** as a between-run configuration difference; **still open** as a
  possible stabilizing reformulation.
- **Next:** controlled scaling test and standardized real-like synthetic regime.

### H4 — The closed-form implementation has a transpose, bias, dtype, or apply bug

- **Motivation:** A materialization bug could fit correctly in formula form but fail when
  applied through `nn.Linear`.
- **Diagnostic:** Compare formula and checkpoint forward; compare against independent
  NumPy and scikit-learn/augmented-matrix least squares.
- **Preliminary result:** Production code contains an explicit formula-vs-`nn.Linear`
  assertion. Independent synthetic recovery succeeds. Saved anchor MSE near `4e-12`
  is inconsistent with a gross transpose/apply error.
- **Status:** **partially rejected**; remaining cross-solver and saved-checkpoint checks
  are pending at this stage.
- **Next:** run exact independent-solver comparison and inspect checkpoint ranks.

### H5 — The supplied experiments differ in folds, anchors, features, checkpoints, seed, or preprocessing

- **Motivation:** An uncontrolled pair cannot attribute MAE to projector choice.
- **Diagnostic:** Diff aggregate configs/YAMLs and byte-compare corresponding anchor,
  old-test, and projector split CSVs for all 25 pairs.
- **Preliminary result:** Model lists, fold Cartesian product, seed, anchor count/selection,
  source/target split names, loss, normalization, and backbone pair match. Refinement
  hyperparameters do not match.
- **Status:** **partially supported**. There is a real refinement confound, but code proves
  it cannot affect the `refinement=3` headline MAE. Full CSV hash audit is pending at
  this stage.
- **Next:** complete all-subtrial artifact matching.

### H6 — Exact OLS overfits because `K << D_old`

- **Motivation:** With 100 centered anchors and 384 input dimensions, rank is at most 99;
  infinitely many affine maps interpolate the anchors. Anchor noise/model mismatch can
  be fit exactly rather than generalized.
- **Diagnostic:** Compare anchor vs held-out error, rank, parameter norms, and controlled
  tests varying `K/D`, noise, and rank deficiency.
- **Preliminary result:** First pair: near-zero closed-form anchor MSE but dramatically worse
  held-out reconstruction and 10x/225x parameter norms. This is direct overfit evidence.
- **Status:** **supported**, pending replication across all 50 pairs and the synthetic
  sweep at this stage.
- **Next:** quantify across every checkpoint and reproduce the transition synthetically.

### H7 — Numerical ill-conditioning and the `rcond=1e-5` cutoff amplify weak directions

- **Motivation:** Even within rank 99, small retained singular values produce large
  coefficients. Hard truncation does not smoothly shrink marginal directions.
- **Diagnostic:** Measure anchor spectra/condition numbers where recoverable; sweep
  `rcond` and compare weight norm/generalization under controlled ill-conditioning.
- **Preliminary result:** Huge closed-form norms and errors are consistent with this mechanism,
  but no raw real-anchor spectrum has yet been safely recovered.
- **Status:** **inconclusive but plausible**.
- **Next:** synthetic singular-spectrum sweep; seek non-pickle anchor embedding artifacts.

### H8 — Learned `linear` is effectively regularized despite zero weight decay

- **Motivation:** Finite-step gradient methods learn large-singular-value directions first;
  small initialization, low LR, minibatching, and validation checkpointing can act like
  spectral shrinkage/early stopping.
- **Diagnostic:** Compare learned norms/anchor error/best epochs; controlled GD vs OLS vs
  ridge, one factor at a time.
- **Preliminary result:** First learned checkpoint has 10x lower weight norm and accepts
  nonzero anchor error. Its best epoch is 750, so early *selection* did not truncate this
  particular run, but finite low-LR AdamW dynamics still did not reach the OLS interpolant.
- **Status:** **partially supported**.
- **Next:** all-pair best-epoch distribution and synthetic optimization path.

### H9 — Refinement causes the headline gap

- **Motivation:** Refinement recipes differ substantially and can alter both projector and
  classifier.
- **Diagnostic:** Trace result-selection code and compare pre/post fields.
- **Result:** With `refinement=3`, both modes run independently and neither replaces the
  headline projection. Per-subtrial summaries repeat the same pre-refinement MAE in both
  mode rows. Refinement recipes make refined outcomes incomparable but do not generate
  the supplied headline gap.
- **Status:** **rejected for headline MAE; supported as a confound for refinement claims**.
- **Next:** compare same-recipe synthetic refinement only if the base-map explanation is
  insufficient.

### H10 — Anchor selection itself is unusually bad for `linear_close`

- **Motivation:** Different random anchor rows would alter an unstable OLS fit greatly.
- **Diagnostic:** Hash every corresponding `anchors.csv` and verify seed/call order.
- **Preliminary result:** Both methods use seed 42 and deterministic selection; first pair's
  anchor files appear aligned. Full hash result is pending at this stage.
- **Status:** **inconclusive**.
- **Next:** all-pair hash audit.

## Chronological experiment log

### E0 — Repository and artifact map

- **Question:** What code and evidence already exist, and can investigation avoid changing
  the pipeline?
- **Motivation:** Establishing provenance and non-destructive boundaries prevents diagnostics
  from being confused with production changes or unrelated worktree edits.
- **Method:** Read current code, git diff, formulation document, result directories, YAMLs,
  and artifact schemas.
- **Result:** `linear_close` is already a centered SVD OLS implementation with `rcond=1e-5`.
  The working tree has unrelated user changes, including launch-config snapshot support;
  they are preserved. Rich per-subtrial safe artifacts exist.
- **Interpretation:** A separate analysis module/document is sufficient; no pipeline edit
  is needed.
- **Conclusion:** Proceed with isolated diagnostics.
- **Next:** establish pair comparability and baseline gap.

### E1 — Aggregate comparison and config diff

- **Question:** Do the two reported pairs differ only by projector?
- **Motivation:** A configuration confound must be ruled out before attributing any MAE gap
  to the projector estimator.
- **Method:** Read aggregate configs/results, diff aggregate logs, and inspect all four
  YAMLs.
- **Result:** The base data/model/projector settings match except method, while refinement
  recipes differ as tabulated above. Headline gaps are +4.42 and +4.85 MAE.
- **Interpretation:** Refined outcomes are not controlled. Headline outcomes can still be
  attributed after tracing `refinement=3` reporting behavior.
- **Conclusion:** Treat refinement mismatch separately; continue with pure projection.
- **Next:** first matched-subtrial parameter/generalization check.

### E2 — First matched-subtrial overfit diagnostic

- **Question:** Is worse source MAE accompanied by exact anchor interpolation, unstable
  parameters, and poor held-out embedding reconstruction?
- **Motivation:** A single paired fold is the smallest real-data test of the proposed
  interpolation-versus-generalization mechanism.
- **Method:** Load both projector checkpoints with `weights_only=True`; read summary and
  reconstruction CSVs.
- **Result:** Closed form has ~zero anchor MSE, 10.3x weight norm, 225x bias norm, and much
  worse held-out reconstruction and source MAE.
- **Interpretation:** The failure happens before the classifier/refinement and has the
  classic exact-fit/high-variance pattern.
- **Conclusion:** H6 supported on the first pair; H2/H4 gross implementation bugs unlikely.
- **Next:** replicate across all pairs and run controlled synthetic transitions.

### E3 — Synthetic affine formula sanity checks

- **Question:** Does the independently reproduced closed form recover a known affine map,
  and is its intercept essential?
- **Motivation:** An ideal known mapping isolates algebra, transpose, and intercept handling
  from the difficult statistics of the real experiment.
- **Method:** Generate well-determined Gaussian `X`, exact `Y=XW+b`, then compare the
  centered solve with a forced zero-intercept solve. Also verify ridge shrinks weights in
  a noisy underdetermined setting.
- **Result:** Exact affine recovery to `1e-10`; MSE below `1e-20`. Removing the intercept
  raises MSE above 40. Ridge reduces coefficient norm in the underdetermined control.
- **Interpretation:** The formula and intercept are sound in the ideal regime; instability
  requires underdetermination/noise/conditioning rather than a universal implementation
  error.
- **Conclusion:** H2 rejected; ideal-case portion of H1/H4 passes.
- **Next:** add learned-linear and progressive real-like synthetic conditions.

### E4 — Independent solver equivalence

- **Question:** Does production `_fit_linear_closed_form` disagree with a separate solver
  because of centering, transpose, materialization, or dtype?
- **Motivation:** Agreement across independent implementations is the narrowest way to reject
  a production least-squares/materialization bug.
- **Method:** On the same random determined problem, compare held-out predictions from
  production code, the independent diagnostic implementation, scikit-learn
  `LinearRegression`, and PyTorch `torch.linalg.lstsq` on an augmented `[X, 1]` design.
- **Result:** All held-out predictions agree within `2e-5`; production versus the
  independent NumPy and PyTorch solutions agrees within `2e-6`.
- **Interpretation:** The implementation encodes the standard affine OLS solution. Float64
  SVD followed by float32 materialization is not causing the multi-point MAE gap.
- **Conclusion:** H4 rejected.
- **Next:** audit every saved fold pair for data identity, rank, and gain.

### E5 — All-50-pair artifact audit

- **Question:** Does the first-pair exact-fit/high-gain pattern replicate, and are the
  saved row manifests and configurations truly paired?
- **Motivation:** Replication and byte-level pairing are needed to distinguish a systematic
  method effect from one fold or an accidental data/config mismatch.
- **Method:** Pair directories by `(new_fold, old_fold)`, byte-hash anchors, source-test
  rows, and all three projector split CSVs, then load the 100 projector checkpoints with
  `weights_only=True`.
- **Result:** All 25/25 MIntPAIN pairs and 25/25 UNBC pairs have identical anchor and
  source-test row manifests and train/validation/test split files. The configs reference
  the same feature paths and model checkpoints; raw embedding matrices are not duplicated
  in the safe subtrial artifacts and therefore cannot be independently byte-hashed.
  Summary statistics:

| Statistic (mean across 25) | MInt `linear_close` | MInt `linear` | UNBC `linear_close` | UNBC `linear` |
|---|---:|---:|---:|---:|
| Source-test MAE | 5.9585 | 1.5277 | 5.7423 | 0.9729 |
| Anchor MSE | `1.59e-11` | 0.0891 | `2.67e-11` | 0.0862 |
| Weight Frobenius norm | 270.26 | 13.16 | 318.80 | 13.13 |
| Bias norm | 277.90 | 0.683 | 280.64 | 0.664 |
| Matrix rank | 99 | 384 | 99 | 384 |
| Operator norm (largest singular value) | 261.09 | 1.678 | 307.27 | 1.415 |
| Map condition number on retained singular values | 18,612 | 18.0 | 26,739 | 15.1 |
| Learned best epoch | — | 749.4 | — | 747.2 |

  `linear_close` is worse in 49/50 fold pairs. Across all 50, its weight norm correlates
  strongly with its MAE (Pearson `r=0.705`, `p=1.08e-8`; Spearman `rho=0.508`,
  `p=1.67e-4`). The close-form rank is exactly the centered-anchor maximum, `K-1=99`,
  in every supplied run. `rcond=1e-5` therefore retains every nonzero sample direction
  relevant to the fitted map; it does not regularize this regime meaningfully.
- **Interpretation:** Fold, row selection, checkpoint, path, and preprocessing differences
  are eliminated. Exact interpolation, low rank, extreme gain, and poor generalization
  repeat across both datasets. Direct raw-tensor identity remains unverified.
- **Conclusion:** H10 rejected; H5 rejected for every saved/configured factor but remains
  narrowly inconclusive for independent byte identity of the regenerated feature tensors;
  H6 and H7 supported.
- **Next:** determine where the generalization failure appears—within BioVid or during
  cross-domain application.

### E6 — Target-domain versus source-domain reconstruction

- **Question:** Is OLS already catastrophic on held-out BioVid rows, or specifically on
  the source-domain data to which the cross-space map is ultimately applied?
- **Motivation:** Localizing where error grows distinguishes ordinary target-domain overfit
  from unstable extrapolation under dataset shift.
- **Method:** Read the saved projector test metrics (BioVid `val.csv`, disjoint from
  BioVid train anchors) and safe source-domain oracle-reconstruction logs for fold pair
  `(0,0)`.
- **Result:**

| Dataset being projected | Metric | `linear_close` | `linear` |
|---|---|---:|---:|
| BioVid held-out (MInt source model) | embedding MSE | 0.0701 | 0.0621 |
| MIntPAIN source-domain | embedding L2 mean | 32.144 | 6.924 |
| MIntPAIN source-domain | embedding cosine mean | 0.797 | 0.956 |
| BioVid held-out (UNBC source model) | embedding MSE | 0.1302 | 0.0486 |
| UNBC source-domain | embedding L2 mean | 36.503 | 7.638 |
| UNBC source-domain | embedding cosine mean | 0.309 | 0.962 |

  The closed form is somewhat worse on held-out target-domain rows, but the collapse is
  much larger after crossing datasets. Projected norms are not uniformly enormous—the
  MInt fold `(0,0)` mean projected/true norm ratio is 0.989—so the decisive error is also
  directional, not merely a scalar norm explosion.
- **Interpretation:** The map is learned on BioVid anchors in the source backbone's
  embedding space, then extrapolated to MIntPAIN/UNBC embeddings. Its very large singular
  gains make small or domain-specific components matter. The bounded learned map suppresses
  those components, producing embeddings much closer in direction to the real target-space
  oracle embeddings.
- **Conclusion:** Cross-domain extrapolation interacts with underregularized OLS and is the
  immediate route from anchor overfit to downstream MAE.
- **Next:** inspect downstream prediction distributions and reproduce the interaction
  synthetically.

### E7 — Downstream prediction-distribution diagnostic

- **Question:** Does the high-gain map create implausible classifier outputs rather than a
  subtle MAE shift?
- **Motivation:** Downstream ranges, clipping, and constant baselines reveal whether the MAE
  gap reflects projector instability or merely a small calibration offset.
- **Method:** Compare aggregate projected prediction quantiles to labels, source-model
  predictions, clipping controls, and constant baselines.
- **Result:**

| Aggregate | Prediction mean / std | Min / max | Outside `[0,4]` | MAE |
|---|---:|---:|---:|---:|
| MInt `linear_close` | 4.424 / 7.905 | -26.48 / 31.89 | 65.9% | 5.953 |
| MInt `linear` | 2.026 / 0.317 | 1.25 / 3.64 | 0% | 1.528 |
| UNBC `linear_close` | 5.691 / 6.795 | -11.21 / 28.80 | 73.2% | 5.826 |
| UNBC `linear` | 1.367 / 0.660 | 0.14 / 2.75 | 0% | 0.978 |

  Clipping close-form predictions to `[0,4]` lowers MAE to 2.222 (MInt) and 1.945
  (UNBC), still much worse than `linear`; clipping is therefore a symptom mask, not a
  projection fix. The median-label constant baselines score 1.252 and 1.275 respectively.
  MInt `linear` is itself worse than the constant median, while UNBC `linear` is better.
- **Interpretation:** Exact OLS sends source-domain embeddings into target-head-sensitive
  directions that were not controlled by the anchors. Learned `linear` is stable but, in
  MInt, much of its apparent advantage is collapse toward central predictions rather than
  strong recovery of the true cross-space map.
- **Conclusion:** The large MAE gap is real but must not be interpreted as proof that the
  current `linear` estimator accurately recovers a mapping.
- **Next:** reproduce determined, underdetermined, low-rank, regularized, and domain-shift
  regimes synthetically.

### E8 — Progressive synthetic experiments

- **Question:** Under which controlled conditions do the estimators agree and diverge?
- **Motivation:** Progressively changing one factor at a time is required to separate a
  universal implementation failure from underdetermination, noise, conditioning, and shift.
- **Method:** Use the fixed-seed, executable scenarios in
  `analysis/linear_close_synthetic.py`, varying one factor at a time. Evaluate anchor MSE,
  held-out embedding MSE, downstream linear-head MAE, rank, and parameter norm. The learned
  diagnostic uses the production recipe's `nn.Linear`, AdamW, validation selection,
  minibatching, and initialization. Running `python -m analysis.linear_close_synthetic`
  regenerates every number below.
- **Results:**

1. **Well-determined, noiseless `Y=XW+b` (200 anchors, 8 inputs).** OLS test MSE is
   `3.16e-14`; converged Adam test MSE is `1.47e-12`; maximum held-out disagreement is
   `5.57e-6`. Thus the two formulations can be equivalent when the solution is unique
   and optimization converges.
2. **Sufficient anchors at real input dimension (800 anchors, 384 inputs, additive
   noise).** OLS test MSE is 0.00947 and ridge(10) is 0.00975; OLS has full rank 384.
   Exact OLS is not intrinsically worse in a sufficiently determined problem.
3. **Underdetermined but noiseless (100 anchors, 384 inputs).** OLS has rank 99 and
   anchor MSE `6.32e-30`, but test MSE is 0.762 because unseen directions are
   unidentified; ridge(10) is 0.763. Underdetermination alone creates ambiguity but does
   not guarantee that ridge—or finite Adam—will recover information absent from anchors.
4. **Low-rank/noisy anchors (latent rank 20, 100 anchors, 384 inputs).** Retaining all 99
   sample directions gives zero anchor MSE, weight norm 102.8, and test MSE 0.00137.
   `rcond=0.01` keeps rank 20, accepts anchor MSE 0.00194, lowers the norm to 3.91, and
   improves test MSE to 0.000571. Ridge(1) gives norm 3.86 and MSE 0.000671.
5. **Cross-domain nuisance shift.** At anchor-like nuisance scale 0.001, OLS has weight
   norm 277.7 and test MSE 0.00879. Raising only deployment nuisance to 0.05 and 0.10
   raises OLS MSE to 16.29 and 65.14. Ridge remains at 0.00359/0.01284; production-recipe
   finite Adam remains near 1.147 rather than amplifying the shift. At scale 0.10 the
   downstream linear-head MAEs are 4.635 (OLS), 0.091 (ridge), and 0.747 (Adam). This
   recreates the qualitative failure: exact high-gain fitting is best near the anchor
   distribution but collapses under a domain-specific direction, while bounded Adam can
   be stable yet substantially underfit.
6. **Anchor-count sweep (80 inputs, rank-12 noisy relation).** OLS test MSE is 0.0149,
   0.0112, **9.958**, 0.0101, and 0.00319 for K=20, 40, 80, 160, and 320. At the
   interpolation threshold K≈D its norm jumps to 897.2. Ridge(1) remains 0.0214,
   0.00679, 0.00270, 0.00111, and 0.000546. This is the classic high-variance
   interpolation peak; more anchors help once the problem becomes overdetermined.
7. **Coordinate scaling control.** When feature scales span `1` to `1e-8`, raw
   `rcond=1e-5` OLS drops five informative dimensions and has test MSE 7.40;
   standardization restores rank 12 and numerical-zero error (`2.85e-29`). Scaling can
   matter, but it is not a config difference in the supplied runs and does not solve
   correlated rank deficiency by itself.

- **Interpretation:** No single slogan such as "closed form is bad" is correct. The
  failure requires the actual combination: few anchors relative to dimension/effective
  rank, retained weak directions, noise or nonlinearity in the cross-model relation, and
  application under dataset shift. Ridge or stronger spectral truncation trades negligible
  anchor fit for large stability gains.
- **Conclusion:** H1 is rejected in the ideal regime but supported as *estimator*
  non-equivalence in the real regime; H6–H8 are supported with controlled causal evidence.
- **Next:** use existing real anchor sweeps and refinement outcomes as independent checks.

### E9 — Existing real anchor-count sweep

- **Question:** Does increasing anchors improve real `linear_close` before refinement?
- **Motivation:** A real-data anchor sweep tests whether the K-to-D ratio identified
  synthetically also predicts behavior outside the two headline experiment pairs.
- **Method:** Analyze the existing BioVid-DFER -> UNBC-VideoMAE search summary, using the
  invariant `mae_micro_old_oncsv_before` rather than recipe-dependent refined MAE.
- **Result:** At K=100, 250, 1000, `linear_close` source-test MAE is 20.156, 17.247,
  and 9.610; anchor MSE is approximately zero at K=100/250 and 0.150 at K=1000.
  The corresponding learned-linear MAEs are 0.995, 1.039, and 1.134, with anchor MSE
  0.502, 0.408, and 0.374. Closed-form checkpoint ranks are 99, 248, and 381; their
  weight norms are 868, 3,979, and 3,714, versus 13.1–13.25 for learned linear.
- **Interpretation:** More anchors constrain OLS and halve its MAE, but even 1,000 anchors
  do not make the cross-domain relation sufficiently affine/stable. Rank remains below
  384 and gain remains extreme. More anchors help but are not a substitute for
  regularization.
- **Conclusion:** H6 is strengthened; "only K=100" is not the full explanation.
- **Next:** quantify refinement behavior.

### E10 — Refinement interaction

- **Question:** Does refinement create, worsen, hide, or repair the base gap?
- **Motivation:** Refinement is configured in all supplied runs and must be separated from
  the pure projector result, even though its hyperparameters are not matched.
- **Method:** Aggregate both refinement modes' before/after fields across the 25 folds.
- **Result:** As established in E1, refinement=3 cannot alter headline MAE. Nevertheless,
  the saved close-form refinement runs reduce mean source MAE from 5.959 to 1.325/1.330
  (MInt, linear-only/projector+linear) and from 5.742 to 1.050/0.944 (UNBC). Joint
  refinement increases close-form anchor MSE from about zero to 0.078 (MInt) and 0.063
  (UNBC), explicitly moving away from interpolation. Target test MAE worsens from 1.082
  to about 1.17–1.22. Learned-linear refinements start much closer to the final values and
  preserve target MAE near 1.08.
- **Interpretation:** Refinement did not cause the reported gap. It can mask/repair the
  source failure by abandoning the exact map, at a target-preservation cost. The two
  supplied refinement recipes differ by orders of magnitude, so their refined outcomes
  are not a controlled method comparison and should not be ranked against each other.
- **Conclusion:** H9 rejected as a cause; refinement interaction is real and supports the
  overfit explanation.
- **Next:** synthesize final disposition and recommendation.

### E11 — Real fold-0 `rcond=1e-2` probe

- **Question:** Can the existing `linear_close` implementation approach learned `linear`
  stability and MAE by changing only `closed_form_rcond` from `1e-5` to `1e-2`?
- **Motivation:** The synthetic spectrum test identified `1e-2` as a promising hard cutoff,
  but a real, paired test is required before recommending it for this pipeline.
- **Method:** On 2026-08-19, rerun model pair `(new_fold=0, old_fold=0)` independently for
  MIntPAIN and UNBC with seed 42, the same 100 balanced BioVid-train anchors, raw features,
  model checkpoints, source test split, and `linear_close` implementation. Disable
  refinement because it cannot affect the pure-projection headline. Change only
  `closed_form_rcond` to `0.01`. Compare against the saved `rcond=1e-5` and learned-linear
  fold-0 runs. Directly deserialize the trusted result artifacts under the sandbox-safe
  Manager mock and verify that source tensors and both old/new anchor embedding matrices
  are bit-identical across all three methods (maximum absolute difference `0.0`). Recompute
  anchor MSE from weights-only checkpoints and use the same saved target-space oracle
  embeddings for source reconstruction.
- **Result:**

| Dataset / diagnostic | `rcond=1e-5` | `rcond=1e-2` | learned `linear` |
|---|---:|---:|---:|
| **MInt source MAE** | 4.2228 | **1.5726** | 1.4493 |
| MInt CCC | -0.009 | -0.058 | 0.012 |
| MInt retained map rank | 99 | **31** | 384 |
| MInt anchor MSE | `4.08e-12` | **0.02444** | 0.06170 |
| MInt weight / bias norm | 135.66 / 153.41 | **24.72 / 26.04** | 13.13 / 0.682 |
| MInt operator norm | 133.51 | **24.11** | 1.555 |
| MInt held-out BioVid embedding MSE | 0.07008 | **0.04281** | 0.06212 |
| MInt source oracle L2 / cosine | 32.79 / 0.788 | **9.24 / 0.932** | 7.00 / 0.953 |
| MInt predictions outside `[0,4]` | 85.3% | **1.9%** | 0% |
| **UNBC source MAE** | 3.6130 | **1.1614** | 1.0804 |
| UNBC CCC | 0.286 | 0.300 | 0.360 |
| UNBC retained map rank | 99 | **21** | 384 |
| UNBC anchor MSE | `1.46e-11` | **0.02287** | 0.05334 |
| UNBC weight / bias norm | 312.88 / 84.26 | **17.95 / 22.20** | 13.14 / 0.666 |
| UNBC operator norm | 309.27 | **17.11** | 1.503 |
| UNBC held-out BioVid embedding MSE | 0.13018 | **0.04169** | 0.04865 |
| UNBC source oracle L2 / cosine | 36.50 / 0.309 | **5.28 / 0.978** | 7.64 / 0.962 |
| UNBC predictions outside `[0,4]` | 66.7% | **0%** | 0% |

  The stronger cutoff lowers MAE by 62.8% for MInt and 67.9% for UNBC relative to
  `rcond=1e-5`. It finishes only 0.123 and 0.081 MAE above learned `linear` on these folds.
  MInt predictions contract from `[0.51, 8.62]` to `[-0.33, 4.20]`; UNBC predictions
  contract from `[-7.35, 5.64]` to `[1.31, 2.62]`. The output directories are:
  `Cross_projection/diagnostics/cross_space_projection_linear_close_rcond_1e-2_mint_fold00_K100_balance_class_random_train_test_linear_close_linclose_rc0.01_normF_sp0-50-50_1787155894`
  and
  `Cross_projection/diagnostics/cross_space_projection_linear_close_rcond_1e-2_unbc_fold00_K100_balance_class_random_train_test_linear_close_linclose_rc0.01_normF_sp0-50-50_1787155894`.
- **Interpretation:** `rcond=1e-5` was indeed retaining harmful weak directions. A cutoff of
  `1e-2` gives up exact anchor interpolation, reduces projector gain by factors of 5.5
  (MInt) and 18.1 (UNBC), improves held-out BioVid reconstruction beyond learned `linear`,
  and nearly closes the source-MAE gap. This is strong causal evidence for the
  weak-direction/high-gain explanation. It does not make the estimators identical: the
  hard-cutoff maps still have much larger bias and operator norms than learned `linear`,
  MInt CCC does not improve, and the result covers only one fold pair per dataset.
- **Conclusion:** **Supported as a promising configuration, not established as an optimum.**
  The existing implementation can become dramatically more stable with `rcond=1e-2`, but
  a 25-pair cutoff sweep is required before using it as the default or claiming equivalent
  aggregate performance.
- **Next:** Run at least `rcond in {1e-3, 1e-2, 3e-2, 1e-1}` over all matched folds, select
  by held-out projector validation rather than source-test MAE, and compare the selected
  cutoff to validation-selected ridge under identical refinement settings.

## Final hypothesis disposition

| Hypothesis | Final status | Decisive evidence |
|---|---|---|
| Different affine function family | Rejected | Both implement `x @ W.T + b`; ideal synthetic maps agree |
| Different nominal anchor objective | Rejected for supplied MSE runs | Both use anchor embedding MSE; OLS is its exact minimizer |
| Different practical estimator / implicit objective | Supported | Finite AdamW remains bounded/full-rank; OLS is rank-99 exact interpolation |
| Missing/wrong intercept | Rejected | Formula, large-offset synthetic recovery, and four-solver agreement |
| Implementation/transposition/apply bug | Rejected | Production, NumPy, PyTorch, sklearn, and `nn.Linear` materialization agree |
| Floating-point solver failure | Rejected | Four independent solvers agree; the instability is estimator sensitivity, not erroneous arithmetic |
| No effective regularization in OLS | Supported | `rcond=1e-5` retains rank 99, zero anchor loss, huge gain; ridge/truncation controls work |
| Stronger hard truncation improves real stability | Supported on two fold-0 probes | `rcond=1e-2` reduces rank to 31/21 and MAE by 62.8%/67.9%; full cross-validation remains pending |
| Rank deficiency / underdetermination | Supported | K=100, D=384, centered rank ceiling 99; all saved OLS maps rank 99 |
| High-gain/anisotropic fitted map | Supported | Operator norm 261/307 vs 1.68/1.42; map condition 18k/27k; MAE tracks weight norm |
| Ill-conditioned real anchor design | Partially supported | Exact interpolation and extreme gain are consistent with weak retained design directions, and synthetic controls establish causality; raw real anchor spectra were unavailable in safe artifacts |
| Feature normalization differs between runs | Rejected | Both supplied methods use raw features (`normalize_embeddings=false`) |
| Scaling/centering can matter in general | Partially supported | Intercept centering is correct; synthetic scale test benefits from standardization |
| Different anchors/folds/checkpoints/seeds/preprocessing | Rejected for headline comparison | 50/50 row manifests match byte-for-byte; same feature paths, models, seed, and preprocessing |
| Independently verified raw feature-tensor identity | Inconclusive | Safe subtrial artifacts do not duplicate raw embedding matrices; identical paths/checkpoints/rows make a difference unlikely but do not constitute a tensor hash |
| Anchor overfit | Supported | `~1e-11` anchor MSE with poor source reconstruction and MAE; ridge controls |
| Cross-domain interaction | Supported | modest BioVid error becomes severe MInt/UNBC directional error; synthetic reproduction |
| Refinement causes headline gap | Rejected | refinement=3 deliberately reports pure projection |
| Refinement can repair/mask OLS failure | Supported | source MAE falls as joint refinement gives up exact anchor fit |

## Evidence-based explanation

The tested `linear_close` implementation is correct: production, NumPy, PyTorch, and
scikit-learn solutions agree, and it correctly materializes the centered affine intercept.
No tested evidence supports an implementation bug. The large MAE gap instead arises
because that mathematically valid estimator is inappropriate for the actual statistical
regime.

The supplied runs fit 196,608 weights plus 512 biases (197,120 parameters total) from only
100 anchor pairs. After centering, at most 99 sample directions are observable.
`rcond=1e-5` retains all 99 and OLS exactly interpolates anchor-specific cross-model
variation. The resulting map has roughly 150–220 times the learned map's operator norm
and is extremely anisotropic. In the inspected held-out BioVid folds it is moderately
worse, but it is then applied to a different dataset's source embeddings. Small
domain-specific or weakly constrained directions are amplified and rotated into
target-head-sensitive directions. Real oracle-reconstruction cosine collapses,
predictions range far outside the pain scale, and MAE explodes. Because safe artifacts do
not contain the raw anchor matrices, the report does not claim a directly measured real
design-matrix condition number; the saved map gains and controlled spectrum tests provide
the conditioning evidence.

`linear` does not provide the "same OLS, just optimized numerically." Its very low learning
rate and finite 750-epoch AdamW trajectory are a different, implicitly regularized
estimator.
In 43/50 runs validation still prefers the final epoch; the others stop only at 735–746,
so classic early checkpointing is minor. The key is finite optimization from bounded
`nn.Linear` initialization. PyTorch's expected initialization norms for 384->512 are
approximately 13.06 (weights) and 0.667 (bias), almost exactly the saved 13.13–13.16 and
0.664–0.683. It learns useful high-support directions while never approaching the
high-gain interpolating solution. This bounded underfit is robust under dataset shift,
although MInt results show it can behave more like a central-prediction baseline than an
accurate cross-space inverse.

## Recommendation

1. **Do not "fix" the SVD call or replace it with a matrix inverse.** `np.linalg.lstsq`
   is correct and safer than normal equations. The issue is the estimator, not the solver.
2. **Remove current `linear_close` from the main method-comparison pipeline until it is
   regularized.** It may remain available as an explicitly experimental failure-mode
   baseline.
3. **Rename the existing method to `affine_ols_tsvd` (or `linear_ols_tsvd`).**
   `linear_close` implies a closed-form equivalent of `linear`, which is false for the
   finite AdamW estimator used in experiments.
4. **Add a separate validation-selected ridge projector as the primary closed-form method.**
   Use centered SVD ridge coefficients
   \(C=V\,\mathrm{diag}(s/(s^2+\lambda))\,U^T B_c\), do not penalize the intercept,
   and select \(\lambda\) on held-out reconstruction plus downstream/source-like
   validation where methodologically allowed. A logarithmic grid should include values
   on both sides of the effective singular scale; store rank, spectrum, operator norm,
   bias norm, anchor/held-out errors, and prediction-range diagnostics.
5. **Keep coordinate standardization optional but enabled in the ridge sweep.** It handles
   unequal feature scales, but it cannot replace ridge because the main problem includes
   correlated low-rank structure and domain shift.
6. **Use more and more diverse anchors, but not as the sole remedy.** Existing K=1000
   evidence improves OLS yet remains poor. Anchor coverage should include variation close
   to the deployment/source domain if that does not violate the experimental protocol.
7. **Do not remove the intercept.** It is correct and necessary. Do not clip predictions
   as the projection remedy; clipping only hides the instability.
8. **Rerun refinement comparisons with identical refinement hyperparameters.** The current
   refined artifacts are useful mechanistic evidence but not a fair ranking. Keep pure
   projection and each refinement mode reported separately.
9. **Clarify the learned baseline name/documentation too.** `linear_adam` or
   `linear_finite_adamw` would communicate that its stability comes from the training
   trajectory, not from solving unregularized OLS to convergence.

The recommended end state is therefore: retain the present implementation only as a
renamed diagnostic OLS baseline, introduce a regularized ridge formulation for any
production/main comparison, and keep the current main pipeline unchanged until that new
method is implemented and validated in directly comparable runs.

The E11 follow-up refines this recommendation: the existing implementation with
`rcond=1e-2` is a credible *regularized truncated-SVD candidate* and is suitable for a
full validation sweep. It should not replace the default based on two fold-0 results, and
ridge remains preferable as the smoother regularizer to benchmark against.

## Reproduction

Run the focused diagnostic tests:

```bash
pytest -q tests/test_linear_close_investigation.py
```

Regenerate the complete progressive synthetic suite as JSON:

```bash
python -m analysis.linear_close_synthetic
```

The safe artifact API avoids general subtrial-pickle deserialization:

```python
from analysis.linear_close_investigation import compare_experiment_roots

rows = compare_experiment_roots(
    "Cross_projection/mintVMAE-bioDFER/refinement3_linear_close_cross-validation",
    "Cross_projection/mintVMAE-bioDFER/refinement3_linear_cross-validation",
)
```

`fit_affine_closed_form`, `fit_affine_ridge`, `fit_affine_adam`, and
`affine_metrics` reproduce the controlled experiments without importing or modifying the
main pipeline.
