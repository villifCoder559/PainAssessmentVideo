# Skip LaTeX Consistency Checks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--skip-consistency-checks` so LaTeX generation can bypass the two numeric consistency guards while remaining strict by default.

**Architecture:** Thread one boolean from `main()` through the existing table-generation call chain into `_load_direction()`. Use it to guard the projector-only and repeated-baseline consistency checks; leave every other validation unchanged.

**Tech Stack:** Python standard-library `argparse`, pandas, unittest.

## Global Constraints

- `skip_consistency_checks` defaults to `False`.
- The flag skips only projector-only and repeated-baseline numeric consistency checks.
- No new dependency or abstraction.

---

### Task 1: Add and verify the consistency-check bypass

**Files:**
- Modify: `tests/test_generate_cross_projection_latex_table.py`
- Modify: `cross_space_generate_latex_table.py`

**Interfaces:**
- Consumes: existing `generate_table(first_root, second_root, *, projection, stage, fake_distribution=None, decimals=2)`.
- Produces: `generate_table(..., skip_consistency_checks: bool = False)` and CLI option `--skip-consistency-checks`.

- [ ] **Step 1: Extend the existing rejection tests with bypass assertions**

In `test_rejects_inconsistent_baselines`, call the same inputs again with:

```python
latex = generate_table(
  first,
  second,
  projection="real",
  stage="projector_linear",
  skip_consistency_checks=True,
)
self.assertIn(r"\begin{table}[H]", latex)
```

In `test_projector_only_rejects_disagreeing_before_values`, call the same
inputs again with:

```python
latex = generate_table(
  first,
  second,
  projection="real",
  stage="projector_only",
  skip_consistency_checks=True,
)
self.assertIn(r"\begin{table}[H]", latex)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python3 -m unittest \
  tests.test_generate_cross_projection_latex_table.TestGenerateTable.test_rejects_inconsistent_baselines \
  tests.test_generate_cross_projection_latex_table.TestGenerateTable.test_projector_only_rejects_disagreeing_before_values
```

Expected: both tests error because `generate_table()` does not yet accept
`skip_consistency_checks`.

- [ ] **Step 3: Implement the minimal boolean plumbing**

Add `skip_consistency_checks: bool = False` to `_load_direction`,
`_generate_stage_table`, and `generate_table`. Pass it through both directions
and every stage. Guard both existing checks:

```python
if not skip_consistency_checks and (
  values.isna().any() or values.max() - values.min() > 1e-10
):
```

Add the CLI option:

```python
parser.add_argument(
  "--skip-consistency-checks",
  action="store_true",
  help="Skip repeated projector-only and baseline metric consistency checks.",
)
```

Pass `args.skip_consistency_checks` to `generate_table()` in `main()`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command again.

Expected: `Ran 2 tests` and `OK`.

- [ ] **Step 5: Run the complete test file**

Run:

```bash
python3 -m unittest tests.test_generate_cross_projection_latex_table
```

Expected: all tests pass.

- [ ] **Step 6: Reproduce the requested CLI command with the flag**

Run:

```bash
python3 cross_space_generate_latex_table.py \
  Cross_projection/bioVmae_to_mintDfer \
  Cross_projection/mintVMAE-bioDFER \
  --projection real \
  --stage all \
  --skip-consistency-checks \
  --output /tmp/cross_projection_all.tex
```

Expected: exit code 0, a `Saved LaTeX table to:` message, and three LaTeX
tables in `/tmp/cross_projection_all.tex`.

- [ ] **Step 7: Review the final diff**

Run:

```bash
git diff --check
git diff -- cross_space_generate_latex_table.py tests/test_generate_cross_projection_latex_table.py
```

Expected: no whitespace errors; only the flag plumbing and two test assertions
are present.
