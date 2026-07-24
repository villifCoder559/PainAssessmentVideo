# Skip LaTeX Table Consistency Checks

## Goal

Allow `cross_space_generate_latex_table.py` to generate a table when repeated
floating-point metrics differ slightly, while preserving the current strict
behavior by default.

## Interface

Add the optional CLI flag:

```text
--skip-consistency-checks
```

The corresponding `generate_table` keyword defaults to `False`.

## Behavior

When enabled, the flag skips only:

- the projector-only check that duplicated pre-refinement values agree across
  refinement modes;
- the baseline check that repeated native values agree across projection
  methods.

Missing columns or rows, duplicate or mismatched methods, conflicting model
metadata, and non-inverse datasets remain errors.

## Data flow

Pass one boolean from the CLI through `generate_table`,
`_generate_stage_table`, and `_load_direction`. Guard the two existing numeric
consistency checks with it. No new abstraction or dependency is needed.

## Verification

Add one regression test proving inconsistent projector-only values still fail
by default and succeed when the option is enabled. Run the focused test file
and the originally failing command with the new flag.
