# Augmentation Sample-ID Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `check_augmentation_completeness.py` audit original and augmented safetensor sample IDs, log every problem, finish the complete scan, print totals, and exit successfully.

**Architecture:** Keep relative file paths as the original-to-augmentation join key. Read only `list_sample_id`, derive each sibling's shift from `custom.helper`, and collect every warning in one list that is written to the agreed log and used for the final totals.

**Tech Stack:** Python standard library, `safetensors.safe_open`, NumPy-backed safetensor reads, existing `custom.helper`, pytest.

## Global Constraints

- Use `helper.set_step_shift()` and `helper.get_shift_for_sample_id()` as the only augmentation-ID rules.
- `$N` folders use the shift of the augmentation name before `$N`.
- Never stop scanning because of a safetensor validation problem.
- Always overwrite `<original_folder_name>_unmatched_augmented.txt` beside the original folder.
- A completed scan exits 0; an invalid or unrecognized original folder exits 1.
- Add no dependency and no new abstraction beyond the small helpers needed to avoid duplicated validation and logging.

---

### Task 1: Add the complete non-stopping sample-ID audit

**Files:**
- Create: `tests/test_check_augmentation_completeness.py`
- Modify: `check_augmentation_completeness.py:1-127`

**Interfaces:**
- Consumes: `helper.set_step_shift(folder_feature: str)`, `helper.get_shift_for_sample_id(folder_feature: str)`, and safetensors containing `list_sample_id`.
- Produces: unchanged CLI entry point `main() -> int`; log `<original_folder_name>_unmatched_augmented.txt`; final `tot_completed` and `tot_fails` console fields.

- [ ] **Step 1: Write the failing end-to-end test**

Create `tests/test_check_augmentation_completeness.py`:

```python
import sys

import numpy as np
from safetensors.numpy import save_file

import check_augmentation_completeness as checker


def _write_ids(path, ids):
  path.parent.mkdir(parents=True, exist_ok=True)
  save_file({'list_sample_id': np.asarray(ids, dtype=np.int32)}, path)


def test_logs_all_failures_continues_and_returns_success(tmp_path, monkeypatch, capsys):
  original = tmp_path / 'features_UNBC'
  augmented = tmp_path / 'features_UNBC_hflip$0'

  _write_ids(original / 'subject' / 'valid.safetensors', [3, 3])
  _write_ids(original / 'subject' / 'wrong_aug_source.safetensors', [4, 4])
  _write_ids(original / 'subject' / 'bad_original.safetensors', [5, 6])
  _write_ids(original / 'subject' / 'missing.safetensors', [7, 7])

  _write_ids(augmented / 'subject' / 'valid.safetensors', [203, 203])
  _write_ids(augmented / 'subject' / 'wrong_aug_source.safetensors', [999, 999])
  _write_ids(augmented / 'subject' / 'bad_original.safetensors', [205, 205])
  _write_ids(augmented / 'subject' / 'extra.safetensors', [208, 208])

  monkeypatch.setattr(sys, 'argv', [
    'check_augmentation_completeness.py',
    '--original_folder', str(original),
  ])

  assert checker.main() == 0

  output = capsys.readouterr().out
  assert 'WARNING:' in output
  assert 'tot_completed=4' in output
  assert 'tot_fails=5' in output

  log = tmp_path / 'features_UNBC_unmatched_augmented.txt'
  lines = log.read_text(encoding='utf-8').splitlines()
  assert len(lines) == 5
  assert any('invalid_original_id' in line and 'bad_original.safetensors' in line for line in lines)
  assert any('wrong_sample_id' in line and 'wrong_aug_source.safetensors' in line for line in lines)
  assert any('invalid_reference' in line and 'bad_original.safetensors' in line for line in lines)
  assert any('missing_expected' in line and 'missing.safetensors' in line for line in lines)
  assert any('unmatched_augmented' in line and 'extra.safetensors' in line for line in lines)
```

- [ ] **Step 2: Run the new test and confirm the old checker fails it**

Run:

```bash
pytest -q tests/test_check_augmentation_completeness.py
```

Expected: FAIL because the current checker returns 1 for the missing file and does not create the failure log or sample-ID totals.

- [ ] **Step 3: Implement the minimal audit in the checker**

Replace `check_augmentation_completeness.py` with:

```python
"""Audit augmented feature folders for file completeness and sample IDs.

Usage:
  python3 check_augmentation_completeness.py --original_folder PATH [-v] [--strict]
"""

import argparse
import sys
from pathlib import Path

from safetensors import safe_open

from custom import helper


def collect_safetensors(folder: Path) -> set[str]:
  """Return safetensor paths relative to folder."""
  return {p.relative_to(folder).as_posix() for p in folder.rglob('*.safetensors')}


def find_augmented_siblings(original: Path) -> list[Path]:
  """Return sibling directories extending the original folder name."""
  base = original.name
  return [
    entry for entry in sorted(original.parent.iterdir())
    if entry.is_dir()
    and entry != original
    and (entry.name.startswith(base + '_') or entry.name.startswith(base + '$'))
  ]


def read_sample_ids(path: Path) -> list[int]:
  """Read and flatten list_sample_id without loading feature tensors."""
  with safe_open(path, framework='numpy') as tensors:
    if 'list_sample_id' not in tensors.keys():
      raise ValueError("missing 'list_sample_id'")
    ids = tensors.get_tensor('list_sample_id').reshape(-1).tolist()
  if not ids:
    raise ValueError("empty 'list_sample_id'")
  return [int(sample_id) for sample_id in ids]


def original_sample_id(path: Path, step_shift: int) -> int:
  """Return a valid original ID or raise a concise validation error."""
  ids = read_sample_ids(path)
  if any(sample_id != ids[0] for sample_id in ids[1:]):
    raise ValueError(f'inconsistent IDs: {sorted(set(ids))}')
  if not 1 <= ids[0] <= step_shift:
    raise ValueError(f'ID {ids[0]} outside 1..{step_shift}')
  return ids[0]


def augmentation_suffix(original: Path, sibling: Path) -> str:
  """Return the augmentation name, excluding folder prefix and $N variant."""
  return sibling.name[len(original.name):].removeprefix('_').split('$', 1)[0]


def record_failure(failures, kind, folder, relative_path, reason, verbose):
  """Append one log line and print a non-stopping warning."""
  line = f'{kind}\t{folder}\t{relative_path}\t{reason}'
  failures.append(line)
  detail = f': {reason}' if verbose else ''
  print(f'WARNING: {kind}: {folder}/{relative_path}{detail}')


def main() -> int:
  parser = argparse.ArgumentParser(
    description='Audit augmented feature folders for completeness and sample IDs.'
  )
  parser.add_argument(
    '--original_folder', type=Path, required=True,
    help='Path to the original feature folder.',
  )
  parser.add_argument(
    '-v', '--verbose', action='store_true',
    help='Include failure reasons in console warnings.',
  )
  parser.add_argument(
    '--strict', action='store_true',
    help='Retained for CLI compatibility; unmatched files are always logged.',
  )
  args = parser.parse_args()
  original = args.original_folder.resolve()

  if not original.is_dir():
    print(f"ERROR: '{original}' is not a directory.", file=sys.stderr)
    return 1
  try:
    helper.set_step_shift(str(original))
  except ValueError as error:
    print(f'ERROR: {error}', file=sys.stderr)
    return 1

  original_files = collect_safetensors(original)
  siblings = find_augmented_siblings(original)
  log_path = original.parent / f'{original.name}_unmatched_augmented.txt'
  failures = []
  completed = 0
  original_ids = {}

  print(f'Scanning original: {original}')
  for relative_path in sorted(original_files):
    try:
      original_ids[relative_path] = original_sample_id(
        original / relative_path, helper.step_shift,
      )
      completed += 1
    except Exception as error:
      original_ids[relative_path] = None
      record_failure(
        failures, 'invalid_original_id', original.name,
        relative_path, str(error), args.verbose,
      )

  for sibling in siblings:
    sibling_files = collect_safetensors(sibling)
    shift = helper.get_shift_for_sample_id(augmentation_suffix(original, sibling))
    completed_before = completed
    failed_before = len(failures)

    for relative_path in sorted(original_files - sibling_files):
      record_failure(
        failures, 'missing_expected', sibling.name, relative_path,
        'file is present in the original folder only', args.verbose,
      )
    for relative_path in sorted(sibling_files - original_files):
      record_failure(
        failures, 'unmatched_augmented', sibling.name, relative_path,
        'no file at the same relative path in the original folder', args.verbose,
      )
    for relative_path in sorted(original_files & sibling_files):
      try:
        ids = read_sample_ids(sibling / relative_path)
        reference_id = original_ids[relative_path]
        if reference_id is None:
          raise ValueError('matched original file has an invalid sample ID')
        expected = reference_id + shift
        if any(sample_id != expected for sample_id in ids):
          raise ValueError(f'expected only {expected}, found {sorted(set(ids))}')
        completed += 1
      except Exception as error:
        kind = 'invalid_reference' if original_ids[relative_path] is None else 'wrong_sample_id'
        record_failure(
          failures, kind, sibling.name, relative_path, str(error), args.verbose,
        )

    print(
      f'[DONE] {sibling.name}  '
      f'completed={completed - completed_before} '
      f'fails={len(failures) - failed_before}'
    )

  try:
    log_path.write_text(
      '\n'.join(failures) + ('\n' if failures else ''), encoding='utf-8',
    )
  except OSError as error:
    print(f'WARNING: could not write failure log {log_path}: {error}')

  print(f'Failure log: {log_path}')
  print(f'Summary: tot_completed={completed} tot_fails={len(failures)}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
```

- [ ] **Step 4: Run the focused test and confirm it passes**

Run:

```bash
pytest -q tests/test_check_augmentation_completeness.py
```

Expected: `1 passed`.

- [ ] **Step 5: Exercise the real UNBC folder supplied for acceptance testing**

Run:

```bash
python3 check_augmentation_completeness.py --original_folder UNBC/video/features/DFER/spatial_pooled_features_UNBC_B_last143_stride16_interpol
```

Expected: every sibling reaches `[DONE]`, the final output contains both `Summary: tot_completed=` and `tot_fails=`, the process exits 0, and the reported log exists beside the original folder. Nonzero `tot_fails` is valid real-data audit output, not a test failure.

- [ ] **Step 6: Run final regression checks**

Run:

```bash
pytest -q tests/test_check_augmentation_completeness.py tests/test_variant_augmentation_folders.py
python3 check_augmentation_completeness.py --help
```

Expected: all tests pass and help lists required `--original_folder`, `--verbose`, and compatibility `--strict` options.

- [ ] **Step 7: Commit the implementation**

```bash
git add check_augmentation_completeness.py tests/test_check_augmentation_completeness.py
git commit -m "Check sample IDs in augmentation audit"
```
