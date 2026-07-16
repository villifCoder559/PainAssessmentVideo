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
  if any(isinstance(sample_id, bool) for sample_id in ids):
    raise ValueError(f'boolean IDs are invalid: {ids}')
  try:
    integer_ids = [int(sample_id) for sample_id in ids]
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError(f'non-integral IDs: {ids}') from error
  if any(sample_id != integer_id for sample_id, integer_id in zip(ids, integer_ids)):
    raise ValueError(f'non-integral IDs: {ids}')
  return integer_ids


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
    try:
      shift = helper.get_shift_for_sample_id(augmentation_suffix(original, sibling))
      shift_error = None
    except Exception as error:
      shift = None
      shift_error = error
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
        if shift_error is not None:
          raise ValueError(f'could not determine augmentation shift: {shift_error}')
        ids = read_sample_ids(sibling / relative_path)
        reference_id = original_ids[relative_path]
        if reference_id is None:
          raise ValueError('matched original file has an invalid sample ID')
        expected = reference_id + shift
        if any(sample_id != expected for sample_id in ids):
          raise ValueError(f'expected only {expected}, found {sorted(set(ids))}')
        completed += 1
      except Exception as error:
        if shift_error is not None:
          kind = 'invalid_augmentation'
        else:
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
  else:
    print(f'Failure log: {log_path}')

  print(f'Summary: tot_completed={completed} tot_fails={len(failures)}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
