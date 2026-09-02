"""Discovery and validation for user-provided train/val/test CSV splits."""

import csv
from itertools import combinations
from pathlib import Path


REQUIRED_COLUMNS = (
  'subject_id',
  'subject_name',
  'class_id',
  'class_name',
  'sample_id',
  'sample_name',
)
SPLIT_ROLES = ('train', 'val', 'test')


def _read_subject_ids(csv_path):
  with csv_path.open(newline='', encoding='utf-8') as csv_file:
    reader = csv.DictReader(csv_file, delimiter='\t')
    fieldnames = reader.fieldnames or []
    missing_columns = [
      column for column in REQUIRED_COLUMNS if column not in fieldnames
    ]
    if missing_columns:
      raise ValueError(
        f"CSV '{csv_path}' is missing required columns: "
        f"{', '.join(missing_columns)}"
      )

    rows = list(reader)
    if not rows:
      raise ValueError(f"CSV '{csv_path}' must contain at least one row")

  return {row['subject_id'] for row in rows}


def discover_predefined_csv_splits(csv_path, subject_independent=True):
  """Return predefined split paths when ``csv_path`` is a directory.

  A regular file selects the legacy generated cross-validation path and
  returns ``None``. Directory mode requires exactly three immediate CSV files,
  one matching each split role by case-insensitive filename substring.
  """
  csv_path = Path(csv_path)
  if csv_path.is_file():
    return None
  if not csv_path.exists():
    raise FileNotFoundError(f"CSV path does not exist: {csv_path}")
  if not csv_path.is_dir():
    raise ValueError(f"CSV path must be a file or directory: {csv_path}")

  csv_files = sorted(
    path
    for path in csv_path.iterdir()
    if path.is_file() and path.suffix.lower() == '.csv'
  )
  if len(csv_files) != 3:
    raise ValueError(
      f"Predefined split directory '{csv_path}' must contain exactly 3 "
      f"immediate CSV files; found {len(csv_files)}"
    )

  split_paths = {}
  for path in csv_files:
    matches = [role for role in SPLIT_ROLES if role in path.name.lower()]
    if len(matches) > 1:
      raise ValueError(
        f"CSV filename '{path.name}' matches multiple split roles: "
        f"{', '.join(matches)}"
      )
    if not matches:
      raise ValueError(
        f"CSV filename '{path.name}' must contain train, val, or test"
      )

    role = matches[0]
    if role in split_paths:
      raise ValueError(
        f"Multiple CSV files match the '{role}' split role: "
        f"'{Path(split_paths[role]).name}' and '{path.name}'"
      )
    split_paths[role] = str(path)

  missing_roles = [role for role in SPLIT_ROLES if role not in split_paths]
  if missing_roles:
    raise ValueError(
      f"Predefined split directory '{csv_path}' is missing split roles: "
      f"{', '.join(missing_roles)}"
    )

  subject_ids = {
    role: _read_subject_ids(Path(path)) for role, path in split_paths.items()
  }
  if subject_independent:
    for first_role, second_role in combinations(SPLIT_ROLES, 2):
      overlap = subject_ids[first_role] & subject_ids[second_role]
      if overlap:
        overlap_text = ', '.join(sorted(overlap))
        raise ValueError(
          f"subject_id overlap between {first_role} and {second_role}: "
          f"{overlap_text}"
        )

  return {role: split_paths[role] for role in SPLIT_ROLES}


def configure_csv_input(config):
  """Resolve ``config['csv']`` and configure predefined directory mode."""
  split_paths = discover_predefined_csv_splits(
    config['csv'],
    subject_independent=bool(config.get('is_subject_independent', 1)),
  )
  config['predefined_csv_splits'] = split_paths
  config['training_csv'] = (
    split_paths['train'] if split_paths is not None else config['csv']
  )
  if split_paths is None:
    return None

  conflicting_flags = []
  if not config.get('validation_enabled', 1):
    conflicting_flags.append('--validation_enabled 0')
  if config.get('skip_test', 0):
    conflicting_flags.append('--skip_test 1')
  if config.get('use_test_as_val', 0):
    conflicting_flags.append('--use_test_as_val 1')
  if conflicting_flags:
    raise ValueError(
      'Predefined CSV directory mode requires the supplied train/val/test '
      'roles to remain enabled; incompatible options: '
      + ', '.join(conflicting_flags)
    )

  if config.get('stop') != [1, 1]:
    print(
      f"Predefined CSV splits detected at {config['csv']}. "
      "Forcing --stop 1 1 to disable cross-validation loops."
    )
  config['stop'] = [1, 1]
  return split_paths
