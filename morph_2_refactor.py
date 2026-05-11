"""
morph_2_refactor.py

Converts the MORPH-2 dataset from its Kaggle split layout into the subject-centric
structure expected by the BioVid pipeline:

  CSV output : MORPH_2/samples.csv
               columns: subject_id, subject_name, class_id, class_name, sample_id, sample_name

  Image output: MORPH_2/images_new/{subject_name}/{filename}
                (original Train/Validation/Test folders are left untouched)

Age is parsed from the filename (e.g. '18' from '00189_01M18.JPG') rather than
using the offset-encoded value in the index CSVs.
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


FILENAME_RE = re.compile(
  r'^([^_]+)_(\d+)([MFmf])(\d+)\.(JPG|jpg|jpeg|JPEG)$'
)

SPLITS = ['Train', 'Validation', 'Test']


def parse_args() -> argparse.Namespace:
  """
  Parse command-line arguments.

  Returns:
    Parsed namespace with morph_root, out_csv, out_images, dry_run.
  """
  p = argparse.ArgumentParser(description='Refactor MORPH-2 dataset layout.')
  p.add_argument(
    '--morph_root',
    default='MORPH_2',
    help='Path to the MORPH_2 root directory (default: MORPH_2).',
  )
  p.add_argument(
    '--out_csv',
    default=None,
    help='Output CSV path (default: {morph_root}/samples.csv).',
  )
  p.add_argument(
    '--out_images',
    default=None,
    help='Output image root (default: {morph_root}/images_new/).',
  )
  p.add_argument(
    '--dry_run',
    action='store_true',
    help='Print what would happen without writing any files.',
  )
  p.add_argument(
    '--check_only',
    action='store_true',
    help='Load out_csv and run validation only; skip generation and image copy.',
  )
  p.add_argument(
    '--zero_stripping',
    action='store_true',
    default=False,
    help='Strip leading zeros from subject_name (e.g. "00013" -> "13"). '
         'Warning: different subjects sharing a numeric root will be merged.',
  )
  return p.parse_args()


def load_and_merge_csvs(morph_root: Path) -> pd.DataFrame:
  """
  Read Train.csv, Validation.csv, Test.csv from {morph_root}/Index/ and merge.

  Each row gets a 'source_split' column ('Train', 'Validation', 'Test') that
  identifies where the image currently lives on disk.

  Args:
    morph_root: Path to the MORPH_2 root directory.

  Returns:
    Concatenated DataFrame with columns: age, gender, filename, source_split.
  """
  frames = []
  for split in SPLITS:
    csv_path = morph_root / 'Index' / f'{split}.csv'
    if not csv_path.exists():
      raise FileNotFoundError(f'Missing index CSV: {csv_path}')
    df = pd.read_csv(csv_path)
    df['source_split'] = split
    frames.append(df)
    print(f'  Loaded {len(df):,} rows from {csv_path.name}')

  merged = pd.concat(frames, ignore_index=True)
  merged = merged.drop(columns=['filepath'], errors='ignore')
  print(f'  Total rows merged: {len(merged):,}')
  return merged


def parse_filename_fields(df: pd.DataFrame, zero_stripping: bool = False) -> pd.DataFrame:
  """
  Extract subject_name, session, gender_char, and age_from_filename from the
  filename column using a regex.

  Rows that do not match the expected pattern are flagged and printed as
  warnings; they are kept in the DataFrame with NaN fields so the caller can
  decide how to handle them.

  Args:
    df:             DataFrame with a 'filename' column.
    zero_stripping: If True, strip leading zeros from subject_name
                    (e.g. '00013' -> '13'). Can merge distinct subjects
                    that share a numeric root — use with caution.

  Returns:
    DataFrame with added columns: subject_name, session, gender_char,
    age_from_filename, sample_name.
  """
  parsed = df['filename'].str.extract(FILENAME_RE)
  parsed.columns = ['subject_name', 'session', 'gender_char', 'age_from_filename', '_ext']
  parsed = parsed.drop(columns=['_ext'])
  parsed['age_from_filename'] = pd.to_numeric(parsed['age_from_filename'], errors='coerce')
  if zero_stripping:
    parsed['subject_name'] = parsed['subject_name'].apply(
      lambda x: str(int(x)) if pd.notna(x) else x
    )
    print('  Zero-stripping enabled: leading zeros removed from subject_name')

  parsed['subject_name'] = parsed['subject_name'] + '_' + parsed['gender_char'].str.upper()

  bad_mask = parsed['subject_name'].isna()
  if bad_mask.any():
    print(f'\n  WARNING: {bad_mask.sum()} filename(s) did not match expected pattern:')
    for fn in df.loc[bad_mask, 'filename'].tolist():
      print(f'    {fn}')

  df = pd.concat([df.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)

  df['sample_name'] = df['filename'].str.rsplit('.', n=1).str[0]

  age_diff = df['age_from_filename'] - df['age']
  print(
    f'\n  Age offset (filename_age - csv_age): '
    f'min={age_diff.min():.0f}, max={age_diff.max():.0f}, '
    f'mean={age_diff.mean():.2f}, '
    f'unique values={sorted(age_diff.dropna().unique().astype(int).tolist())}'
  )

  return df


def assign_subject_ids(df: pd.DataFrame) -> pd.DataFrame:
  """
  Assign a 1-based sequential subject_id to each unique subject_name, sorted
  alphabetically.  All rows that share the same subject_name receive the same
  subject_id.

  Args:
    df: DataFrame with a 'subject_name' column.

  Returns:
    DataFrame with an added 'subject_id' column.
  """
  unique_subjects = sorted(df['subject_name'].dropna().unique(), key=lambda x: int(x.split('_')[0]))
  subject_map = {name: idx + 1 for idx, name in enumerate(unique_subjects)}
  df['subject_id'] = df['subject_name'].map(subject_map)
  print(f'\n  Unique subjects: {len(unique_subjects):,}')
  return df


def build_final_df(df: pd.DataFrame) -> pd.DataFrame:
  """
  Sort rows deterministically, assign global sample_ids, validate uniqueness,
  and return a DataFrame with exactly the columns required by the pipeline CSV:
  subject_id, subject_name, class_id, class_name, sample_id, sample_name.

  Args:
    df: DataFrame after parse_filename_fields and assign_subject_ids.

  Returns:
    Final DataFrame with the six pipeline columns, sorted by subject_name then
    sample_name.

  Raises:
    AssertionError: If sample_name values are not globally unique.
  """
  df = df.sort_values(['subject_name', 'sample_name'], ignore_index=True)

  dupes = df['sample_name'].duplicated()
  if dupes.any():
    dup_names = df.loc[dupes, 'sample_name'].tolist()
    raise AssertionError(
      f'sample_name is not globally unique — duplicates found:\n  ' +
      '\n  '.join(dup_names[:20])
    )

  df['sample_id'] = range(1, len(df) + 1)
  df['class_id'] = df['age_from_filename'].astype(int)
  df['class_name'] = df['class_id'].astype(str)

  return df[['subject_id', 'subject_name', 'class_id', 'class_name', 'sample_id', 'sample_name']]


def validate_csv(df: pd.DataFrame) -> bool:
  """
  Validate the pipeline CSV for internal consistency.

  Three checks are performed:
    A) sample_name global uniqueness (error if violated).
    B) Per-subject gender consistency — all samples of a subject must share the
       same gender character parsed from sample_name (error if violated).
    C) Per-subject age variety — subjects with only one unique age are flagged as
       a warning (does not affect the return value).

  Args:
    df: DataFrame with columns subject_id, subject_name, class_id, sample_name.
        Works on both the in-memory final_df and a CSV loaded from disk.

  Returns:
    True if all error-level checks pass, False otherwise.
  """
  print('\n--- CSV Validation ---')
  ok = True

  # A — sample_name uniqueness
  dupes = df['sample_name'].duplicated(keep=False)
  if dupes.any():
    dup_list = ', '.join(df.loc[dupes, 'sample_name'].unique()[:10].tolist())
    print(f'  [FAIL] sample_name: {dupes.sum():,} duplicate row(s) — e.g. {dup_list}')
    ok = False
  else:
    print(f'  [OK]  sample_name: {len(df):,} unique values, no duplicates')

  # B — parse gender from sample_name
  gender_parsed = (df['sample_name'] + '.JPG').str.extract(FILENAME_RE)
  gender_parsed.columns = ['_s', '_sess', 'gender_char', '_age', '_ext']
  df = df.copy()
  df['gender_char'] = gender_parsed['gender_char'].str.upper()

  unparsed = df['gender_char'].isna().sum()
  if unparsed:
    print(f'  [WARN] {unparsed:,} sample_name(s) could not be parsed for gender (skipped)')

  parseable = df.dropna(subset=['gender_char'])
  gender_nunique = parseable.groupby('subject_id')['gender_char'].nunique()
  bad_subjects = gender_nunique[gender_nunique > 1]

  if bad_subjects.empty:
    n_subjects = parseable['subject_id'].nunique()
    print(f'  [OK]  Gender: all {n_subjects:,} subjects have consistent gender')
  else:
    print(f'  [FAIL] Gender inconsistency: {len(bad_subjects):,} subject(s) affected')
    for sid in bad_subjects.index[:10]:
      rows = parseable[parseable['subject_id'] == sid]
      sname = rows['subject_name'].iloc[0]
      genders = rows['gender_char'].unique().tolist()
      examples = rows['sample_name'].tolist()[:4]
      print(f'         subject_id={sid} (name={sname}): genders={genders}  e.g. {examples}')
    ok = False

  # C — age variety (informational only)
  age_nunique = df.groupby('subject_id')['class_id'].nunique()
  mono = age_nunique[age_nunique == 1]
  if mono.empty:
    print(f'  [OK]  Age variety: all subjects have >1 unique age')
  else:
    examples = df[df['subject_id'].isin(mono.index[:10])]['subject_name'].unique()[:10].tolist()
    print(f'  [WARN] Age variety: {len(mono):,} subject(s) have only one unique age value')
    print(f'         Examples: {", ".join(examples)}')

  print(f'\n  Validation {"passed" if ok else "FAILED"}.')
  return ok


def write_csv(final_df: pd.DataFrame, out_csv: Path, dry_run: bool) -> None:
  """
  Write the final pipeline CSV to disk.

  Args:
    final_df: DataFrame with pipeline columns.
    out_csv:  Destination path.
    dry_run:  If True, print a preview instead of writing.
  """
  if dry_run:
    print(f'\n[DRY RUN] Would write CSV to: {out_csv}')
    print(f'  Shape: {final_df.shape}')
    print(final_df.head(10).to_string(index=False))
    return

  out_csv.parent.mkdir(parents=True, exist_ok=True)
  final_df.to_csv(out_csv, index=False)
  print(f'\n  CSV written: {out_csv}  ({len(final_df):,} rows)')


def copy_images(
  df_full: pd.DataFrame,
  morph_root: Path,
  out_images: Path,
  dry_run: bool,
) -> None:
  """
  Copy each image from its original split folder to the subject-centric output
  tree: {out_images}/{subject_name}/{filename}.

  Args:
    df_full:    Full DataFrame (before column selection) containing source_split,
                subject_name, and filename columns.
    morph_root: Path to the MORPH_2 root directory.
    out_images: Root of the new image tree (MORPH_2/images_new/).
    dry_run:    If True, count and describe copies without performing them.
  """
  copied = skipped = missing = 0

  rows = list(df_full[['source_split', 'subject_name', 'filename']].itertuples(index=False))
  progress = tqdm(rows, unit='img', desc='Copying images', disable=dry_run)

  for row in progress:
    src = morph_root / 'Images' / row.source_split / row.filename
    dst_dir = out_images / row.subject_name
    dst = dst_dir / row.filename

    if not src.exists():
      missing += 1
      if missing <= 10:
        tqdm.write(f'  WARNING: source file not found: {src}')
      continue

    if dry_run:
      copied += 1
      continue

    if dst.exists():
      skipped += 1
      continue

    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied += 1

  action = 'Would copy' if dry_run else 'Copied'
  print(
    f'\n  {action}: {copied:,} | Skipped (already exists): {skipped:,} | '
    f'Missing src: {missing:,}'
  )


def main() -> None:
  """
  Entry point: orchestrates CSV merging, field parsing, ID assignment, CSV
  writing, and image copying.
  """
  args = parse_args()
  morph_root = Path(args.morph_root)

  if not morph_root.exists():
    raise FileNotFoundError(f'morph_root not found: {morph_root}')

  out_csv = Path(args.out_csv) if args.out_csv else morph_root / 'samples.csv'
  out_images = Path(args.out_images) if args.out_images else morph_root / 'images_new'

  if args.check_only:
    if not out_csv.exists():
      raise FileNotFoundError(f'CSV not found for --check_only: {out_csv}')
    print(f'\n=== MORPH-2 CSV Validation ===')
    print(f'  csv : {out_csv.resolve()}')
    df_check = pd.read_csv(out_csv)
    ok = validate_csv(df_check)
    sys.exit(0 if ok else 1)

  print(f'\n=== MORPH-2 Refactor {"(DRY RUN) " if args.dry_run else ""}===')
  print(f'  morph_root : {morph_root.resolve()}')
  print(f'  out_csv    : {out_csv.resolve()}')
  print(f'  out_images : {out_images.resolve()}')

  print('\n--- Step 1: Load and merge CSVs ---')
  df = load_and_merge_csvs(morph_root)

  print('\n--- Step 2: Parse filename fields ---')
  df = parse_filename_fields(df, zero_stripping=args.zero_stripping)

  print('\n--- Step 3: Assign subject IDs ---')
  df = assign_subject_ids(df)

  print('\n--- Step 4 & 5: Build final columns and write CSV ---')
  final_df = build_final_df(df)
  write_csv(final_df, out_csv, args.dry_run)
  validate_csv(final_df)

  print('\n--- Step 6: Copy images ---')
  copy_images(df, morph_root, out_images, args.dry_run)

  print('\n=== Done ===')


if __name__ == '__main__':
  main()
