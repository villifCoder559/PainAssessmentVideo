"""
check_augmentation_completeness.py

Verifies that every augmented sibling folder of a reference feature folder
contains all the same .safetensors files as the original.

Augmented sibling folders are detected by name prefix:
  {original_name}_<suffix>   e.g. spatial_pooled_features_UNBC_G_last143_hflip
  {original_name}$<N>        e.g. spatial_pooled_features_UNBC_G_last143$0

Usage:
  python3 check_augmentation_completeness.py <original_folder> [-v] [--strict]
"""

import argparse
import sys
from pathlib import Path


def collect_safetensors(folder: Path) -> set[str]:
  """
  Recursively collect all .safetensors file paths relative to folder.

  Args:
    folder: Root directory to search.

  Returns:
    Set of relative path strings (using forward slashes).
  """
  return {p.relative_to(folder).as_posix() for p in folder.rglob("*.safetensors")}


def find_augmented_siblings(original: Path) -> list[Path]:
  """
  Find sibling directories whose name extends the original folder name.

  A sibling qualifies if its name starts with '{original.name}_' or
  '{original.name}$'.

  Args:
    original: Path to the original feature folder.

  Returns:
    Sorted list of matching sibling directories.
  """
  parent = original.parent
  base = original.name
  siblings = []
  for entry in sorted(parent.iterdir()):
    if not entry.is_dir() or entry == original:
      continue
    name = entry.name
    if name.startswith(base + "_") or name.startswith(base + "$"):
      siblings.append(entry)
  return siblings


def main() -> int:
  """
  Entry point. Parses arguments, compares safetensors sets, and prints results.

  Returns:
    0 if all augmented folders are complete, 1 if any are incomplete or missing.
  """
  parser = argparse.ArgumentParser(
    description="Check that augmented feature folders contain all safetensors from the original."
  )
  parser.add_argument("--original_folder", type=Path, help="Path to the original feature folder.")
  parser.add_argument(
    "-v", "--verbose",
    action="store_true",
    help="Print each missing (and extra) filename instead of just counts.",
  )
  parser.add_argument(
    "--strict",
    action="store_true",
    help="Also flag augmented folders that contain extra files not in the original.",
  )
  args = parser.parse_args()

  original: Path = args.original_folder.resolve()

  if not original.is_dir():
    print(f"ERROR: '{original}' is not a directory.", file=sys.stderr)
    return 1

  print(f"Scanning original: {original}")
  original_files = collect_safetensors(original)
  print(f"Original : {original.name}  ({len(original_files)} files)\n")

  siblings = find_augmented_siblings(original)
  if not siblings:
    print("No augmented folders found.")
    return 0

  any_incomplete = False

  for sibling in siblings:
    sibling_files = collect_safetensors(sibling)
    missing = sorted(original_files - sibling_files)
    extra = sorted(sibling_files - original_files) if args.strict else []

    is_ok = (len(missing) == 0) and (len(extra) == 0)
    if not is_ok:
      any_incomplete = True

    tag = "[OK ]" if is_ok else "[MISS]"
    count_str = f"({len(sibling_files)}/{len(original_files)})"

    line = f"{tag}  {sibling.name:<70}  {count_str}"
    if missing:
      line += f"  {len(missing)} missing"
    if extra:
      line += f"  {len(extra)} extra"
    print(line)

    if args.verbose:
      for f in missing:
        print(f"         MISSING  {f}")
      for f in extra:
        print(f"         EXTRA    {f}")

  return 1 if any_incomplete else 0


if __name__ == "__main__":
  sys.exit(main())
