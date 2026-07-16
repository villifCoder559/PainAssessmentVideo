# Augmentation Sample-ID Validation Design

## Goal

Extend `check_augmentation_completeness.py` so it checks the stored
`list_sample_id` in original and augmented safetensors while preserving a
complete, non-stopping audit of all folders.

## Validation

The script initializes `custom.helper.step_shift` from `--original_folder`.
It reads only `list_sample_id` from each safetensor.

An original file is valid when `list_sample_id` exists, is non-empty, contains
one repeated integer value, and that value is in `1..step_shift`.

For an augmented sibling, the script removes the original folder prefix and an
optional `$N` variant suffix, then obtains the shift with
`helper.get_shift_for_sample_id()`. A matched augmented file is valid when all
its IDs equal the matched original ID plus that shift. A file whose original
reference is invalid cannot be validated and is therefore also a failure.

The existing relative-path comparison remains the pairing mechanism. Every
expected file missing from an augmented folder and every augmented file with
no original counterpart is a failure.

## Reporting and Exit Behavior

Every problem is printed as a warning and written as one tab-separated line to
`<original_folder_name>_unmatched_augmented.txt` beside the original folder.
Despite the historical filename, this file records all failure types, their
folder and relative path, and a short reason. It is overwritten on every run,
including successful runs, so it never contains stale results.

The final console summary prints:

- `tot_completed`: original and augmented files that passed validation.
- `tot_fails`: invalid or unreadable files, unmatched augmented files, and
  expected-but-missing files. Each encountered file or missing expected file
  is counted once.

Validation problems never interrupt the scan and the completed audit exits
with status 0. An invalid `--original_folder` is still a command error and
exits with status 1 because no scan can be performed.

## Error Handling

Missing or empty `list_sample_id`, inconsistent IDs within one tensor, invalid
original ranges, wrong augmented shifts, and safetensor read errors are logged
and scanning continues. Failure details are always saved; `--verbose` retains
its role of printing individual path details to the console.

## Test

Add one focused test using temporary safetensors. It covers a valid original
and augmentation, a wrong shifted ID, an invalid original, a `$N` augmentation
variant, an unmatched augmented file, a missing expected file, the summary
counts, log contents, continued scanning, and exit status 0.
