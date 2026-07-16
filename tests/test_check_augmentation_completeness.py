import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

import check_augmentation_completeness as checker


def _write_ids(path, ids, dtype=np.int32):
  path.parent.mkdir(parents=True, exist_ok=True)
  save_file({'list_sample_id': np.asarray(ids, dtype=dtype)}, path)


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


def test_invalid_augmentation_continues_file_and_sibling_audit(tmp_path, monkeypatch, capsys):
  original = tmp_path / 'features_UNBC'
  unsupported = tmp_path / 'features_UNBC_hflip_zoom'
  later = tmp_path / 'features_UNBC_jitter'

  _write_ids(original / 'matched.safetensors', [3])
  _write_ids(original / 'missing.safetensors', [4])
  _write_ids(unsupported / 'matched.safetensors', [203])
  _write_ids(unsupported / 'extra.safetensors', [205])
  _write_ids(later / 'matched.safetensors', [403])

  monkeypatch.setattr(sys, 'argv', [
    'check_augmentation_completeness.py',
    '--original_folder', str(original),
  ])

  assert checker.main() == 0

  output = capsys.readouterr().out
  assert '[DONE] features_UNBC_hflip_zoom' in output
  assert '[DONE] features_UNBC_jitter' in output
  assert 'tot_completed=3' in output
  assert 'tot_fails=4' in output

  lines = (tmp_path / 'features_UNBC_unmatched_augmented.txt').read_text(
    encoding='utf-8',
  ).splitlines()
  assert any(
    'invalid_augmentation' in line
    and 'features_UNBC_hflip_zoom' in line
    and 'matched.safetensors' in line
    and 'Mixed augmentation not recognized' in line
    for line in lines
  )
  assert any(
    'missing_expected' in line
    and 'features_UNBC_hflip_zoom' in line
    and 'missing.safetensors' in line
    for line in lines
  )
  assert any(
    'unmatched_augmented' in line
    and 'features_UNBC_hflip_zoom' in line
    and 'extra.safetensors' in line
    for line in lines
  )


def test_read_sample_ids_rejects_fractional_values(tmp_path):
  path = tmp_path / 'fractional.safetensors'
  _write_ids(path, [3.5], dtype=np.float32)

  with pytest.raises(ValueError, match='non-integral'):
    checker.read_sample_ids(path)


def test_read_sample_ids_rejects_boolean_values(tmp_path):
  path = tmp_path / 'boolean.safetensors'
  _write_ids(path, [True], dtype=np.bool_)

  with pytest.raises(ValueError, match='boolean'):
    checker.read_sample_ids(path)


def test_does_not_claim_failure_log_when_write_fails(tmp_path, monkeypatch, capsys):
  original = tmp_path / 'features_UNBC'
  _write_ids(original / 'valid.safetensors', [3])

  def fail_write(*args, **kwargs):
    raise OSError('disk full')

  monkeypatch.setattr(Path, 'write_text', fail_write)
  monkeypatch.setattr(sys, 'argv', [
    'check_augmentation_completeness.py',
    '--original_folder', str(original),
  ])

  assert checker.main() == 0

  output = capsys.readouterr().out
  assert 'WARNING: could not write failure log' in output
  assert 'Failure log:' not in output
