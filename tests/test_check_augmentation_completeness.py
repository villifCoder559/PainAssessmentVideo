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
