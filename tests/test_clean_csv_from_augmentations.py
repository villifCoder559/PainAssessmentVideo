import threading
from pathlib import Path
import os
import sys

import pandas as pd
import pytest
import torch

# Match the repository's standalone-test import convention.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import custom.helper as helper
import cross_space_projection as csp
from log_cross_attention_from_model import clean_csv_from_augmentations


def _write_source_csv(path):
  pd.DataFrame({
    'subject_id': [1, 1, 1],
    'subject_name': ['s1', 's1', 's1'],
    'class_id': [0, 1, 2],
    'class_name': ['c0', 'c1', 'c2'],
    'sample_id': [1, 2, 3],
    'sample_name': ['a', 'b', 'augmented'],
  }).to_csv(path, index=False, sep='\t')


def test_unique_cleaned_csv_isolated_from_persistent_output(tmp_path, monkeypatch):
  source = tmp_path / 'val.csv'
  _write_source_csv(source)
  monkeypatch.setattr(helper, 'step_shift', 2)

  cleaned = Path(clean_csv_from_augmentations(str(source), unique=True))
  try:
    assert cleaned != tmp_path / 'val_cleaned.csv'
    assert cleaned.parent == tmp_path
    assert pd.read_csv(cleaned, sep='\t')['sample_id'].tolist() == [1, 2]
  finally:
    cleaned.unlink(missing_ok=True)


def test_concurrent_unique_cleaners_return_distinct_complete_files(tmp_path, monkeypatch):
  source = tmp_path / 'val.csv'
  _write_source_csv(source)
  monkeypatch.setattr(helper, 'step_shift', 2)
  start = threading.Barrier(3)
  cleaned_paths = []
  writer_errors = []

  def clean_in_thread():
    try:
      start.wait(timeout=5)
      cleaned_paths.append(Path(clean_csv_from_augmentations(str(source), unique=True)))
    except BaseException as exc:
      writer_errors.append(exc)

  writers = [threading.Thread(target=clean_in_thread) for _ in range(2)]
  for writer in writers:
    writer.start()
  start.wait(timeout=5)
  for writer in writers:
    writer.join(timeout=5)

  try:
    assert not writer_errors
    assert len(cleaned_paths) == 2
    assert cleaned_paths[0] != cleaned_paths[1]
    for cleaned in cleaned_paths:
      assert pd.read_csv(cleaned, sep='\t')['sample_id'].tolist() == [1, 2]
  finally:
    for cleaned in cleaned_paths:
      cleaned.unlink(missing_ok=True)


def test_persistent_cleaned_csv_never_exposes_truncated_write(tmp_path, monkeypatch):
  source = tmp_path / 'val.csv'
  cleaned = tmp_path / 'val_cleaned.csv'
  _write_source_csv(source)
  _write_source_csv(cleaned)
  monkeypatch.setattr(helper, 'step_shift', 2)

  write_started = threading.Event()
  allow_write = threading.Event()
  writer_errors = []
  original_to_csv = pd.DataFrame.to_csv

  def delayed_to_csv(df, path, *args, **kwargs):
    Path(path).write_bytes(b'')
    write_started.set()
    if not allow_write.wait(timeout=5):
      raise TimeoutError('test did not release delayed CSV writer')
    return original_to_csv(df, path, *args, **kwargs)

  monkeypatch.setattr(pd.DataFrame, 'to_csv', delayed_to_csv)

  def write_cleaned_csv():
    try:
      clean_csv_from_augmentations(str(source))
    except BaseException as exc:  # propagate thread failures through the test thread
      writer_errors.append(exc)

  writer = threading.Thread(target=write_cleaned_csv)
  writer.start()
  assert write_started.wait(timeout=5)
  try:
    observed = pd.read_csv(cleaned, sep='\t')
    assert observed['sample_id'].tolist() == [1, 2, 3]
  finally:
    allow_write.set()
    writer.join(timeout=5)

  assert not writer.is_alive()
  assert not writer_errors
  assert pd.read_csv(cleaned, sep='\t')['sample_id'].tolist() == [1, 2]


def test_failed_persistent_write_preserves_previous_csv(tmp_path, monkeypatch):
  source = tmp_path / 'val.csv'
  cleaned = tmp_path / 'val_cleaned.csv'
  _write_source_csv(source)
  _write_source_csv(cleaned)
  monkeypatch.setattr(helper, 'step_shift', 2)
  files_before = set(tmp_path.iterdir())

  def failing_to_csv(df, path, *args, **kwargs):
    Path(path).write_bytes(b'partial')
    raise OSError('simulated write failure')

  monkeypatch.setattr(pd.DataFrame, 'to_csv', failing_to_csv)

  with pytest.raises(OSError, match='simulated write failure'):
    clean_csv_from_augmentations(str(source))

  assert pd.read_csv(cleaned, sep='\t')['sample_id'].tolist() == [1, 2, 3]
  assert set(tmp_path.iterdir()) == files_before


def test_empty_source_csv_reports_its_path(tmp_path, monkeypatch):
  source = tmp_path / 'val.csv'
  source.write_bytes(b'')
  monkeypatch.setattr(helper, 'step_shift', 2)

  with pytest.raises(ValueError, match=r'Source CSV is empty or has no columns: .*val\.csv'):
    clean_csv_from_augmentations(str(source))


def test_source_csv_requires_sample_id_column(tmp_path, monkeypatch):
  source = tmp_path / 'val.csv'
  pd.DataFrame({'sample_name': ['a']}).to_csv(source, index=False, sep='\t')
  monkeypatch.setattr(helper, 'step_shift', 2)

  with pytest.raises(ValueError, match=r"Source CSV .*val\.csv.*missing required column 'sample_id'"):
    clean_csv_from_augmentations(str(source))


def test_source_csv_requires_at_least_one_original_sample(tmp_path, monkeypatch):
  source = tmp_path / 'val.csv'
  pd.DataFrame({'sample_id': [3], 'sample_name': ['augmented']}).to_csv(
    source, index=False, sep='\t',
  )
  monkeypatch.setattr(helper, 'step_shift', 2)

  with pytest.raises(
    ValueError,
    match=r'Source CSV .*val\.csv.*contains no original samples.*step_shift=2',
  ):
    clean_csv_from_augmentations(str(source))


class _EmbeddingModelStub:
  def __init__(self, fail=False):
    self.path_to_extracted_features = 'UNBC/features'
    self.dataset_type = 'stub'
    self.fail = fail

  def test_pretrained_model(self, **kwargs):
    assert Path(kwargs['csv_path']).is_file()
    if self.fail:
      raise RuntimeError('simulated inference failure')
    helper.LOG_VIDEO_EMBEDDINGS['embeddings'].append(torch.tensor([[1.0, 2.0]]))
    helper.LOG_VIDEO_EMBEDDINGS['labels'].append(1.0)
    helper.LOG_VIDEO_EMBEDDINGS['sample_ids'].append(7)
    helper.LOG_VIDEO_EMBEDDINGS['predictions'].append(0.75)


def _embedding_config():
  return {'config': {
    'criterion': None,
    'concatenate_temp_dim': False,
    'concatenate_quadrants': False,
    'CCC_loss': False,
  }}


def test_extract_embeddings_removes_unique_cleaned_csv_after_success(tmp_path, monkeypatch):
  private_csv = tmp_path / '.val_cleaned_private.csv'
  unique_requests = []

  def fake_cleaner(csv_path, *, unique=False):
    unique_requests.append(unique)
    private_csv.write_text('sample_id\n7\n')
    return str(private_csv)

  monkeypatch.setattr(csp, 'clean_csv_from_augmentations', fake_cleaner)

  result = csp._extract_embeddings(
    _EmbeddingModelStub(), 'model.pt', 'val.csv', _embedding_config(),
  )

  assert unique_requests == [True]
  assert not private_csv.exists()
  assert result['embeddings'].tolist() == [[1.0, 2.0]]
  assert result['sample_ids'].tolist() == [7]


def test_extract_embeddings_removes_unique_cleaned_csv_after_failure(tmp_path, monkeypatch):
  private_csv = tmp_path / '.val_cleaned_private.csv'
  unique_requests = []

  def fake_cleaner(csv_path, *, unique=False):
    unique_requests.append(unique)
    private_csv.write_text('sample_id\n7\n')
    return str(private_csv)

  monkeypatch.setattr(csp, 'clean_csv_from_augmentations', fake_cleaner)

  with pytest.raises(RuntimeError, match='simulated inference failure'):
    csp._extract_embeddings(
      _EmbeddingModelStub(fail=True), 'model.pt', 'val.csv', _embedding_config(),
    )

  assert unique_requests == [True]
  assert not private_csv.exists()
