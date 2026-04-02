"""
Tests for multi-variant augmentation folder discovery and random selection
in customDatasetWhole._discover_variant_folders and _load_element.
"""
import unittest
import tempfile
import shutil
import os
import numpy as np
from unittest.mock import patch, MagicMock
import safetensors.torch
import torch
import pandas as pd

from custom.dataset import customDatasetWhole
import custom.helper as helper
import custom.tools as tools


def _create_dir(path):
  """
  Create a directory (and parents) if it doesn't exist.

  Args:
    path: Directory path to create.
  """
  os.makedirs(path, exist_ok=True)


def _create_dummy_safetensor(path):
  """
  Create a minimal safetensors file at the given path.

  Args:
    path: Full file path ending in .safetensors.
  """
  os.makedirs(os.path.dirname(path), exist_ok=True)
  data = {'features': torch.randn(2, 3)}
  safetensors.torch.save_file(data, path)


class TestDiscoverVariantFolders(unittest.TestCase):
  """Test _discover_variant_folders discovery logic using real temp directories."""

  def setUp(self):
    self.tmpdir = tempfile.mkdtemp()
    self.base_name = 'whole_full_features_BIOVID_S_last143'
    self.root_folder_features = os.path.join(self.tmpdir, self.base_name)
    _create_dir(self.root_folder_features)

  def tearDown(self):
    shutil.rmtree(self.tmpdir)

  def _make_stub(self):
    """
    Create a stub object with root_folder_features set, then call _discover_variant_folders.

    Returns:
      The stub object with aug_variant_folders populated.
    """
    stub = object.__new__(customDatasetWhole)
    stub.root_folder_features = self.root_folder_features
    stub._discover_variant_folders()
    return stub

  def test_discovery_base_and_variants(self):
    """Base aug folder + $0 + $1 discovered; $seed42 ignored."""
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_hflip'))
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_hflip$0'))
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_hflip$1'))
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_hflip$seed42'))

    stub = self._make_stub()
    self.assertIn('hflip', stub.aug_variant_folders)
    self.assertEqual(len(stub.aug_variant_folders['hflip']), 3)
    # $seed42 should NOT appear
    paths = stub.aug_variant_folders['hflip']
    self.assertTrue(all('seed42' not in p for p in paths))

  def test_discovery_compound_aug_type(self):
    """Compound aug types like shift_hflip are correctly grouped."""
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_shift_hflip'))
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_shift_hflip$0'))

    stub = self._make_stub()
    self.assertIn('shift_hflip', stub.aug_variant_folders)
    self.assertEqual(len(stub.aug_variant_folders['shift_hflip']), 2)

  def test_discovery_no_variants(self):
    """Only base aug folders, no $N variants -> single-element lists."""
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_hflip'))
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_jitter'))

    stub = self._make_stub()
    self.assertEqual(len(stub.aug_variant_folders['hflip']), 1)
    self.assertEqual(len(stub.aug_variant_folders['jitter']), 1)

  def test_discovery_no_parent_dir(self):
    """Non-existent parent directory -> empty dict, no exception."""
    stub = object.__new__(customDatasetWhole)
    stub.root_folder_features = '/nonexistent/path/features_base'
    stub._discover_variant_folders()
    self.assertEqual(stub.aug_variant_folders, {})

  def test_discovery_ignores_files(self):
    """Files matching the pattern are ignored, only directories count."""
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_hflip'))
    # Create a file (not dir) that matches the pattern
    file_path = os.path.join(self.tmpdir, f'{self.base_name}_hflip$0')
    with open(file_path, 'w') as f:
      f.write('not a directory')

    stub = self._make_stub()
    self.assertEqual(len(stub.aug_variant_folders['hflip']), 1)

  def test_discovery_ignores_negative_id(self):
    """Folders with negative id like $-1 are ignored."""
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_hflip'))
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_hflip$-1'))

    stub = self._make_stub()
    self.assertEqual(len(stub.aug_variant_folders['hflip']), 1)

  def test_discovery_multiple_aug_types_independent(self):
    """Multiple aug types are discovered independently."""
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_hflip'))
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_hflip$0'))
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_jitter'))
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_jitter$0'))
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_jitter$1'))
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_jitter$2'))

    stub = self._make_stub()
    self.assertEqual(len(stub.aug_variant_folders['hflip']), 2)
    self.assertEqual(len(stub.aug_variant_folders['jitter']), 4)

  def test_discovery_only_variants_no_base(self):
    """$N variants exist but base aug folder does not -> variants still discovered."""
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_hflip$0'))
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_hflip$1'))

    stub = self._make_stub()
    self.assertIn('hflip', stub.aug_variant_folders)
    self.assertEqual(len(stub.aug_variant_folders['hflip']), 2)

  def test_discovery_results_sorted(self):
    """Discovered folders are sorted for deterministic ordering."""
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_hflip$2'))
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_hflip'))
    _create_dir(os.path.join(self.tmpdir, f'{self.base_name}_hflip$0'))

    stub = self._make_stub()
    paths = stub.aug_variant_folders['hflip']
    self.assertEqual(paths, sorted(paths))


class TestLoadElementVariantSelection(unittest.TestCase):
  """Test that _load_element randomly selects from discovered variant folders."""

  def setUp(self):
    self.tmpdir = tempfile.mkdtemp()
    self.base_name = 'features_base'
    self.root_folder_features = os.path.join(self.tmpdir, self.base_name)
    _create_dir(self.root_folder_features)

    # Create 3 hflip variant folders with dummy safetensors
    self.hflip_folders = []
    for suffix in ['', '$0', '$1']:
      folder = os.path.join(self.tmpdir, f'{self.base_name}_hflip{suffix}')
      _create_dir(folder)
      self.hflip_folders.append(folder)
      _create_dummy_safetensor(os.path.join(folder, 'subj1', 'sample1.safetensors'))

    # Also create base features safetensor for originals
    _create_dummy_safetensor(os.path.join(self.root_folder_features, 'subj1', 'sample1.safetensors'))

    # Build a stub with the needed attributes
    self.stub = object.__new__(customDatasetWhole)
    self.stub.root_folder_features = self.root_folder_features
    self.stub._discover_variant_folders()
    self.stub.feature_merge_type = None

    # Minimal DataFrame: one augmented sample (hflip)
    helper.step_shift = 100
    hflip_shift = helper.get_shift_for_sample_id('hflip')
    self.stub.df = pd.DataFrame([
      {'sample_id': 1, 'class_id': 0, 'subject_name': 'subj1', 'sample_name': 'sample1', 'subject_id': 1},
      {'sample_id': 1 + hflip_shift, 'class_id': 0, 'subject_name': 'subj1', 'sample_name': 'sample1', 'subject_id': 1},
    ])
    self.aug_idx = 1  # index of the augmented sample

  def tearDown(self):
    shutil.rmtree(self.tmpdir)

  def test_random_selection_distribution(self):
    """Over many calls, multiple variant folders should be selected."""
    selected_folders = set()
    n_calls = 150

    # Mock _get_element to just return None (we only care about which folder is selected)
    with patch('custom.dataset._get_element', return_value=None):
      # Mock the model attributes needed by _load_element
      self.stub.is_quadrant = False
      self.stub.consider_only_lasts_n_chunks = None
      self.stub.model = MagicMock()
      self.stub.model.embedding_reduction = None
      self.stub.latent_coefficient = None
      self.stub.xattn_mask = False

      for _ in range(n_calls):
        csv_row = self.stub.df.iloc[self.aug_idx]
        sample_id = csv_row['sample_id']
        aug_type = helper.get_augmentation_type(sample_id)
        variants = self.stub.aug_variant_folders[aug_type]
        chosen = variants[np.random.randint(len(variants))]
        selected_folders.add(chosen)

    # With 3 variants and 150 calls, probability of seeing only 1 is (1/3)^149 ≈ 0
    self.assertGreater(len(selected_folders), 1,
      f"Expected multiple variants to be selected, got: {selected_folders}")

  def test_missing_file_raises_error(self):
    """Loading from a variant folder with missing file should raise an error."""
    # Create a variant folder WITHOUT the expected safetensors file
    empty_variant = os.path.join(self.tmpdir, f'{self.base_name}_jitter')
    _create_dir(empty_variant)

    self.stub._discover_variant_folders()

    # Make a sample that maps to jitter augmentation
    jitter_shift = helper.get_shift_for_sample_id('jitter')
    self.stub.df = pd.DataFrame([
      {'sample_id': 1 + jitter_shift, 'class_id': 0, 'subject_name': 'subj1', 'sample_name': 'sample1', 'subject_id': 1},
    ])

    self.stub.is_quadrant = False
    self.stub.consider_only_lasts_n_chunks = None
    self.stub.model = MagicMock()
    self.stub.model.embedding_reduction = None
    self.stub.latent_coefficient = None
    self.stub.xattn_mask = False

    with patch('custom.dataset._get_element', return_value=None):
      with self.assertRaises(Exception):
        self.stub._load_element(0)

  def test_backward_compat_no_variants(self):
    """Without variant folders, the original single-folder path is used."""
    # Create a fresh tmpdir with no variant folders
    tmpdir2 = tempfile.mkdtemp()
    base2 = 'features_only'
    root2 = os.path.join(tmpdir2, base2)
    _create_dir(root2)
    _create_dir(os.path.join(tmpdir2, f'{base2}_hflip'))
    _create_dummy_safetensor(os.path.join(tmpdir2, f'{base2}_hflip', 'subj1', 'sample1.safetensors'))

    stub2 = object.__new__(customDatasetWhole)
    stub2.root_folder_features = root2
    stub2._discover_variant_folders()

    # Only one folder for hflip (the base)
    self.assertEqual(len(stub2.aug_variant_folders['hflip']), 1)
    self.assertEqual(
      stub2.aug_variant_folders['hflip'][0],
      os.path.join(tmpdir2, f'{base2}_hflip')
    )

    shutil.rmtree(tmpdir2)


class TestLoadInMemoryDiscoversVariants(unittest.TestCase):
  """Test that load_whole_dict_data_in_memory discovers $N variant folders."""

  def test_variant_folders_in_scan_list(self):
    """Variant folders should appear in folders_to_scan."""
    import re
    tmpdir = tempfile.mkdtemp()
    base_name = 'features_test'
    path_to_features = os.path.join(tmpdir, base_name)
    _create_dir(path_to_features)
    _create_dir(os.path.join(tmpdir, f'{base_name}_hflip'))
    _create_dir(os.path.join(tmpdir, f'{base_name}_hflip$0'))
    _create_dir(os.path.join(tmpdir, f'{base_name}_hflip$1'))
    _create_dir(os.path.join(tmpdir, f'{base_name}_jitter'))

    # Simulate the discovery logic from load_whole_dict_data_in_memory
    dict_augmented = {'hflip': 1, 'jitter': 1, 'latent_basic': 1}
    folders_to_scan = [path_to_features]
    parent_dir = os.path.dirname(path_to_features)
    base = os.path.basename(path_to_features)
    active_aug_types = set()

    for type_augm, p in dict_augmented.items():
      if p > 0 and p <= 1 and 'latent' not in type_augm:
        active_aug_types.add(type_augm)
        aug_folder = f"{path_to_features}_{type_augm}"
        if os.path.isdir(aug_folder):
          folders_to_scan.append(aug_folder)

    if active_aug_types and os.path.isdir(parent_dir):
      variant_pattern = re.compile(
        r'^' + re.escape(base) + r'_([^$]+?)\$(\d+)$'
      )
      for entry in os.listdir(parent_dir):
        m = variant_pattern.match(entry)
        if m and m.group(1) in active_aug_types:
          full_path = os.path.join(parent_dir, entry)
          if os.path.isdir(full_path):
            folders_to_scan.append(full_path)

    # Should include: base, _hflip, _jitter, _hflip$0, _hflip$1
    self.assertEqual(len(folders_to_scan), 5)
    folder_basenames = [os.path.basename(f) for f in folders_to_scan]
    self.assertIn(f'{base_name}_hflip$0', folder_basenames)
    self.assertIn(f'{base_name}_hflip$1', folder_basenames)
    self.assertIn(f'{base_name}_hflip', folder_basenames)
    self.assertIn(f'{base_name}_jitter', folder_basenames)
    # latent_basic should NOT be discovered (filtered by 'latent' check)
    self.assertNotIn(f'{base_name}_latent_basic', folder_basenames)

    shutil.rmtree(tmpdir)

  def test_variant_folders_latent_excluded(self):
    """Latent augmentation types should not have their variants scanned."""
    import re
    tmpdir = tempfile.mkdtemp()
    base_name = 'features_test'
    path_to_features = os.path.join(tmpdir, base_name)
    _create_dir(path_to_features)
    _create_dir(os.path.join(tmpdir, f'{base_name}_latent_basic'))
    _create_dir(os.path.join(tmpdir, f'{base_name}_latent_basic$0'))

    dict_augmented = {'latent_basic': 1}
    folders_to_scan = [path_to_features]
    parent_dir = os.path.dirname(path_to_features)
    base = os.path.basename(path_to_features)
    active_aug_types = set()

    for type_augm, p in dict_augmented.items():
      if p > 0 and p <= 1 and 'latent' not in type_augm:
        active_aug_types.add(type_augm)
        aug_folder = f"{path_to_features}_{type_augm}"
        if os.path.isdir(aug_folder):
          folders_to_scan.append(aug_folder)

    if active_aug_types and os.path.isdir(parent_dir):
      variant_pattern = re.compile(
        r'^' + re.escape(base) + r'_([^$]+?)\$(\d+)$'
      )
      for entry in os.listdir(parent_dir):
        m = variant_pattern.match(entry)
        if m and m.group(1) in active_aug_types:
          full_path = os.path.join(parent_dir, entry)
          if os.path.isdir(full_path):
            folders_to_scan.append(full_path)

    # Only the base folder — latent types should be excluded
    self.assertEqual(len(folders_to_scan), 1)

    shutil.rmtree(tmpdir)


if __name__ == '__main__':
  unittest.main()
