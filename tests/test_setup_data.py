"""Tests for the reproduce/setup_<dataset>.py helpers on tiny synthetic data."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "reproduce"))
sys.path.insert(0, str(REPO_ROOT))

import _setup_common  # noqa: E402
import setup_unbc  # noqa: E402
import setup_xite  # noqa: E402
from _common import load_yaml  # noqa: E402
from custom import unbc_crop  # noqa: E402


class UnbcCsvTest(unittest.TestCase):
  SEQUENCES = [("042-ll042", "ll042a"), ("042-ll042", "ll042b"), ("115-jy115", "jy115a")]

  def order(self, permutation: str):
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "permutation.txt"
      path.write_text(permutation)
      with mock.patch.object(setup_unbc, "PERMUTATION", path):
        return setup_unbc.paper_order(self.SEQUENCES[::-1])

  def test_permutation_reorders_the_sorted_sequences(self):
    self.assertEqual(self.order("2\n0\n1\n"),
                     [("115-jy115", "jy115a"), ("042-ll042", "ll042a"), ("042-ll042", "ll042b")])

  def test_incomplete_release_is_rejected(self):
    with self.assertRaises(SystemExit):
      self.order("0\n1\n")

  def test_csvs_follow_the_paper_label_rules(self):
    order = self.order("2\n0\n1\n")
    vas = {order[0]: 0, order[1]: 3, order[2]: 10}
    opi = {order[0]: 1, order[1]: 5, order[2]: 4}
    csvs = setup_unbc.build_csvs(order, vas, opi)
    vas_csv = csvs["samples.csv"]
    self.assertEqual(list(vas_csv.columns), setup_unbc.COLUMNS)
    self.assertEqual(vas_csv.subject_id.tolist(), [115, 42, 42])
    self.assertEqual(vas_csv.sample_id.tolist(), [1, 2, 3])
    self.assertEqual(vas_csv.class_name.tolist(), ["VAS_0", "VAS_3", "VAS_10"])
    self.assertEqual(csvs["samples_OPI.csv"].class_id.tolist(), [1, 5, 4])
    opi_0_4 = csvs["samples_OPI_0_to_4.csv"]
    self.assertEqual(opi_0_4.class_id.tolist(), [1, 4, 4])
    self.assertEqual(opi_0_4.class_name.tolist(), ["OPI_1", "OPI_4_5", "OPI_4_5"])
    self.assertEqual(csvs["samples_adversarial.csv"].subject_id.tolist(), [0, 1, 1])

  def test_shipped_permutation_covers_the_release(self):
    permutation = [int(v) for v in setup_unbc.PERMUTATION.read_text().split()]
    self.assertEqual(sorted(permutation), list(range(200)))


class UnbcCropTest(unittest.TestCase):
  def frames(self):
    white = np.full((40, 40, 3), 255, np.uint8)
    face1 = white.copy()
    face1[8:31, 5:21] = 100
    face1[10, 10] = 255  # a white pixel inside the face is background too
    face2 = white.copy()
    face2[4:21, 10:34] = 100
    return [white, face1, face2]

  def test_crop_drops_empty_frames_and_blackens_the_background(self):
    cropped = unbc_crop.crop_frames(self.frames())
    self.assertEqual(len(cropped), 2)
    # Largest box: width max(20-5, 33-10)=23 -> 22, height max(30-8, 20-4)=22 -> 22.
    self.assertEqual(cropped[0].shape, (22, 22, 3))
    for frame in cropped:
      self.assertFalse((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) >= unbc_crop.WHITE_THRESHOLD).any())

  def test_crop_fails_without_any_face(self):
    with self.assertRaises(ValueError):
      unbc_crop.crop_frames([np.full((8, 8, 3), 255, np.uint8)])

  def test_pad_mirrors_the_start_to_a_multiple_of_16(self):
    frames = [np.full((2, 2, 3), i, np.uint8) for i in range(18)]
    padded = unbc_crop.pad_frames(frames)
    self.assertEqual(len(padded), 32)
    self.assertEqual([int(f[0, 0, 0]) for f in padded[:14]], list(range(13, -1, -1)))
    self.assertEqual([int(f[0, 0, 0]) for f in padded[14:]], list(range(18)))


class FeatureFoldersTest(unittest.TestCase):
  def test_runs_expand_to_dollar_variants(self):
    config = {"base_name": "b", "backbones": {"S": {"augmentations": {
      "jitter": {"runs": 3, "flags": ["--color_jitter"]}}}}}
    self.assertEqual(_setup_common.feature_folders(config, "S", base_only=False),
                     [("b", []), ("b_jitter", ["--color_jitter"]), ("b_jitter$0", ["--color_jitter"]),
                      ("b_jitter$1", ["--color_jitter"])])
    self.assertEqual(_setup_common.feature_folders(config, "S", base_only=True), [("b", [])])

  def test_configs_list_the_paper_feature_folders(self):
    expected = {"biovid": (40, 22), "unbc": (53, 36), "mintpain": (36, 36), "xite": (1, 1)}
    for dataset, (dfer, vmae) in expected.items():
      config = load_yaml(REPO_ROOT / "configs" / "features" / f"{dataset}.yaml")
      with self.subTest(dataset=dataset):
        self.assertEqual(len(_setup_common.feature_folders(config, "DFER", False)), dfer)
        self.assertEqual(len(_setup_common.feature_folders(config, "S", False)), vmae)

  def test_count_features_ignores_dollar_files(self):
    with tempfile.TemporaryDirectory() as tmp:
      folder = Path(tmp) / "s1"
      folder.mkdir()
      for name in ("a.safetensors", "a$top.safetensors", "b.safetensors", "log_file.txt"):
        (folder / name).touch()
      self.assertEqual(_setup_common.count_features(Path(tmp)), 2)


class ReferenceLandmarksTest(unittest.TestCase):
  def test_root_copy_wins_over_the_shipped_one(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      shipped = REPO_ROOT / _setup_common.REF_LANDMARKS
      self.assertEqual(_setup_common.reference_landmarks(root), shipped)
      self.assertTrue(shipped.is_file())
      built = root / _setup_common.REF_LANDMARKS
      built.parent.mkdir(parents=True)
      built.touch()
      self.assertEqual(_setup_common.reference_landmarks(root), built)


class XiteLabelsTest(unittest.TestCase):
  def test_test_labels_map_segments_to_samples(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "Labels_test_data.csv"
      path.write_text("Segment ID,True Label,Prob_PL_1\nS006_seg0,PL_1,\nS006_seg12,PL_2,\n")
      self.assertEqual(setup_xite.test_labels(path), {"S006_0": (0, "PL1"), "S006_12": (1, "PL2")})


if __name__ == "__main__":
  unittest.main()
