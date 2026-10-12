import os
import pickle
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "reproduce"))

import fill_cross_space_checkpoints as fill  # noqa: E402

DIRECTION = "unbcVmae_to_mintDfer_cross_validation"
REAL_YAML = (REPO_ROOT / "Cross_projection_yaml" / "config_paper_tests_seed_42" / DIRECTION
             / "refinement3_linear.yaml")


def make_run(runs_root: Path, config: str, name: str, folds: int, mtime: float) -> list[str]:
  """Fake training run with `folds` finished folds; returns the expected checkpoints."""
  run = runs_root / config / f"history_run_{name}" / f"{name}_DFER_ATTENTIVE_JEPA"
  results, expected = {}, []
  for k in range(folds):
    ckpt = (run / "train_ATTENTIVE_JEPA" / f"k{k}_cross_val" / f"k{k}_cross_val_sub_1"
            / f"best_model_ep_{10 + k}.pt")
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    expected.append(ckpt.as_posix())
    results[f"k{k}_cross_val_final"] = {
      "best_model": {"fold_sub_fold_idx": (k, 1), "best_model_idx": 10 + k}}
  pkl = run / "k_fold_results.pkl"
  with open(pkl, "wb") as handle:
    pickle.dump({"results": results}, handle)
  os.utime(pkl, (mtime, mtime))
  return expected


class FillCrossSpaceCheckpointsTest(unittest.TestCase):
  def setUp(self):
    self.tmp = Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, self.tmp)
    self.runs = self.tmp / "runs"
    self.yaml_dir = self.tmp / "yaml" / "config_paper_tests_seed_42" / DIRECTION
    self.yaml_dir.mkdir(parents=True)
    self.yaml_path = self.yaml_dir / REAL_YAML.name
    shutil.copy(REAL_YAML, self.yaml_path)

  def run_main(self, *extra):
    return fill.main(["--yaml-root", str(self.tmp / "yaml"), "--runs-root", str(self.runs), *extra])

  def test_fills_newest_complete_run_and_keeps_comments(self):
    make_run(self.runs, "unbc_opi0-4_vmae_s", "old", 5, mtime=1000)
    source = make_run(self.runs, "unbc_opi0-4_vmae_s", "new", 5, mtime=2000)
    make_run(self.runs, "unbc_opi0-4_vmae_s", "partial", 3, mtime=3000)  # newest but incomplete
    target = make_run(self.runs, "mintpain_dfer", "t", 5, mtime=1000)

    self.assertEqual(self.run_main(), 0)

    text = self.yaml_path.read_text()
    config = yaml.safe_load(text)
    self.assertEqual(config["old_model_pth"], source)
    self.assertEqual(config["new_model_pth"], target)
    self.assertIn("# --- Sweep axes (lists) ---", text)
    self.assertEqual(set(config), set(yaml.safe_load(REAL_YAML.read_text())))

  def test_run_override_and_dry_run(self):
    source = make_run(self.runs, "unbc_opi0-4_vmae_s", "a", 5, mtime=1000)
    make_run(self.runs, "unbc_opi0-4_vmae_s", "b", 5, mtime=2000)
    make_run(self.runs, "mintpain_dfer", "t", 5, mtime=1000)
    override = f"unbc_opi0-4_vmae_s={self.runs / 'unbc_opi0-4_vmae_s' / 'history_run_a'}"

    self.assertEqual(self.run_main("--run", override, "--dry-run"), 0)
    self.assertEqual(self.yaml_path.read_text(), REAL_YAML.read_text())

    self.assertEqual(self.run_main("--run", override), 0)
    self.assertEqual(yaml.safe_load(self.yaml_path.read_text())["old_model_pth"], source)

  def test_missing_target_run_leaves_yaml_unchanged(self):
    make_run(self.runs, "unbc_opi0-4_vmae_s", "s", 5, mtime=1000)
    self.assertEqual(self.run_main(), 0)
    self.assertEqual(self.yaml_path.read_text(), REAL_YAML.read_text())

  def test_every_tracked_yaml_can_be_rewritten(self):
    paths = [f"runs/x/ckpt_{k}.pt" for k in range(5)]
    yaml_root = REPO_ROOT / "Cross_projection_yaml"
    for config_set, directions in fill.CONFIG_SETS.items():
      for name in directions:
        yamls = sorted((yaml_root / config_set / name).glob("*.yaml"))
        self.assertTrue(yamls, f"no YAMLs in {config_set}/{name}")
        for yaml_path in yamls:
          with self.subTest(yaml=yaml_path.relative_to(REPO_ROOT)):
            fill.fill_yaml_text(yaml_path.read_text(), paths, paths[::-1])


if __name__ == "__main__":
  unittest.main()
