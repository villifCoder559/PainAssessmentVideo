import importlib
import io
import os
import pickle
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "reproduce"))

import _common  # noqa: E402

SCRIPTS = sorted(p.stem for p in (REPO_ROOT / "reproduce").glob("*.py") if not p.stem.startswith("_"))


def make_run(runs_root: Path, config: str, folds: int) -> Path:
  run = runs_root / config / "history_run_1" / "1_DFER_ATTENTIVE_JEPA"
  results = {}
  for k in range(folds):
    ckpt = run / "train_ATTENTIVE_JEPA" / f"k{k}_cross_val" / f"k{k}_cross_val_sub_0" / "best_model_ep_3.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    results[f"k{k}_cross_val_final"] = {"best_model": {"fold_sub_fold_idx": (k, 0), "best_model_idx": 3}}
  pkl = run / "k_fold_results.pkl"
  with open(pkl, "wb") as handle:
    pickle.dump({"results": results}, handle)
  return pkl


class ReproduceScriptsTest(unittest.TestCase):
  def setUp(self):
    cwd = os.getcwd()
    self.addCleanup(os.chdir, cwd)

  def test_every_script_has_help(self):
    for name in SCRIPTS:
      with self.subTest(script=name):
        module = importlib.import_module(name)
        with redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as exit_:
          module.main(["--help"])
        self.assertEqual(exit_.exception.code, 0)
        self.assertIn("usage:", out.getvalue())

  def test_newest_complete_run_uses_fold_count_of_config(self):
    with tempfile.TemporaryDirectory() as tmp:
      runs = Path(tmp)
      xite = make_run(runs, "xite_dfer", folds=1)  # stop: [1, 1]
      make_run(runs, "biovid_dfer", folds=3)      # stop: [5, 1] -> incomplete
      found = _common.newest_complete_run("xite_dfer", runs)
      self.assertEqual(found[0], xite)
      self.assertEqual(len(found[1]), 1)
      self.assertIsNone(_common.newest_complete_run("biovid_dfer", runs))

  def test_cross_space_done_needs_aggregated_pkl(self):
    with tempfile.TemporaryDirectory() as tmp:
      os.chdir(tmp)
      tag = "paper_tests_seed_42/x_cross_validation/refinement3_linear"
      aggregate = Path("Cross_projection") / tag / "aggregated_1"
      aggregate.mkdir(parents=True)
      self.assertFalse(_common.cross_space_done(tag))
      (aggregate / "results_1.pkl").touch()
      self.assertTrue(_common.cross_space_done(tag))

  def test_train_all_skips_complete_runs_and_trains_the_rest(self):
    train_all = importlib.import_module("train_all")
    with tempfile.TemporaryDirectory() as tmp:
      make_run(Path(tmp), "xite_dfer", folds=1)
      calls = []
      real = _common.newest_complete_run
      with mock.patch.object(train_all, "newest_complete_run",
                             lambda c: real(c, Path(tmp))), \
           mock.patch.object(train_all, "run", lambda cmd, gpu=None: calls.append((cmd, gpu))), \
           redirect_stdout(io.StringIO()):
        train_all.main(["--gpu", "1", "--only", "xite_dfer", "xite_vmae_s"])
    self.assertEqual(calls, [([sys.executable, "train_model.py", "--config",
                               "configs/train/xite_vmae_s.yaml"], "1")])

  def test_all_runs_steps_in_order_and_reports_failures(self):
    all_ = importlib.import_module("all")
    calls = []

    def recorder(name, code=0):
      def main(argv):
        calls.append((name, argv))
        return code
      return main

    modules = ["train_all", "fill_cross_space_checkpoints", "run_cross_space", "summarize_cross_space"]
    patches = [mock.patch.object(getattr(all_, m), "main", recorder(m)) for m in modules]
    patches += [mock.patch.object(step, "main", recorder(step.__name__, 1 if step.__name__ == "table_baselines" else 0))
                for step in all_.PAPER_STEPS]
    for patch in patches:
      patch.start()
      self.addCleanup(patch.stop)
    with redirect_stdout(io.StringIO()) as out:
      code = all_.main(["--gpu", "2", "--force"])
    self.assertEqual(code, 1)
    self.assertIn("Incomplete: table_baselines", out.getvalue())
    self.assertEqual(calls[:4], [
      ("train_all", ["--gpu", "2"]),
      ("fill_cross_space_checkpoints", []),
      ("run_cross_space", ["--gpu", "2", "--force"]),
      ("summarize_cross_space", ["--force"]),
    ])
    self.assertEqual([c[0] for c in calls[4:]], [s.__name__ for s in all_.PAPER_STEPS])


if __name__ == "__main__":
  unittest.main()
