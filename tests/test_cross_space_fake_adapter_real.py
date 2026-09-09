"""Opt-in acceptance tests against saved BioVid -> MIntPAIN artifacts."""

import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cross_space_fake_projection import PROJECTOR_KINDS, discover_results, replay_result


REAL_ROOT = REPO_ROOT / "Cross_projection" / "bioVmae_to_mintDfer"


@unittest.skipUnless(
  os.environ.get("RUN_REAL_CROSS_SPACE_TESTS") == "1" and REAL_ROOT.is_dir(),
  "set RUN_REAL_CROSS_SPACE_TESTS=1 to replay the local real artifacts",
)
class RealFakeAdapterAcceptanceTest(unittest.TestCase):
  def test_replays_one_original_artifact_per_learned_adapter_family(self):
    representatives = {}
    sources = discover_results(REAL_ROOT)
    self.assertTrue(sources)
    self.assertTrue(all(
      not any(part.startswith(("fake_projection_", "fake_adapter_"))
              for part in path.parts)
      for path in sources
    ))
    for source in sources:
      with source.open("rb") as stream:
        data = pickle.load(stream)
      config = data.get("trial_params") or data.get("config_cross_space_projection") or {}
      kind = str(config.get("interpolation_similarity", "")).lower()
      if kind in PROJECTOR_KINDS:
        representatives.setdefault(kind, source)
      if set(representatives) == PROJECTOR_KINDS:
        break
    self.assertEqual(set(representatives), PROJECTOR_KINDS)

    with tempfile.TemporaryDirectory() as tmp:
      for kind, source in sorted(representatives.items()):
        with self.subTest(kind=kind, source=source):
          output = Path(tmp) / kind / source.name
          replay_result(source, output, control="fake_adapter", seed=42)
          with output.open("rb") as stream:
            replayed = pickle.load(stream)
          self.assertEqual(replayed["fake_projection_control"], "fake_adapter")
          self.assertNotIn("fake_source_embeddings", replayed)
          for evaluation in replayed["fake_projection_evaluations"].values():
            self.assertNotIn("control_source_embeddings", evaluation)
            self.assertFalse(np.array_equal(
              evaluation["fake_predictions"], evaluation["real_predictions"]))


if __name__ == "__main__":
  unittest.main()
