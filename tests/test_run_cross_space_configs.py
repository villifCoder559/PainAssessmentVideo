import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "run_cross_space_configs.sh"


class RunCrossSpaceConfigsTest(unittest.TestCase):
  def test_runs_all_configs_in_order_and_reports_failures(self):
    with tempfile.TemporaryDirectory(dir=REPO_ROOT) as config_tmp, \
         tempfile.TemporaryDirectory() as command_tmp, \
         tempfile.TemporaryDirectory() as cwd_tmp:
      config_dir = Path(config_tmp)
      for name in ("c.yaml", "a.yaml", "b_fail.yaml"):
        (config_dir / name).touch()

      command_dir = Path(command_tmp)
      call_log = command_dir / "calls"
      fake_python = command_dir / "python3"
      fake_python.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s|%s\\n" "$CUDA_VISIBLE_DEVICES" "$3" >> "$CALL_LOG"\n'
        '[[ "$3" != *b_fail.yaml ]]\n'
      )
      fake_python.chmod(0o755)

      env = os.environ.copy()
      env["PATH"] = f"{command_dir}:{env['PATH']}"
      env["CALL_LOG"] = str(call_log)
      relative_config_dir = config_dir.relative_to(REPO_ROOT)

      result = subprocess.run(
        ["bash", str(LAUNCHER), "2", str(relative_config_dir)],
        cwd=cwd_tmp,
        env=env,
        text=True,
        capture_output=True,
      )

      self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
      calls = call_log.read_text().splitlines()
      self.assertEqual([Path(line.split("|", 1)[1]).name for line in calls],
                       ["a.yaml", "b_fail.yaml", "c.yaml"])
      self.assertTrue(all(line.startswith("2|") for line in calls))
      self.assertIn("Passed: 2", result.stdout)
      self.assertIn("Failed: 1", result.stdout)
      self.assertIn("Failed configs:\n  b_fail.yaml\n", result.stdout)

  def test_rejects_invalid_inputs(self):
    with tempfile.TemporaryDirectory() as empty_dir:
      (Path(empty_dir) / "not_a_config.yaml").mkdir()
      cases = (
        ([], "Usage:"),
        (["gpu", empty_dir], "GPU_ID must be a non-negative integer"),
        (["0", str(Path(empty_dir) / "missing")], "Config directory not found"),
        (["0", empty_dir], "No .yaml configs found"),
      )
      for args, message in cases:
        with self.subTest(args=args):
          result = subprocess.run(
            ["bash", str(LAUNCHER), *args],
            text=True,
            capture_output=True,
          )
          self.assertNotEqual(result.returncode, 0)
          self.assertIn(message, result.stderr)


if __name__ == "__main__":
  unittest.main()
