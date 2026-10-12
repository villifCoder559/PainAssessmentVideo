from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / 'configs' / 'train').glob('*.yaml'))
# timeout is rewritten to seconds, global_folder_name gets a run suffix, n_trials is overridden below.
REWRITTEN = {'timeout', 'global_folder_name', 'n_trials'}


def _run(*args):
  return subprocess.run([sys.executable, 'train_model.py', *args], cwd=ROOT, capture_output=True, text=True)


@pytest.mark.parametrize('config', CONFIGS, ids=lambda p: p.stem)
def test_yaml_config_values_reach_the_resolved_arguments(config, tmp_path):
  cfg = yaml.safe_load(config.read_text())
  if not (ROOT / cfg['ffsp']).exists():
    pytest.skip(f"features not available: {cfg['ffsp']}")

  # n_trials 0: the run stops (optuna error) right after _config_prompt.txt is written.
  _run('--config', str(config), '--n_trials', '0', '--global_folder_name', str(tmp_path / 'run'))
  prompt, = tmp_path.glob('run_*/_config_prompt.txt')
  resolved = dict(line.split(': ', 1) for line in prompt.read_text().splitlines()[1:])

  for key, value in cfg.items():
    if key not in REWRITTEN:
      assert resolved[key] == str(value), key


def test_unknown_yaml_key_is_rejected(tmp_path):
  bad = tmp_path / 'bad.yaml'
  bad.write_text('lr: [0.001]\nnot_a_flag: 1\n')
  result = _run('--config', str(bad))
  assert result.returncode == 2
  assert "Unknown keys" in result.stderr and 'not_a_flag' in result.stderr
