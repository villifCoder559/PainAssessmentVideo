#!/usr/bin/env python3
"""Verify the captured environment without importing application/research code."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = json.loads((ROOT / 'locks/environment-snapshot.json').read_text())
errors = []


def check(condition, message):
    if not condition:
        errors.append(message)


check(platform.system() == 'Linux' and platform.machine() == 'x86_64',
      'Snapshot requires Linux x86_64.')
check(platform.python_version() == SNAPSHOT['python'],
      f"Python must be {SNAPSHOT['python']}, found {platform.python_version()}.")
metadata = Path(sys.prefix) / 'conda-meta'
check(metadata.is_dir(), 'Run with the Python interpreter of the recreated Conda environment.')
expected_conda = {x['name']: (x['version'], x['build'], x['url'])
                  for x in SNAPSHOT['conda_packages']}
actual_conda = {}
for file in metadata.glob('*.json'):
    record = json.loads(file.read_text())
    actual_conda[record['name']] = (record['version'], record['build'], record.get('url'))
check(actual_conda == expected_conda, 'Conda package versions/builds/URLs differ from the snapshot.')


def normalized(name):
    return re.sub(r'[-_.]+', '-', name).lower()


# Preserve overlapping Conda/pip metadata: do not resolve only one distribution by name.
expected_python = {(normalized(x['name']), x['version'])
                   for x in SNAPSHOT['python_distributions']}
actual_python = {(normalized(x.metadata['Name']), x.version)
                 for x in importlib.metadata.distributions()}
missing = sorted(expected_python - actual_python)
extra = sorted(actual_python - expected_python)
check(not missing, f'Missing Python distribution versions: {missing}')
check(not extra, f'Unexpected Python distribution versions: {extra}')

for artifact in SNAPSHOT['pip_artifacts']:
    if artifact['name'] == 'torchsort':
        continue
    # PyPI wheels are remotely retrievable; their selected hashes are in the pip lock.
    check(artifact['sha256'] in (ROOT / 'locks/pip-linux-64.lock').read_text(),
          f"Pip artifact hash missing: {artifact['name']}")
for artifact in SNAPSHOT.get('local_artifacts', []):
    file = ROOT / 'locks/artifacts' / artifact['filename']
    check(file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest() == artifact['sha256'],
          f'Preserved wheel differs: {file.name}')
for native in SNAPSHOT.get('native_pip_files', []):
    file = Path(sys.prefix) / 'lib/python3.10/site-packages' / native['file']
    check(file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest() == native['installed_sha256'],
          f"Native pip library differs: {native['package']}/{native['file']}")

if errors:
    for error in errors:
        print('FAIL:', error, file=sys.stderr)
    sys.exit(1)
print(f"Verified Python {platform.python_version()}, {len(actual_conda)} Conda packages, "
      f"{len(actual_python)} Python name/version pairs, and "
      f"{len(SNAPSHOT.get('native_pip_files', []))} native pip libraries.")
