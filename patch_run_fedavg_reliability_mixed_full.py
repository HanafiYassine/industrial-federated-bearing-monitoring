"""
patch_run_fedavg_reliability_mixed_full.py

Updates run_fedavg_reliability.py so that --scenario mixed_full
is accepted and reads the mixed_full schedule directory.
"""

from pathlib import Path

path = Path("run_fedavg_reliability.py")

if not path.exists():
    raise FileNotFoundError(path.resolve())

text = path.read_text(encoding="utf-8")

text = text.replace(
    '"mixed",\n',
    '"mixed",\n            "mixed_full",\n',
)

text = text.replace(
    'choices=[\n            "clean",\n            "sensor_noise",\n            "stale_updates",\n            "persistent_failure",\n            "mixed",\n        ],',
    'choices=[\n            "clean",\n            "sensor_noise",\n            "stale_updates",\n            "persistent_failure",\n            "mixed",\n            "mixed_full",\n        ],',
)

path.write_text(text, encoding="utf-8")

print("Patched:", path.resolve())
print("mixed_full support added.")
