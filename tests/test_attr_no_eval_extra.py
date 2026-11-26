"""`reid attributes` must work without the [eval] extra (the Kaggle notebook installs [models])."""

import subprocess
import sys

CODE = """
import sys
sys.modules["motmetrics"] = None  # simulate the [eval] extra not being installed
import reid.eval.par_metrics, reid.eval.aicity_progress
import reid.attributes.report, reid.attributes.train, reid.attributes.embed, reid.cli_attributes
from reid.eval import evaluate, save_result, to_markdown
print("ok")
"""


def test_attributes_import_without_motmetrics():
    import pytest

    pytest.importorskip("torch")
    out = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")
