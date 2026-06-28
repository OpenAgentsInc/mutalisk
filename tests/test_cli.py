"""End-to-end CLI test for the local (dependency-free) optimizer path."""

import json
import sys

from mutalisk.optimize import main


def test_cli_local_emits_candidate(tmp_path, capsys):
    rc = main(["--optimizer", "local", "--out-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mutalisk.local_search@" in out
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["signature"] == "mutalisk.sentiment_classify.v1"
    assert data["metric_name"] == "val_accuracy"


def test_cli_module_invocation(tmp_path):
    import os
    import pathlib
    import subprocess

    # Make the package importable in the subprocess without requiring an install.
    src = pathlib.Path(__file__).resolve().parent.parent / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(src), env.get("PYTHONPATH", "")])
    rc = subprocess.run(
        [sys.executable, "-m", "mutalisk.optimize", "--optimizer", "local", "--out-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert rc.returncode == 0, rc.stderr
    assert list(tmp_path.glob("*.json"))
