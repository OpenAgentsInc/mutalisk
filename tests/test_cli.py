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
    assert data["metric"]["name"] == "val_accuracy"


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


def test_cli_accepts_sanitized_trace_eval_jsonl(tmp_path):
    trace_evals = tmp_path / "trace-evals.jsonl"
    rows = [
        {
            "public_text": "great green build",
            "label": "positive",
            "split": "train",
            "trace_ref": "trace://run/train-1",
            "eval_ref": "eval://run/train-1",
        },
        {
            "public_text": "broken red build",
            "label": "negative",
            "split": "train",
            "trace_ref": "trace://run/train-2",
            "eval_ref": "eval://run/train-2",
        },
        {
            "public_text": "great build",
            "label": "positive",
            "split": "val",
            "trace_ref": "trace://run/val-1",
            "eval_ref": "eval://run/val-1",
        },
        {
            "public_text": "broken build",
            "label": "negative",
            "split": "val",
            "trace_ref": "trace://run/val-2",
            "eval_ref": "eval://run/val-2",
        },
    ]
    trace_evals.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    out_dir = tmp_path / "candidates"
    rc = main(["--optimizer", "local", "--out-dir", str(out_dir), "--trace-evals", str(trace_evals)])
    assert rc == 0
    data = json.loads(next(out_dir.glob("*.json")).read_text())
    assert data["metric"]["eval_dataset_ref"].startswith("trace-eval://sha256:")
    assert data["trace_provenance"]["record_count"] == 4


def test_cli_emits_khala_fleet_delegation_gepa_candidate(tmp_path, capsys):
    out_dir = tmp_path / "delegation-candidates"
    rc = main(
        [
            "--target",
            "khala-fleet-delegation",
            "--optimizer",
            "gepa",
            "--max-metric-calls",
            "40",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "khala.fleet.delegation" in out
    data = json.loads(next(out_dir.glob("*.json")).read_text())
    assert data["signature"] == "khala.fleet.delegation"
    assert data["metric"]["value"] > data["metric"]["base_value"]
