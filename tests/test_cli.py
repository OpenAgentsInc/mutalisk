"""End-to-end CLI test for the local (dependency-free) optimizer path."""

import json
from pathlib import Path
import sys

from mutalisk.optimize import main

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "khala_fleet_delegation_demo.json"


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


def test_cli_demo_khala_fleet_delegation_emits_summary_and_bridge_command(tmp_path, capsys):
    summary_path = tmp_path / "summary.json"
    candidate_out_dir = tmp_path / "candidate-artifacts"
    rc = main(
        [
            "demo",
            "khala-fleet-delegation",
            "--dataset",
            str(FIXTURE),
            "--max-metric-calls",
            "8",
            "--candidate-out-dir",
            str(candidate_out_dir),
            "--emit-openagents-summary",
            str(summary_path),
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Mutalisk Khala fleet delegation demo: PASS" in out
    assert "candidateManifestRef:" in out
    assert "candidateRef:" in out
    assert "metricValueBps:" in out
    assert "candidateArtifact:" in out
    assert "openagentsSummary:" in out
    assert "part2-gepa-manifest-bridge.ts --summary" in out

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifact = json.loads(next(candidate_out_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert summary == artifact["candidateManifest"]
    assert summary["schemaVersion"] == "psionic.probe_gepa_candidate_manifest.v1"
    assert summary["signature"] == "khala.fleet.delegation"
    assert summary["metricName"] == "khala.fleet.delegation"
    assert isinstance(summary["metricValueBps"], int)
    assert summary["evalEvidenceRefs"]
    assert summary["traceProvenanceRefs"]
