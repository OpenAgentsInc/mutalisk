"""CLI: run the offline optimization loop end-to-end and emit a candidate.

    python -m mutalisk.optimize --optimizer local
    python -m mutalisk.optimize --optimizer gepa --max-metric-calls 60

Offline and bounded: no network, no production writes. Output candidate JSON
lands in the gitignored ``candidates/`` directory.
"""

from __future__ import annotations

import argparse
import sys

from .demo import (
    DEFAULT_KHALA_FLEET_DELEGATION_DEMO_DATASET,
    DEFAULT_KHALA_FLEET_DELEGATION_SUMMARY,
    run_khala_fleet_delegation_demo,
)
from .emitter import FileCandidateEmitter
from .optimizer import DelegationGepaOptimizer, GepaOptimizer, LocalSearchOptimizer, Optimizer
from .runner import run_delegation_optimization, run_optimization
from .trace_eval import TraceEvalDataset


def _build_optimizer(name: str, max_metric_calls: int, seed: int) -> Optimizer:
    if name == "local":
        return LocalSearchOptimizer()
    if name == "gepa":
        return GepaOptimizer(max_metric_calls=max_metric_calls, seed=seed)
    raise ValueError(f"unknown optimizer: {name}")


def main(argv: list[str] | None = None) -> int:
    argsv = list(sys.argv[1:] if argv is None else argv)
    if argsv and argsv[0] == "demo":
        return _demo_main(argsv[1:])

    parser = argparse.ArgumentParser(prog="mutalisk.optimize")
    parser.add_argument(
        "--optimizer",
        choices=["local", "gepa"],
        default="local",
        help="optimizer backend (default: local; gepa requires the gepa package)",
    )
    parser.add_argument(
        "--out-dir",
        default="candidates",
        help="candidate sink directory (gitignored; default: candidates)",
    )
    parser.add_argument(
        "--target",
        choices=["sentiment", "khala-fleet-delegation"],
        default="sentiment",
        help="optimization target (default: sentiment fixture)",
    )
    parser.add_argument(
        "--trace-evals",
        help=(
            "sanitized executed trace/eval JSONL file; records must contain "
            "public_text, label, split, trace_ref, and eval_ref"
        ),
    )
    parser.add_argument("--max-metric-calls", type=int, default=60, help="GEPA budget")
    parser.add_argument("--seed", type=int, default=0, help="reproducibility seed")
    args = parser.parse_args(argsv)

    emitter = FileCandidateEmitter(args.out_dir)
    if args.target == "khala-fleet-delegation":
        if args.optimizer != "gepa":
            parser.error("--target khala-fleet-delegation currently requires --optimizer gepa")
        if args.trace_evals:
            parser.error("--trace-evals is only supported for the sentiment fixture target")
        optimizer = DelegationGepaOptimizer(
            max_metric_calls=args.max_metric_calls,
            seed=args.seed,
        )
        out = run_delegation_optimization(optimizer, emitter)
        r = out.result
        print(f"optimizer:   {r.optimizer_id}")
        print(f"signature:   {out.candidate.signature}")
        print(f"metric:      {r.metric_name} {r.base_metric_value:.3f} -> {r.metric_value:.3f}")
        print(f"components:  {r.optimized_candidate}")
        print(f"candidate:   {out.sink_path}")
        return 0

    optimizer = _build_optimizer(args.optimizer, args.max_metric_calls, args.seed)
    trace_eval_dataset = (
        TraceEvalDataset.from_jsonl(args.trace_evals) if args.trace_evals else None
    )

    out = run_optimization(optimizer, emitter, trace_eval_dataset=trace_eval_dataset)
    r = out.result
    print(f"optimizer:   {r.optimizer_id}")
    print(f"signature:   {out.candidate.signature}")
    print(f"metric:      {r.metric_name} {r.base_metric_value:.3f} -> {r.metric_value:.3f}")
    print(f"components:  {r.optimized_candidate}")
    print(f"candidate:   {out.sink_path}")
    return 0


def _demo_main(argv: list[str]) -> int:
    if not argv:
        parser = argparse.ArgumentParser(prog="mutalisk-optimize demo")
        parser.error("missing demo name")
    demo_name = argv[0]
    if demo_name != "khala-fleet-delegation":
        parser = argparse.ArgumentParser(prog="mutalisk-optimize demo")
        parser.error(f"unknown demo: {demo_name}")

    parser = argparse.ArgumentParser(
        prog="mutalisk-optimize demo khala-fleet-delegation",
        description=(
            "Run the bounded, offline Khala fleet-delegation GEPA demo and "
            "emit an OpenAgents-ingestible manifest summary."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_KHALA_FLEET_DELEGATION_DEMO_DATASET),
        help="public-safe delegation fixture dataset JSON",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=8,
        help="offline GEPA metric-call budget (default: 8)",
    )
    parser.add_argument("--seed", type=int, default=0, help="reproducibility seed")
    parser.add_argument(
        "--candidate-out-dir",
        help=(
            "detailed candidate artifact directory; defaults to a candidates/ "
            "directory beside --emit-openagents-summary"
        ),
    )
    parser.add_argument(
        "--emit-openagents-summary",
        default=str(DEFAULT_KHALA_FLEET_DELEGATION_SUMMARY),
        help=(
            "path for the psionic.probe_gepa_candidate_manifest.v1 summary JSON "
            f"(default: {DEFAULT_KHALA_FLEET_DELEGATION_SUMMARY})"
        ),
    )
    args = parser.parse_args(argv[1:])

    out = run_khala_fleet_delegation_demo(
        candidate_out_dir=args.candidate_out_dir,
        dataset_path=args.dataset,
        max_metric_calls=args.max_metric_calls,
        seed=args.seed,
        summary_path=args.emit_openagents_summary,
    )
    summary = out.summary
    print("Mutalisk Khala fleet delegation demo: PASS")
    print("mode:                 offline_gepa_no_lm_no_network")
    print(f"signature:            {summary['signature']}")
    print(f"metricName:           {summary['metricName']}")
    print(f"metricValueBps:       {summary['metricValueBps']}")
    print(f"candidateManifestRef: {summary['candidateManifestRef']}")
    print(f"candidateRef:         {summary['candidateRef']}")
    print(f"candidateArtifact:    {out.candidate_artifact_path}")
    print(f"openagentsSummary:    {out.summary_path}")
    print("OpenAgents no-UI bridge:")
    print(
        "  bun clients/khala-code-desktop/scripts/part2-gepa-manifest-bridge.ts "
        f"--summary {out.summary_path} "
        "--out out/khala-gepa-bridge-proof.json"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
