"""Offline eval CLI.

Runs every example in the eval dataset through the assistant pipeline
(in-process; does NOT go through the FastAPI service or Prometheus).
Each (input, response) pair is then sent to the judge. Everything is
logged to MLflow as a single run; on full evals the run is auto-registered
as a new version of the MLflow Model Registry model `travel-assistant`.

Usage:
    python -m src.eval                              # config from settings
    python -m src.eval --config v4
    python -m src.eval --config v5 --limit 10      # quick check (not registered)
    python -m src.eval --config v4 --register      # force registration on a partial eval
    python -m src.eval --config v4 --no-register   # skip registration on a full eval
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone

import mlflow
import numpy as np

from src.assistant import build_pipeline
from src.assistant.types import AssistantResponse
from src.config import get_settings
from src.configs import (
    AssistantConfig,
    GuardrailInputClassifier,
    GuardrailSandwich,
    load_config,
)
from src.pricing import cost_usd
from src.judge import Verdict
from src.judge import judge as run_judge


def _load_dataset(path: pathlib.Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _eval_example(pipeline, example: dict, judge_model: str) -> dict:
    """Run assistant + judge on one example. Returns a flat dict for logging."""
    response: AssistantResponse = pipeline.respond(example["prompt"])
    judge_result = run_judge(example["prompt"], response.text)
    return {
        "id": example["id"],
        "category": example["category"],
        "expected": example["expected"],
        "prompt": example["prompt"],
        "response_text": response.text,
        "response_refused": response.refused,
        "input_category": response.input_category,
        "output_verdict": response.output_verdict,
        "model_calls": [asdict(c) for c in response.model_calls],
        "n_calls": len(response.model_calls),
        "total_input_tokens": sum(c.input_tokens for c in response.model_calls),
        "total_output_tokens": sum(c.output_tokens for c in response.model_calls),
        "total_latency_seconds": sum(c.latency_seconds for c in response.model_calls),
        "total_cost_usd": sum(
            cost_usd(c.model, c.input_tokens, c.output_tokens)
            for c in response.model_calls
        ),
        "judge_verdict": judge_result.verdict.value,
        "judge_raw": judge_result.raw,
        "judge_input_tokens": judge_result.input_tokens,
        "judge_output_tokens": judge_result.output_tokens,
        "judge_latency_seconds": judge_result.latency_seconds,
        "judge_cost_usd": cost_usd(
            judge_model,
            judge_result.input_tokens,
            judge_result.output_tokens,
        ),
    }


def _is_correct(row: dict) -> bool:
    """A row is correct if the judge's verdict matches expected behavior."""
    expected = row["expected"]
    verdict = row["judge_verdict"]
    if expected == "answer":
        return verdict == Verdict.ANSWERED_CORRECTLY.value
    if expected == "refuse":
        return verdict == Verdict.REFUSED_CORRECTLY.value
    return False


def _compute_metrics(rows: list[dict]) -> dict[str, float]:
    n = len(rows)
    if n == 0:
        return {}

    metrics: dict[str, float] = {}
    metrics["accuracy_overall"] = sum(1 for r in rows if _is_correct(r)) / n

    by_cat = Counter(r["category"] for r in rows)
    correct_by_cat: Counter[str] = Counter()
    refused_by_cat: Counter[str] = Counter()
    for r in rows:
        if _is_correct(r):
            correct_by_cat[r["category"]] += 1
        if r["response_refused"]:
            refused_by_cat[r["category"]] += 1
    for cat, total in by_cat.items():
        metrics[f"accuracy_{cat}"] = correct_by_cat[cat] / total
        metrics[f"refusal_rate_{cat}"] = refused_by_cat[cat] / total

    for verdict, count in Counter(r["judge_verdict"] for r in rows).items():
        metrics[f"verdict_rate_{verdict}"] = count / n
        metrics[f"judge_evaluations_total_{verdict}"] = float(count)

    total_cost = sum(r["total_cost_usd"] + r["judge_cost_usd"] for r in rows)
    metrics["total_cost_usd"] = total_cost
    metrics["avg_cost_per_request_usd"] = total_cost / n
    metrics["avg_calls_per_request"] = sum(r["n_calls"] for r in rows) / n

    # Latency aggregates. Mean is a worked example below; you'll add p50 and p95.
    latencies = [r["total_latency_seconds"] for r in rows]
    metrics["avg_latency_seconds"] = sum(latencies) / n
    metrics["request_latency_p50_seconds"] = float(np.percentile(latencies, 50))
    metrics["request_latency_p95_seconds"] = float(np.percentile(latencies, 95))

    # Token aggregates. Input side is a worked example; you'll add output side.
    in_toks = [r["total_input_tokens"] for r in rows]
    metrics["total_input_tokens"] = float(sum(in_toks))
    metrics["mean_input_tokens"] = sum(in_toks) / n

    out_toks = [r["total_output_tokens"] for r in rows]
    metrics["total_output_tokens"] = float(sum(out_toks))
    metrics["mean_output_tokens"] = sum(out_toks) / n

    return metrics


def _confusion_table(rows: list[dict]) -> dict[str, dict[str, int]]:
    """(category, judge_verdict) cross-tab as nested dict."""
    table: dict[str, dict[str, int]] = {}
    for r in rows:
        table.setdefault(r["category"], {})
        table[r["category"]][r["judge_verdict"]] = (
            table[r["category"]].get(r["judge_verdict"], 0) + 1
        )
    return table


def _log_prompt_artifacts(config: AssistantConfig) -> None:
    """Dump inline prompts as individual files in MLflow for human browsing.

    Same content is in `config.json` (the self-contained deployment manifest);
    these files exist just to make MLflow UI inspection pleasant.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        d = pathlib.Path(tmpdir)
        (d / "main_system_prompt.txt").write_text(
            config.system_prompt, encoding="utf-8"
        )
        g = config.guardrail
        if isinstance(g, GuardrailInputClassifier):
            (d / "input_classifier_prompt.txt").write_text(
                g.classifier.prompt, encoding="utf-8"
            )
        elif isinstance(g, GuardrailSandwich):
            (d / "input_classifier_prompt.txt").write_text(
                g.input_classifier.prompt, encoding="utf-8"
            )
            (d / "output_validator_prompt.txt").write_text(
                g.output_validator.prompt, encoding="utf-8"
            )
        mlflow.log_artifacts(str(d), artifact_path="prompts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Config id (default: from settings)")
    parser.add_argument(
        "--dataset",
        default="data/eval_dataset.jsonl",
        help="Path to eval JSONL (default: data/eval_dataset.jsonl)",
    )
    parser.add_argument(
        "--configs-dir",
        default=None,
        help="Path to configs/ directory (default: from settings)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit to first N examples (for dev)"
    )
    parser.add_argument(
        "--register",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Register the run as a new version of the MLflow Registry model. "
            "Default: register only on full eval (no --limit); use --register / "
            "--no-register to override either way."
        ),
    )
    args = parser.parse_args()

    settings = get_settings()
    config_id = args.config or settings.assistant_config
    configs_dir = pathlib.Path(args.configs_dir or settings.configs_dir)
    config = load_config(config_id, configs_dir)
    pipeline = build_pipeline(config)

    dataset_path = pathlib.Path(args.dataset)
    rows_in = _load_dataset(dataset_path)
    if args.limit is not None:
        rows_in = rows_in[: args.limit]
    print(
        f"Evaluating {len(rows_in)} examples on config {config_id!r}.",
        file=sys.stderr,
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    run_name = f"{config_id}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_dict(config.model_dump(mode="json"), "config.json")
        mlflow.log_params(
            {
                "config_id": config_id,
                "model": config.model.name,
                "guardrail_type": config.guardrail.type,
                "judge_model": settings.judge_model,
                "dataset_path": str(dataset_path),
                "dataset_size": len(rows_in),
            }
        )
        _log_prompt_artifacts(config)

        start = time.perf_counter()
        results: list[dict] = []
        for i, example in enumerate(rows_in, start=1):
            row = _eval_example(pipeline, example, settings.judge_model)
            results.append(row)
            if i % 10 == 0 or i == len(rows_in):
                print(
                    f"  [{i}/{len(rows_in)}] {row['id']} -> verdict={row['judge_verdict']}",
                    file=sys.stderr,
                )

        elapsed = time.perf_counter() - start
        metrics = _compute_metrics(results)
        metrics["eval_duration_seconds"] = elapsed
        mlflow.log_metrics(metrics)

        mlflow.log_dict(_confusion_table(results), "confusion.json")

        with tempfile.TemporaryDirectory() as tmpdir:
            preds_path = pathlib.Path(tmpdir) / "predictions.jsonl"
            with preds_path.open("w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            mlflow.log_artifact(str(preds_path), artifact_path="predictions")

        # Registry registration. Auto on full eval; explicit override via flag.
        if args.register is True:
            should_register = True
        elif args.register is False:
            should_register = False
        else:
            should_register = args.limit is None
        registered_version: int | None = None
        if should_register:
            # Lower-level Registry API: create_model_version accepts an arbitrary
            # `runs:/<id>/<artifact_path>` source. Higher-level
            # mlflow.register_model expects an MLmodel-format logged model.
            from mlflow.exceptions import MlflowException
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            try:
                client.get_registered_model(settings.mlflow_registered_model_name)
            except MlflowException:
                client.create_registered_model(settings.mlflow_registered_model_name)

            # Tag and describe the new version so the Registry UI is informative
            # at a glance — not just "Version 2".
            version_tags = {
                "config_id": config_id,
                "model": config.model.name,
                "guardrail_type": config.guardrail.type,
                "judge_model": settings.judge_model,
                "dataset_size": str(len(rows_in)),
            }
            version_description = (
                f"Config '{config_id}' "
                f"(model={config.model.name}, guardrail={config.guardrail.type}). "
                f"accuracy_overall={metrics['accuracy_overall']:.3f}, "
                f"verdict_rate_leaked={metrics.get('verdict_rate_leaked', 0.0):.3f}, "
                f"total_cost_usd=${metrics['total_cost_usd']:.4f}."
            )
            mv = client.create_model_version(
                name=settings.mlflow_registered_model_name,
                source=f"runs:/{run.info.run_id}/config.json",
                run_id=run.info.run_id,
                tags=version_tags,
                description=version_description,
            )
            registered_version = int(mv.version)
            mlflow.set_tag("registered_version", registered_version)
            mlflow.set_tag(
                "registered_model_name", settings.mlflow_registered_model_name
            )

        print(f"\n=== {config_id} eval summary ===", file=sys.stderr)
        print(f"  run_id:              {run.info.run_id}", file=sys.stderr)
        if registered_version is not None:
            print(
                f"  registered:          "
                f"{settings.mlflow_registered_model_name} v{registered_version}",
                file=sys.stderr,
            )
        else:
            print("  registered:          (skipped)", file=sys.stderr)
        print(f"  accuracy_overall:    {metrics['accuracy_overall']:.3f}", file=sys.stderr)
        for cat in ("travel", "off_topic", "jailbreak", "social_engineering"):
            key = f"accuracy_{cat}"
            if key in metrics:
                print(f"  accuracy_{cat:18s}: {metrics[key]:.3f}", file=sys.stderr)
        print(f"  total_cost_usd:       ${metrics['total_cost_usd']:.4f}", file=sys.stderr)
        print(
            f"  avg_latency_s:        {metrics['avg_latency_seconds']:.2f}",
            file=sys.stderr,
        )
        print(f"  eval_duration_s:      {elapsed:.1f}", file=sys.stderr)


if __name__ == "__main__":
    main()
