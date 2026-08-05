#!/usr/bin/env python3
"""Summarize a completed DMC on-policy sweep from local TensorBoard logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


EVAL_TAGS = (
    "eval/mean",
    "eval/median",
    "eval/std",
    "eval/min",
    "eval/max",
    "eval/length",
)
SMOOTHNESS_TAGS = (
    "eval/action_delta_l2_mean",
    "eval/action_delta_l2_p95",
    "eval/action_accel_l2_mean",
    "eval/action_jerk_l2_mean",
    "eval/action_hf_power_ratio",
)
RUN_FIELDS = (
    "task",
    "algo",
    "seed",
    "attempt",
    "wandb_run_id",
    "eval_points",
    "initial_eval_mean",
    "final_eval_mean",
    "final_eval_median",
    "final_eval_episode_std",
    "final_eval_min",
    "final_eval_max",
    "final_eval_length",
    "best_eval_mean",
    "best_eval_step",
    "eval_auc_normalized",
    "eval_auc_raw",
    "final_action_delta_l2_mean",
    "final_action_delta_l2_p95",
    "final_action_accel_l2_mean",
    "final_action_jerk_l2_mean",
    "final_action_hf_power_ratio",
    "event_runtime_seconds",
    "raw_frames_per_second",
    "policy_transitions_per_second",
    "elapsed_seconds",
    "e2e_raw_frames_per_second",
    "e2e_policy_transitions_per_second",
    "event_file",
    "wandb_summary",
)
OVERALL_FIELDS = (
    "algo",
    "tasks",
    "mean_final_across_tasks",
    "mean_auc_across_tasks",
    "average_rank",
    "task_wins",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-log-root", type=Path, required=True)
    parser.add_argument(
        "--log-root",
        type=Path,
        help="Defaults to <repo_root>/<log_dir> recorded in run-config.txt.",
    )
    parser.add_argument("--tag")
    parser.add_argument("--train-frames", type=int)
    parser.add_argument("--frame-skip", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Summarize only successful .done runs instead of requiring the full manifest.",
    )
    parser.add_argument(
        "--skip-wandb-check",
        action="store_true",
        help="Do not require local W&B summary and Synced markers.",
    )
    return parser.parse_args()


def _parse_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"index", "seed", "task", "algo"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid sweep manifest: {path}")
    return rows


def _series(accumulator: EventAccumulator, tag: str) -> list[tuple[int, float, float]]:
    if tag not in accumulator.Tags().get("scalars", []):
        raise ValueError(f"Missing TensorBoard scalar {tag!r}")
    by_step: dict[int, tuple[int, float, float]] = {}
    for event in accumulator.Scalars(tag):
        point = (int(event.step), float(event.value), float(event.wall_time))
        previous = by_step.get(point[0])
        if previous is None or point[2] >= previous[2]:
            by_step[point[0]] = point
    return [by_step[step] for step in sorted(by_step)]


def _load_final_event(
    tb_dir: Path,
    train_frames: int,
) -> tuple[Path, EventAccumulator, list[tuple[int, float, float]]]:
    candidates: list[
        tuple[int, int, Path, EventAccumulator, list[tuple[int, float, float]]]
    ] = []
    for event_file in sorted(tb_dir.glob("events.out.tfevents.*")):
        accumulator = EventAccumulator(
            str(event_file),
            size_guidance={"scalars": 0},
        )
        accumulator.Reload()
        try:
            eval_mean = _series(accumulator, "eval/mean")
        except ValueError:
            continue
        candidates.append(
            (
                eval_mean[-1][0],
                event_file.stat().st_mtime_ns,
                event_file,
                accumulator,
                eval_mean,
            )
        )
    final_candidates = [item for item in candidates if item[0] == train_frames]
    if not final_candidates:
        observed = sorted({item[0] for item in candidates})
        raise ValueError(
            f"No TensorBoard event in {tb_dir} reaches final step "
            f"{train_frames}; observed maxima={observed}"
        )
    _, _, event_file, accumulator, eval_mean = max(
        final_candidates,
        key=lambda item: item[1],
    )
    return event_file, accumulator, eval_mean


def _point_at(
    accumulator: EventAccumulator,
    tag: str,
    step: int,
) -> float:
    matching = [
        value
        for event_step, value, _ in _series(accumulator, tag)
        if event_step == step
    ]
    if not matching:
        raise ValueError(f"TensorBoard scalar {tag!r} has no value at step {step}")
    return matching[-1]


def _trapezoid(points: list[tuple[int, float, float]]) -> float:
    return sum(
        (right[0] - left[0]) * (left[1] + right[1]) * 0.5
        for left, right in zip(points, points[1:])
    )


def _population_std(values: list[float]) -> float:
    if not values:
        return math.nan
    mean = fmean(values)
    return math.sqrt(fmean((value - mean) ** 2 for value in values))


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    mean = fmean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    fieldnames = list(fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fieldnames} for row in rows
        )


def _collect_run(
    marker: Path,
    repo_root: Path,
    log_root: Path,
    tag: str,
    train_frames: int,
    eval_frames: int,
    frame_skip: int,
    check_wandb: bool,
) -> dict[str, Any]:
    metadata = _parse_key_values(marker)
    if int(metadata.get("exit_code", "-1")) != 0:
        raise ValueError(f"Done marker does not have exit_code=0: {marker}")

    task = metadata["task"]
    algo = metadata["algo"]
    seed = int(metadata["seed"])
    attempt = int(metadata.get("attempt", "1"))
    base_name = f"{task}-{algo}-seed{seed}"
    run_name = base_name if attempt == 1 else f"{base_name}-retry{attempt}"
    run_dir = log_root / algo / tag / task / run_name
    event_file, accumulator, eval_mean = _load_final_event(
        run_dir / "tb",
        train_frames,
    )

    observed_eval_steps = [point[0] for point in eval_mean]
    expected_eval_steps = list(range(0, train_frames + 1, eval_frames))
    if expected_eval_steps[-1] != train_frames:
        expected_eval_steps.append(train_frames)
    if observed_eval_steps != expected_eval_steps:
        raise ValueError(
            f"Unexpected eval schedule in {event_file}: "
            f"observed={observed_eval_steps}, expected={expected_eval_steps}"
        )

    final_step, final_mean, final_wall_time = eval_mean[-1]
    initial_step, initial_mean, initial_wall_time = eval_mean[0]
    if final_step != train_frames:
        raise ValueError(f"Unexpected final step in {event_file}: {final_step}")
    trained_points = [point for point in eval_mean if point[0] > 0]
    best_step, best_mean, _ = max(
        trained_points or eval_mean,
        key=lambda point: point[1],
    )
    auc_raw = _trapezoid(eval_mean)
    covered_frames = final_step - initial_step
    auc_normalized = auc_raw / covered_frames if covered_frames else final_mean
    event_runtime = final_wall_time - initial_wall_time
    elapsed_seconds = float(metadata["elapsed_seconds"])

    row: dict[str, Any] = {
        "task": task,
        "algo": algo,
        "seed": seed,
        "attempt": attempt,
        "wandb_run_id": metadata.get("wandb_run_id", ""),
        "eval_points": len(eval_mean),
        "initial_eval_mean": initial_mean,
        "final_eval_mean": final_mean,
        "best_eval_mean": best_mean,
        "best_eval_step": best_step,
        "eval_auc_normalized": auc_normalized,
        "eval_auc_raw": auc_raw,
        "event_runtime_seconds": event_runtime,
        "raw_frames_per_second": train_frames / event_runtime,
        "policy_transitions_per_second": train_frames / frame_skip / event_runtime,
        "elapsed_seconds": elapsed_seconds,
        "e2e_raw_frames_per_second": train_frames / elapsed_seconds,
        "e2e_policy_transitions_per_second": train_frames
        / frame_skip
        / elapsed_seconds,
        "event_file": str(event_file),
    }
    final_name = {
        "eval/median": "final_eval_median",
        "eval/std": "final_eval_episode_std",
        "eval/min": "final_eval_min",
        "eval/max": "final_eval_max",
        "eval/length": "final_eval_length",
    }
    for tb_tag, field in final_name.items():
        row[field] = _point_at(accumulator, tb_tag, train_frames)
    for tb_tag in SMOOTHNESS_TAGS:
        row[f"final_{tb_tag.removeprefix('eval/')}"] = _point_at(
            accumulator,
            tb_tag,
            train_frames,
        )

    if check_wandb:
        run_log = Path(metadata["log"])
        if not run_log.is_absolute():
            run_log = repo_root / run_log
        if "wandb: Synced" not in run_log.read_text(errors="replace"):
            raise ValueError(f"Missing W&B Synced marker in {run_log}")
        wandb_id = metadata["wandb_run_id"]
        summaries = list(
            run_dir.glob(f"wandb/run-*-{wandb_id}/files/wandb-summary.json")
        )
        if len(summaries) != 1:
            raise ValueError(
                f"Expected one W&B summary for {wandb_id}, found {summaries}"
            )
        summary_path = summaries[0]
        summary = json.loads(summary_path.read_text())
        if int(summary.get("step", -1)) != train_frames:
            raise ValueError(f"W&B summary has wrong final step: {summary_path}")
        for tb_tag in EVAL_TAGS + SMOOTHNESS_TAGS:
            tb_value = _point_at(accumulator, tb_tag, train_frames)
            if tb_tag not in summary or not math.isclose(
                float(summary[tb_tag]),
                tb_value,
                rel_tol=1e-5,
                abs_tol=1e-5,
            ):
                raise ValueError(
                    f"W&B/TensorBoard mismatch for {tb_tag}: {summary_path}"
                )
        row["wandb_summary"] = str(summary_path)
    else:
        row["wandb_summary"] = ""
    return row


def _aggregate(
    run_rows: list[dict[str, Any]],
    tasks: list[str],
    algos: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["task"], row["algo"])].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for task in tasks:
        for algo in algos:
            rows = grouped.get((task, algo), [])
            if not rows:
                continue
            final = [float(row["final_eval_mean"]) for row in rows]
            best = [float(row["best_eval_mean"]) for row in rows]
            auc = [float(row["eval_auc_normalized"]) for row in rows]
            aggregate_rows.append(
                {
                    "task": task,
                    "algo": algo,
                    "n": len(rows),
                    "final_mean": fmean(final),
                    "final_population_std": _population_std(final),
                    "final_sample_std": _sample_std(final),
                    "final_median": median(final),
                    "final_min": min(final),
                    "final_max": max(final),
                    "best_mean": fmean(best),
                    "best_population_std": _population_std(best),
                    "auc_mean": fmean(auc),
                    "auc_population_std": _population_std(auc),
                    "runtime_seconds_mean": fmean(
                        float(row["elapsed_seconds"]) for row in rows
                    ),
                    "raw_frames_per_second_mean": fmean(
                        float(row["raw_frames_per_second"]) for row in rows
                    ),
                    "action_delta_l2_mean": fmean(
                        float(row["final_action_delta_l2_mean"]) for row in rows
                    ),
                    "action_delta_l2_p95": fmean(
                        float(row["final_action_delta_l2_p95"]) for row in rows
                    ),
                    "action_accel_l2_mean": fmean(
                        float(row["final_action_accel_l2_mean"]) for row in rows
                    ),
                    "action_jerk_l2_mean": fmean(
                        float(row["final_action_jerk_l2_mean"]) for row in rows
                    ),
                    "action_hf_power_ratio": fmean(
                        float(row["final_action_hf_power_ratio"]) for row in rows
                    ),
                }
            )

    lookup = {(row["task"], row["algo"]): row for row in aggregate_rows}
    overall_rows: list[dict[str, Any]] = []
    for algo in algos:
        available_tasks = [task for task in tasks if (task, algo) in lookup]
        if not available_tasks:
            continue
        ranks = []
        wins = 0
        for task in available_tasks:
            own_score = float(lookup[(task, algo)]["final_mean"])
            task_scores = [
                float(lookup[(task, other)]["final_mean"])
                for other in algos
                if (task, other) in lookup
            ]
            ranks.append(1 + sum(score > own_score for score in task_scores))
            wins += int(math.isclose(own_score, max(task_scores)))
        overall_rows.append(
            {
                "algo": algo,
                "tasks": len(available_tasks),
                "mean_final_across_tasks": fmean(
                    float(lookup[(task, algo)]["final_mean"])
                    for task in available_tasks
                ),
                "mean_auc_across_tasks": fmean(
                    float(lookup[(task, algo)]["auc_mean"]) for task in available_tasks
                ),
                "average_rank": fmean(ranks),
                "task_wins": wins,
            }
        )
    overall_rows.sort(
        key=lambda row: (
            float(row["average_rank"]),
            -float(row["mean_final_across_tasks"]),
        )
    )
    return aggregate_rows, overall_rows


def _markdown_table(
    title: str,
    tasks: list[str],
    algos: list[str],
    lookup: dict[tuple[str, str], dict[str, Any]],
    mean_field: str,
    std_field: str,
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| task | " + " | ".join(algos) + " |",
        "|---|" + "|".join("---:" for _ in algos) + "|",
    ]
    for task in tasks:
        cells = []
        for algo in algos:
            row = lookup.get((task, algo))
            if row is None:
                cells.append("—")
            else:
                cells.append(
                    f"{float(row[mean_field]):.1f} ± {float(row[std_field]):.1f} "
                    f"(n={row['n']})"
                )
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _write_report(
    path: Path,
    run_log_root: Path,
    train_frames: int,
    frame_skip: int,
    run_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    overall_rows: list[dict[str, Any]],
    tasks: list[str],
    algos: list[str],
    project_url: str | None,
    complete: bool,
) -> None:
    lookup = {(row["task"], row["algo"]): row for row in aggregate_rows}
    lines = [
        "# DMC On-Policy Baseline Summary",
        "",
        f"- Successful runs: {len(run_rows)}",
        f"- Training budget: {train_frames:,} raw DMC frames per run",
        f"- Frame skip: {frame_skip}",
        "- Primary statistic: final `eval/mean` across seeds; `±` is population std (ddof=0).",
        "- Best score excludes the untrained step-0 evaluation.",
        "- Normalized AUC includes step 0 and retains return units.",
        f"- Provenance: `{run_log_root}`",
    ]
    if project_url:
        lines.append(f"- W&B project: {project_url}")
    if not complete:
        lines.extend(
            [
                "",
                "> **PROVISIONAL:** incomplete sweep; sample sizes can differ, "
                "so overall ranking is omitted.",
            ]
        )
    lines.append("")
    lines.extend(
        _markdown_table(
            "Final return",
            tasks,
            algos,
            lookup,
            "final_mean",
            "final_population_std",
        )
    )
    lines.extend(
        _markdown_table(
            "Learning-curve AUC",
            tasks,
            algos,
            lookup,
            "auc_mean",
            "auc_population_std",
        )
    )
    if complete:
        lines.extend(
            [
                "## Overall ranking",
                "",
                "| rank | algo | average task rank | task wins | mean final | mean AUC |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(overall_rows, start=1):
            lines.append(
                f"| {rank} | {row['algo']} | {float(row['average_rank']):.2f} | "
                f"{row['task_wins']} | {float(row['mean_final_across_tasks']):.1f} | "
                f"{float(row['mean_auc_across_tasks']):.1f} |"
            )
    else:
        lines.extend(
            [
                "## Overall ranking",
                "",
                "Omitted until every manifest entry has completed successfully.",
            ]
        )
    lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `runs.csv`: one row per seed/run, including final, best, AUC, smoothness, and throughput.",
            "- `aggregate.csv`: per-task, per-algorithm seed aggregation.",
            "- `overall.csv`: equal-task-weighted ranking (populated only for a complete sweep).",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    args = _parse_args()
    run_log_root = args.run_log_root.resolve()
    run_config = _parse_key_values(run_log_root / "run-config.txt")
    repo_root = Path(run_config["repo_root"]).resolve()
    log_root = (
        args.log_root.resolve()
        if args.log_root
        else (repo_root / run_config["log_dir"]).resolve()
    )
    tag = args.tag or run_config["log_tag"]
    train_frames = args.train_frames or int(run_config["train_frames"])
    eval_frames = int(run_config["eval_frames"])
    frame_skip = args.frame_skip or int(run_config["frame_skip"])

    manifest = _read_manifest(run_log_root / "manifest.tsv")
    tasks = run_config["tasks"].split()
    algos = run_config["algos"].split()
    seeds = [int(seed) for seed in run_config["seeds"].split()]
    manifest_indices = [int(row["index"]) for row in manifest]
    if manifest_indices != list(range(1, len(manifest) + 1)):
        raise RuntimeError("Manifest indices are not contiguous from 1")
    manifest_keys = [(row["task"], row["algo"], int(row["seed"])) for row in manifest]
    if len(set(manifest_keys)) != len(manifest_keys):
        raise RuntimeError("Manifest contains duplicate task/algo/seed rows")
    expected_cartesian = {
        (task, algo, seed) for seed in seeds for task in tasks for algo in algos
    }
    expected = set(manifest_keys)
    if expected != expected_cartesian:
        missing = sorted(expected_cartesian - expected)
        unexpected = sorted(expected - expected_cartesian)
        raise RuntimeError(
            f"Manifest does not match run-config Cartesian product: "
            f"missing={missing}, unexpected={unexpected}"
        )

    done_markers = sorted(run_log_root.glob("*.done"))
    failed_markers = sorted(run_log_root.glob("*.failed"))
    running_markers = sorted(run_log_root.glob("*.running"))
    marker_complete = (
        not failed_markers
        and not running_markers
        and len(done_markers) == len(expected)
    )
    if not args.allow_incomplete and not marker_complete:
        raise RuntimeError(
            f"Sweep incomplete: expected={len(expected)} done={len(done_markers)} "
            f"failed={len(failed_markers)} running={len(running_markers)}"
        )

    run_rows = [
        _collect_run(
            marker=marker,
            repo_root=repo_root,
            log_root=log_root,
            tag=tag,
            train_frames=train_frames,
            eval_frames=eval_frames,
            frame_skip=frame_skip,
            check_wandb=not args.skip_wandb_check,
        )
        for marker in done_markers
    ]
    if not run_rows:
        raise RuntimeError("No successful .done runs to summarize")
    observed_keys = [(row["task"], row["algo"], int(row["seed"])) for row in run_rows]
    observed = set(observed_keys)
    if len(observed) != len(observed_keys):
        raise RuntimeError("Successful markers contain duplicate task/algo/seed runs")
    unexpected = observed - expected
    if unexpected:
        raise RuntimeError(
            f"Done markers not present in manifest: {sorted(unexpected)}"
        )
    missing = expected - observed
    complete = marker_complete and not missing
    if not args.allow_incomplete and not complete:
        raise RuntimeError(
            f"Successful markers are missing manifest rows: {sorted(missing)}"
        )

    order = {key: index for index, key in enumerate(manifest_keys, start=1)}
    run_rows.sort(key=lambda row: order[(row["task"], row["algo"], int(row["seed"]))])
    aggregate_rows, overall_rows = _aggregate(run_rows, tasks, algos)
    if not complete:
        overall_rows = []

    output_dir = (
        args.output_dir.resolve() if args.output_dir else run_log_root / "summary"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "runs.csv", run_rows, RUN_FIELDS)
    aggregate_fields = list(aggregate_rows[0]) if aggregate_rows else []
    _write_csv(output_dir / "aggregate.csv", aggregate_rows, aggregate_fields)
    _write_csv(output_dir / "overall.csv", overall_rows, OVERALL_FIELDS)

    entity = run_config.get("wandb_entity", "")
    project = run_config.get("wandb_project", "")
    project_url = f"https://wandb.ai/{entity}/{project}" if entity and project else None
    _write_report(
        path=output_dir / "report.md",
        run_log_root=run_log_root,
        train_frames=train_frames,
        frame_skip=frame_skip,
        run_rows=run_rows,
        aggregate_rows=aggregate_rows,
        overall_rows=overall_rows,
        tasks=tasks,
        algos=algos,
        project_url=project_url,
        complete=complete,
    )
    print(
        f"Summarized {len(run_rows)}/{len(expected)} successful runs into "
        f"{output_dir} (complete={complete})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
