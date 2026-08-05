#!/usr/bin/env python3
"""Build the best-seed manifest for final IsaacLab baseline videos."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY = (
    ROOT
    / "run_logs"
    / "isaaclab_baseline_final_report_20260617_073508"
    / "data"
    / "eval_history_long.csv.gz"
)
DEFAULT_LOGS_ROOT = ROOT / "logs"
DEFAULT_OUTPUT = (
    ROOT
    / "scripts"
    / "isaaclab"
    / "search_spaces"
    / "isaaclab_baseline_best_final_video72.tsv"
)
ALGORITHMS = ("ppo", "dppo", "fpo", "fpopp", "genpo", "policyflow")
TASK_SLUGS = {
    "Isaac-Ant-v0": "ant",
    "Isaac-Cartpole-v0": "cartpole",
    "Isaac-Humanoid-v0": "humanoid",
    "Isaac-Lift-Cube-Franka-v0": "lift_cube",
    "Isaac-Open-Drawer-Franka-v0": "open_drawer",
    "Isaac-Quadcopter-Direct-v0": "quadcopter",
    "Isaac-Repose-Cube-Shadow-Direct-v0": "repose_cube",
    "Isaac-Velocity-Flat-Anymal-D-v0": "flat_anymal_d",
    "Isaac-Velocity-Flat-G1-v0": "flat_g1",
    "Isaac-Velocity-Rough-G1-v0": "rough_g1",
    "Isaac-Velocity-Rough-H1-v0": "rough_h1",
    "Isaac-Velocity-Rough-Unitree-Go2-v0": "rough_go2",
}
FIELDS = (
    "slug",
    "task",
    "algo",
    "seed",
    "reference_return",
    "reference_frame",
    "selection_status",
    "run_id",
    "commit",
    "metadata_path",
    "config_path",
    "config_sha256",
    "source_status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Restrict to one task. Repeat for more tasks.",
    )
    parser.add_argument(
        "--algo",
        action="append",
        default=[],
        choices=ALGORITHMS,
        help="Restrict to one algorithm. Repeat for more algorithms.",
    )
    parser.add_argument("--terminal-frame", type=int, default=200_000_000)
    return parser.parse_args()


def read_terminal_rows(path: Path) -> dict[str, dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    terminal: dict[str, dict[str, str]] = {}
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"task", "algo", "seed", "run_id", "step", "eval_mean"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        for row in reader:
            run_id = row["run_id"].strip()
            step = float(row["step"])
            score = float(row["eval_mean"])
            if not run_id or not math.isfinite(step) or not math.isfinite(score):
                continue
            previous = terminal.get(run_id)
            if previous is None or step > float(previous["step"]):
                terminal[run_id] = row
    return terminal


def select_rows(
    terminal: dict[str, dict[str, str]],
    *,
    tasks: set[str],
    algos: set[str],
    terminal_frame: int,
) -> list[dict[str, str]]:
    by_cell: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in terminal.values():
        task = row["task"]
        algo = row["algo"]
        if task not in tasks or algo not in algos:
            continue
        by_cell[(task, algo)].append(row)

    selected: list[dict[str, str]] = []
    missing: list[str] = []
    for task in sorted(tasks):
        for algo in ALGORITHMS:
            if algo not in algos:
                continue
            candidates = by_cell.get((task, algo), [])
            if not candidates:
                missing.append(f"{task}/{algo}: no history")
                continue
            complete = [
                row
                for row in candidates
                if float(row["step"]) >= float(terminal_frame)
            ]
            pool = complete or candidates
            status = (
                "strict_200m_terminal_best"
                if complete
                else "historical_incomplete_fallback"
            )
            best = max(
                pool,
                key=lambda row: (float(row["eval_mean"]), -int(row["seed"])),
            ).copy()
            best["selection_status"] = status
            selected.append(best)
    if missing:
        raise RuntimeError("Missing task/algo cells:\n" + "\n".join(missing))
    return selected


def metadata_for_run(logs_root: Path, algo: str, run_id: str) -> Path:
    base = logs_root / algo / "isaaclab-baseline-final-200m"
    matches = list(
        base.glob(f"**/wandb/run-*-{run_id}/files/wandb-metadata.json")
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one metadata file for {algo}/{run_id}; got {matches}"
        )
    return matches[0].resolve()


def resolved_config_for_metadata(
    metadata_path: Path,
    *,
    task: str,
    algo: str,
    seed: int,
) -> tuple[Path, str]:
    config_path = metadata_path.parents[3] / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    actual = {
        "task": str(config.get("task", "")),
        "algo": str(config.get("algo", {}).get("name", "")),
        "seed": int(config.get("seed", -1)),
        "train_frames": int(config.get("train_frames", 0)),
    }
    expected = {
        "task": task,
        "algo": algo,
        "seed": seed,
        "train_frames": 200_000_000,
    }
    if actual != expected:
        raise ValueError(
            f"{config_path}: resolved config mismatch: {actual} != {expected}"
        )
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return config_path.resolve(), digest



def relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def validate_metadata(
    path: Path,
    *,
    task: str,
    algo: str,
    seed: int,
) -> tuple[str, dict[str, str]]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    arguments = {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in metadata.get("args", [])
        if "=" in item
    }
    expected = {"task": task, "algo": algo, "seed": str(seed)}
    actual = {key: arguments.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"{path}: expected {expected}, got {actual}")
    if int(arguments.get("train_frames", "0")) != 200_000_000:
        raise ValueError(f"{path}: train_frames is not 200M")
    if algo == "genpo":
        required_genpo = {
            "algo.batch_size": "4096",
            "algo.num_minibatches": "6",
        }
        actual_genpo = {key: arguments.get(key) for key in required_genpo}
        if actual_genpo != required_genpo:
            raise ValueError(
                f"{path}: GENPO overrides differ: {actual_genpo}"
            )
    commit = str(metadata.get("git", {}).get("commit", "")).strip()
    if not commit:
        commit = "worktree-uncommitted"
    return commit, arguments


def main() -> None:
    args = parse_args()
    if args.terminal_frame <= 0:
        raise ValueError("--terminal-frame must be positive")
    history = args.history.resolve()
    logs_root = args.logs_root.resolve()
    tasks = set(args.task or TASK_SLUGS)
    unknown_tasks = tasks - set(TASK_SLUGS)
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {sorted(unknown_tasks)}")
    algos = set(args.algo or ALGORITHMS)

    terminal = read_terminal_rows(history)
    selected = select_rows(
        terminal,
        tasks=tasks,
        algos=algos,
        terminal_frame=args.terminal_frame,
    )

    output_rows: list[dict[str, str | int]] = []
    fallback_cells: list[str] = []
    for row in selected:
        task = row["task"]
        algo = row["algo"]
        seed = int(row["seed"])
        run_id = row["run_id"]
        metadata_path = metadata_for_run(logs_root, algo, run_id)
        commit, _ = validate_metadata(
            metadata_path,
            task=task,
            algo=algo,
            seed=seed,
        )
        config_path, config_sha256 = resolved_config_for_metadata(
            metadata_path,
            task=task,
            algo=algo,
            seed=seed,
        )
        status = row["selection_status"]
        if status != "strict_200m_terminal_best":
            fallback_cells.append(f"{task}/{algo}")
        output_rows.append(
            {
                "slug": f"{TASK_SLUGS[task]}__{algo}",
                "task": task,
                "algo": algo,
                "seed": seed,
                "reference_return": f"{float(row['eval_mean']):.10g}",
                "reference_frame": str(int(float(row["step"]))),
                "selection_status": status,
                "run_id": run_id,
                "commit": commit,
                "metadata_path": relative_to_root(metadata_path),
                "config_path": relative_to_root(config_path),
                "config_sha256": config_sha256,
                "source_status": (
                    "best_available_uncommitted_policyflow_overlay"
                    if algo == "policyflow"
                    else (
                        "recorded_commit_agent_plus_compat_symbol_overlay"
                        if algo == "fpo"
                        else "recorded_commit_agent_source"
                    )
                ),
            }
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {output}")
    print(
        f"Strict terminal selections: {len(output_rows) - len(fallback_cells)}; "
        f"incomplete-history fallbacks: {len(fallback_cells)}"
    )
    for cell in fallback_cells:
        print(f"FALLBACK {cell}")


if __name__ == "__main__":
    main()
