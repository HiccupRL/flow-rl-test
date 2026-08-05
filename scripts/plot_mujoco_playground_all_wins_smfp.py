#!/usr/bin/env python3
"""Plot every Playground task where the selected ours result beats baselines.

For each task, this script scans completed 30M Playground logs and selects the
single configuration with the highest cross-seed mean best-so-far return.
Configurations must have at least two completed seeds in the same run batch.
It produces both a strict view (ours beats every baseline) and a permissive
view (ours beats at least one baseline), plus a ten-task audit table and plot.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import plot_mujoco_playground_best5_smfp as best5  # noqa: E402

from paper_plot_style import (  # noqa: E402
    BAR_BLUE,
    add_panel_title,
    apply_style,
    configure_axes,
    plot_bar_series,
    plot_learning_curve,
    save_figure,
    style_labels,
)


TASK_METADATA = (
    ("BallInCup", "ball_in_cup-catch", "Ball in Cup", "Ball in\nCup"),
    (
        "CartpoleBalance",
        "cartpole-balance",
        "Cartpole Balance",
        "Cartpole\nBalance",
    ),
    ("CheetahRun", "cheetah-run", "Cheetah Run", "Cheetah\nRun"),
    ("FingerSpin", "finger-spin", "Finger Spin", "Finger\nSpin"),
    (
        "FingerTurnEasy",
        "finger-turn_easy",
        "Finger Turn Easy",
        "Finger Turn\nEasy",
    ),
    (
        "FingerTurnHard",
        "finger-turn_hard",
        "Finger Turn Hard",
        "Finger Turn\nHard",
    ),
    ("FishSwim", "fish-swim", "Fish Swim", "Fish\nSwim"),
    ("PointMass", "point_mass-easy", "Point Mass", "Point\nMass"),
    ("ReacherEasy", "reacher-easy", "Reacher Easy", "Reacher\nEasy"),
    ("ReacherHard", "reacher-hard", "Reacher Hard", "Reacher\nHard"),
)
TASK_BY_PREFIX = {prefix: task for task, prefix, _, _ in TASK_METADATA}
PANEL_TITLE = {task: title for task, _, title, _ in TASK_METADATA}
SHORT_TITLE = {task: title for task, _, _, title in TASK_METADATA}
TASK_ORDER = tuple(task for task, _, _, _ in TASK_METADATA)
METHODS = ("ours",) + best5.BASELINE_METHODS
MIN_COMPLETE_SEEDS = 2
MIN_FINAL_FRAME_M = 30.0
MAX_FRAME_M = 30.1


@dataclass(frozen=True)
class SelectedRun:
    task: str
    prefix: str
    panel_title: str
    short_title: str
    config_id: str
    batch: str
    log_paths: tuple[Path, ...]


def make_ours_result(log_paths: tuple[Path, ...]) -> best5.OursResult:
    seeds: list[int] = []
    curves: list[np.ndarray] = []
    common_x: np.ndarray | None = None

    for path in log_paths:
        seed_match = best5.SEED_PATTERN.search(path.name)
        if seed_match is None:
            raise ValueError(f"Could not parse seed from {path}")
        x, y = best5.parse_eval_log(path)
        keep = x <= MAX_FRAME_M
        x = x[keep]
        y = y[keep]
        if x[-1] < MIN_FINAL_FRAME_M:
            raise ValueError(f"Selected log does not reach 30M: {path}")
        if common_x is None:
            common_x = x
        elif common_x.shape != x.shape or not np.allclose(common_x, x):
            raise ValueError(f"Evaluation grid mismatch in {path.parent}")
        seeds.append(int(seed_match.group(1)))
        curves.append(y)

    assert common_x is not None
    matrix = np.maximum.accumulate(np.vstack(curves), axis=1)
    final_values = matrix[:, -1]
    best_index = int(np.argmax(final_values))
    return best5.OursResult(
        curve=best5.Curve(
            x=common_x,
            mean=np.mean(matrix, axis=0),
            sd=np.std(matrix, axis=0, ddof=1),
            n=np.full(common_x.shape, len(curves), dtype=int),
        ),
        seeds=tuple(seeds),
        final_values=final_values,
        best_seed=seeds[best_index],
        best_seed_value=float(final_values[best_index]),
    )


def select_best_completed_runs(
    final_root: Path,
) -> tuple[dict[str, SelectedRun], dict[str, best5.OursResult]]:
    grouped: dict[
        tuple[str, str, str, str],
        list[tuple[Path, bool]],
    ] = defaultdict(list)

    for path in final_root.glob(
        "run_logs/mujoco_playground*/**/machine_0/*.log"
    ):
        prefix = path.name.split("__", 1)[0]
        task = TASK_BY_PREFIX.get(prefix)
        seed_match = best5.SEED_PATTERN.search(path.name)
        if task is None or seed_match is None:
            continue
        config_id = path.name[: seed_match.start()].rsplit("__", 1)[-1]
        x, _ = best5.parse_eval_log(path)
        complete = bool(x[-1] >= MIN_FINAL_FRAME_M)
        grouped[
            (task, prefix, path.parents[1].name, config_id)
        ].append((path, complete))

    candidates: dict[str, list[tuple[float, SelectedRun, best5.OursResult]]] = (
        defaultdict(list)
    )
    for (task, prefix, batch, config_id), path_records in grouped.items():
        if not all(complete for _, complete in path_records):
            continue
        paths = tuple(
            sorted(
                (path for path, _ in path_records),
                key=lambda path: int(
                    best5.SEED_PATTERN.search(path.name).group(1)  # type: ignore[union-attr]
                ),
            )
        )
        if len(paths) < MIN_COMPLETE_SEEDS:
            continue
        result = make_ours_result(paths)
        selected = SelectedRun(
            task=task,
            prefix=prefix,
            panel_title=PANEL_TITLE[task],
            short_title=SHORT_TITLE[task],
            config_id=config_id,
            batch=batch,
            log_paths=paths,
        )
        candidates[task].append(
            (float(np.mean(result.final_values)), selected, result)
        )

    selected_runs: dict[str, SelectedRun] = {}
    ours: dict[str, best5.OursResult] = {}
    for task in TASK_ORDER:
        if not candidates[task]:
            raise ValueError(f"No completed multi-seed ours configuration for {task}")
        _, selected, result = max(
            candidates[task],
            key=lambda item: item[0],
        )
        selected_runs[task] = selected
        ours[task] = result
    return selected_runs, ours


def make_task_specs(
    selected_runs: dict[str, SelectedRun],
    final_root: Path,
) -> tuple[best5.TaskSpec, ...]:
    specs: list[best5.TaskSpec] = []
    for task in TASK_ORDER:
        selected = selected_runs[task]
        log_dir = selected.log_paths[0].parent.relative_to(final_root)
        specs.append(
            best5.TaskSpec(
                task=task,
                panel_title=selected.panel_title,
                short_title=selected.short_title,
                config_id=selected.config_id,
                log_dir=str(log_dir),
                log_glob=(
                    f"{selected.prefix}*__{selected.config_id}__seed*.log"
                ),
                genpo_reference=0.0,
            )
        )
    return tuple(specs)


def build_rows(
    selected_runs: dict[str, SelectedRun],
    ours: dict[str, best5.OursResult],
    peak_values: dict[tuple[str, str], list[float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []

    for task in TASK_ORDER:
        result = ours[task]
        selected = selected_runs[task]
        ours_mean = float(np.mean(result.final_values))
        ours_sd = best5.sample_sd(result.final_values)
        baseline_stats: dict[str, tuple[float, float, int]] = {}
        for method in best5.BASELINE_METHODS:
            values = peak_values[(task, method)]
            baseline_stats[method] = (
                float(np.mean(values)),
                best5.sample_sd(values),
                len(values),
            )
        strongest_method = max(
            best5.BASELINE_METHODS,
            key=lambda method: baseline_stats[method][0],
        )
        strongest_mean, strongest_sd, strongest_n = baseline_stats[
            strongest_method
        ]
        beaten_methods = tuple(
            method
            for method in best5.BASELINE_METHODS
            if ours_mean > baseline_stats[method][0]
        )
        strict_winner = len(beaten_methods) == len(best5.BASELINE_METHODS)
        broad_winner = bool(beaten_methods)

        summary: dict[str, object] = {
            "task": task,
            "selected_config": selected.config_id,
            "selected_batch": selected.batch,
            "ours_seeds": ",".join(map(str, result.seeds)),
            "ours_n": len(result.seeds),
            "ours_mean": ours_mean,
            "ours_sd": ours_sd,
            "strongest_baseline": strongest_method,
            "strongest_baseline_mean": strongest_mean,
            "strongest_baseline_sd": strongest_sd,
            "strongest_baseline_n": strongest_n,
            "gain_vs_strongest_percent": 100.0 * (ours_mean / strongest_mean - 1.0),
            "beaten_baselines": ",".join(beaten_methods),
            "num_beaten_baselines": len(beaten_methods),
            "strict_winner": strict_winner,
            "broad_winner": broad_winner,
        }
        for method in best5.BASELINE_METHODS:
            mean, sd, n = baseline_stats[method]
            summary[f"{method}_mean"] = mean
            summary[f"{method}_sd"] = sd
            summary[f"{method}_n"] = n
            long_rows.append(
                {
                    "task": task,
                    "method": method,
                    "mean": mean,
                    "sd": sd,
                    "n": n,
                    "strict_winner_task": strict_winner,
                    "broad_winner_task": broad_winner,
                }
            )
        long_rows.append(
            {
                "task": task,
                "method": "ours",
                "mean": ours_mean,
                "sd": ours_sd,
                "n": len(result.seeds),
                "strict_winner_task": strict_winner,
                "broad_winner_task": broad_winner,
            }
        )
        summary_rows.append(summary)
    return summary_rows, long_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def panel_y_top(
    task: str,
    ours: dict[str, best5.OursResult],
    baselines: dict[tuple[str, str], best5.Curve],
) -> float:
    tops = [float(np.max(ours[task].curve.mean + ours[task].curve.sd))]
    for method in best5.BASELINE_METHODS:
        curve = baselines[(task, method)]
        tops.append(float(np.max(curve.mean + curve.sd)))
    raw_top = max(tops) * 1.03
    interval = 100.0 if raw_top > 400.0 else 50.0
    return max(interval * 2.0, math.ceil(raw_top / interval) * interval)


def draw_task_panel(
    ax: plt.Axes,
    task: str,
    panel_index: int,
    ours: dict[str, best5.OursResult],
    baselines: dict[tuple[str, str], best5.Curve],
) -> None:
    for method in best5.BASELINE_METHODS:
        curve = baselines[(task, method)]
        plot_learning_curve(
            ax,
            curve.x,
            curve.mean,
            method=method,
            lower=curve.mean - curve.sd,
            upper=curve.mean + curve.sd,
            label=method,
            profile="main",
            markevery=1,
            zorder=2,
        )
    curve = ours[task].curve
    plot_learning_curve(
        ax,
        curve.x,
        curve.mean,
        method="ours",
        lower=curve.mean - curve.sd,
        upper=curve.mean + curve.sd,
        label="ours",
        profile="main",
        markevery=1,
        zorder=6,
    )
    add_panel_title(ax, panel_index, PANEL_TITLE[task], profile="main")
    configure_axes(ax, profile="main", grid_axis="both", hide_top=False)
    y_top = panel_y_top(task, ours, baselines)
    ax.set_xlim(-0.5, 30.5)
    ax.set_ylim(0.0, y_top)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks(np.linspace(0.0, y_top, 5))


def plot_learning_grid(
    output_path: Path,
    preview_path: Path,
    tasks: list[str],
    ours: dict[str, best5.OursResult],
    baselines: dict[tuple[str, str], best5.Curve],
) -> None:
    apply_style("main")
    if len(tasks) <= 5:
        rows, cols = 1, len(tasks)
        figsize = (4.8 * cols, 5.2)
        legend_panel = False
    else:
        rows, cols = 2, 5
        figsize = (24.0, 9.0)
        legend_panel = len(tasks) < rows * cols

    fig, axes_array = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = list(axes_array.flat)
    for panel_index, task in enumerate(tasks):
        draw_task_panel(axes[panel_index], task, panel_index, ours, baselines)

    handles, labels = axes[0].get_legend_handles_labels()
    if legend_panel:
        legend_ax = axes[len(tasks)]
        legend_ax.axis("off")
        legend_ax.legend(
            handles[::-1],
            labels[::-1],
            loc="center",
            frameon=False,
            ncol=2,
            prop={
                "family": "Times New Roman",
                "weight": "bold",
                "size": 18,
            },
            handlelength=2.4,
            columnspacing=1.8,
        )
        for ax in axes[len(tasks) + 1 :]:
            ax.axis("off")
        top = 0.94
    else:
        fig.legend(
            handles[::-1],
            labels[::-1],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            frameon=False,
            ncol=5,
            prop={
                "family": "Times New Roman",
                "weight": "bold",
                "size": 18,
            },
            handlelength=2.4,
            columnspacing=2.0,
        )
        top = 0.76

    fig.supxlabel(
        "Environment Steps (millions)",
        fontsize=20,
        fontweight="bold",
        fontfamily="Times New Roman",
        y=0.025,
    )
    fig.supylabel(
        "Best-so-far Return",
        fontsize=20,
        fontweight="bold",
        fontfamily="Times New Roman",
        x=0.025,
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        bottom=0.13 if rows == 2 else 0.17,
        top=top,
        wspace=0.25,
        hspace=0.48,
    )
    save_figure(
        fig,
        output_path,
        preview_png=preview_path,
        transparent=True,
        pad_inches=0.03,
    )
    plt.close(fig)


def plot_all10_summary(
    output_dir: Path,
    summary_rows: list[dict[str, object]],
    long_rows: list[dict[str, object]],
) -> None:
    apply_style("compact")
    fig, ax = plt.subplots(figsize=(19.0, 4.6))
    lookup = {
        (str(row["task"]), str(row["method"])): row for row in long_rows
    }
    strongest = {
        str(row["task"]): float(row["strongest_baseline_mean"])
        for row in summary_rows
    }
    methods = ("ours",) + best5.BASELINE_METHODS
    x = np.arange(len(TASK_ORDER), dtype=float)
    width = 0.15
    offsets = (
        np.arange(len(methods)) - (len(methods) - 1) / 2.0
    ) * width
    colors = {
        "ours": BAR_BLUE,
        "PPO": "#D9D9D9",
        "DPPO": "#BDBDBD",
        "FPO": "#969696",
        "GenPO": "#737373",
    }
    hatches = {
        "ours": "///",
        "PPO": "",
        "DPPO": "..",
        "FPO": "xx",
        "GenPO": "--",
    }

    for method, offset in zip(methods, offsets, strict=True):
        means = []
        sds = []
        for task in TASK_ORDER:
            row = lookup[(task, method)]
            denominator = strongest[task]
            means.append(float(row["mean"]) / denominator)
            sds.append(float(row["sd"]) / denominator)
        plot_bar_series(
            ax,
            x,
            means,
            label=method,
            width=width,
            offset=float(offset),
            yerr=sds,
            highlight=method == "ours",
            color=colors[method],
            hatch=hatches[method],
            error_color="#000000" if method == "ours" else "#555555",
            zorder=5 if method == "ours" else 3,
        )

    ax.axhline(1.0, color="#111111", linestyle="--", linewidth=1.1, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_TITLE[task] for task in TASK_ORDER])
    ax.set_ylim(0.0, 1.45)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    style_labels(ax, profile="compact", ylabel="Peak Return / Best Baseline")
    configure_axes(ax, profile="compact", grid_axis="y", hide_top=False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=5,
        frameon=False,
        prop={"family": "Times New Roman", "weight": "bold", "size": 11},
        columnspacing=1.2,
        handlelength=1.8,
    )
    fig.subplots_adjust(left=0.055, right=0.997, bottom=0.22, top=0.82)
    save_figure(
        fig,
        output_dir / "all10_sample_efficiency.pdf",
        preview_png=output_dir / "all10_sample_efficiency.png",
        transparent=True,
        pad_inches=0.03,
    )
    plt.close(fig)


def write_caption(
    output_dir: Path,
    selected_runs: dict[str, SelectedRun],
    ours: dict[str, best5.OursResult],
    strict_tasks: list[str],
    broad_tasks: list[str],
) -> None:
    selections = "; ".join(
        (
            f"{PANEL_TITLE[task]}: `{selected_runs[task].config_id}` "
            f"(seeds {','.join(map(str, ours[task].seeds))})"
        )
        for task in TASK_ORDER
    )
    strict_names = ", ".join(PANEL_TITLE[task] for task in strict_tasks)
    broad_names = ", ".join(PANEL_TITLE[task] for task in broad_tasks)
    lines = [
        "# Figure captions",
        "",
        (
            "**Strict winners.** Best-so-far 0–30M learning curves for every "
            "task where the selected `ours` mean exceeds the mean of all four "
            "baselines at the common 30M endpoint. Curves show mean ±1 sample "
            f"standard deviation. Included tasks: {strict_names}."
        ),
        "",
        (
            "**At-least-one-baseline view.** The same curves for every task "
            "where `ours` exceeds at least one baseline mean. Inclusion does "
            f"not imply that `ours` is best overall. Included tasks: {broad_names}."
        ),
        "",
        (
            "**All-task summary.** Per-run/seed 30M best-so-far return, "
            "normalized by the strongest baseline mean in each task. Error "
            "bars show ±1 sample standard deviation with the same normalization."
        ),
        "",
        (
            "**Data alignment.** Baselines are linearly interpolated run-by-run "
            "onto the exact `ours` evaluation grid and truncated at approximately "
            "30M frames; no run is extrapolated past its final checkpoint."
        ),
        "",
        (
            "**Post-hoc selection.** Within each task, `ours` is the completed "
            "same-batch configuration with at least two seeds and the largest "
            "cross-seed mean best-so-far return. Selected runs: "
            f"{selections}."
        ),
        "",
        (
            "**Scope.** These are development-set comparisons with unequal "
            "numbers of `ours` seeds across tasks. They are useful for diagnosis "
            "and figure selection, but are not a frozen held-out benchmark."
        ),
        "",
    ]
    (output_dir / "CAPTION.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[1]
    default_final_root = workspace_root.parent / "flow-rl-final"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-root",
        type=Path,
        default=Path(os.environ.get("FLOW_RL_FINAL_ROOT", default_final_root)),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            workspace_root
            / "artifacts"
            / "mujoco_playground_all_tasks_smfp"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_root = args.final_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_runs, ours = select_best_completed_runs(final_root)
    task_specs = make_task_specs(selected_runs, final_root)
    target_x = ours[TASK_ORDER[0]].curve.x
    for task in TASK_ORDER[1:]:
        curve_x = ours[task].curve.x
        if curve_x.shape != target_x.shape or not np.allclose(curve_x, target_x):
            raise ValueError(f"Selected ours grid differs for {task}")

    baselines, peak_values = best5.load_baselines(
        final_root / "playground_baselines_10tasks_raw.csv",
        target_x,
        task_specs=task_specs,
    )
    summary_rows, long_rows = build_rows(
        selected_runs,
        ours,
        peak_values,
    )
    strict_tasks = [
        str(row["task"]) for row in summary_rows if row["strict_winner"]
    ]
    broad_tasks = [
        str(row["task"]) for row in summary_rows if row["broad_winner"]
    ]

    write_csv(output_dir / "all10_comparison.csv", summary_rows)
    write_csv(output_dir / "all10_comparison_long.csv", long_rows)
    write_csv(
        output_dir / "strict_winners.csv",
        [row for row in summary_rows if row["strict_winner"]],
    )
    write_csv(
        output_dir / "above_at_least_one_baseline.csv",
        [row for row in summary_rows if row["broad_winner"]],
    )
    plot_learning_grid(
        output_dir / "strict_winners_learning_curves.pdf",
        output_dir / "strict_winners_learning_curves.png",
        strict_tasks,
        ours,
        baselines,
    )
    plot_learning_grid(
        output_dir / "above_at_least_one_learning_curves.pdf",
        output_dir / "above_at_least_one_learning_curves.png",
        broad_tasks,
        ours,
        baselines,
    )
    plot_all10_summary(output_dir, summary_rows, long_rows)
    write_caption(
        output_dir,
        selected_runs,
        ours,
        strict_tasks,
        broad_tasks,
    )

    print(f"Wrote all-task audit figures and tables to {output_dir}")
    for row in summary_rows:
        print(
            f"{row['task']}: ours={float(row['ours_mean']):.3f}; "
            f"best={row['strongest_baseline']} "
            f"{float(row['strongest_baseline_mean']):.3f}; "
            f"gain={float(row['gain_vs_strongest_percent']):+.2f}%; "
            f"beats={row['beaten_baselines'] or 'none'}"
        )


if __name__ == "__main__":
    main()
