#!/usr/bin/env python3
"""Plot peak Playground performance in SMFP paper style.

Each learning curve is the cross-seed mean and standard deviation of the
per-seed best-so-far return.  The summary uses each seed/run's maximum observed
return within the displayed frame budget, again reporting mean ± sample s.d.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


STYLE_HELPER_DIR = Path("/root/.codex/skills/smfp-plot/scripts")
sys.path.insert(0, str(STYLE_HELPER_DIR))

from paper_plot_style import (  # noqa: E402
    BAR_BLUE,
    METHOD_STYLES,
    add_panel_title,
    apply_style,
    configure_axes,
    plot_bar_series,
    plot_learning_curve,
    save_figure,
    style_labels,
)


EVAL_PATTERN = re.compile(
    r"\[EVAL\]\s+frame=(\d+)\s+mean=([-+0-9.eE]+)"
)
SEED_PATTERN = re.compile(r"__seed(\d+)\.log$")
BASELINE_METHODS = ("PPO", "DPPO", "FPO", "GenPO")
BASELINE_INTERPOLATION_CEILING_M = 33.0
OURS_DISPLAY_NAME = "ATP(Ours)"

# The bundled helper does not define these Playground-specific baselines.
# Register a stable mapping, while retaining the helper's SMFP blue emphasis.
METHOD_STYLES.update(
    {
        "PPO": {"color": "#7F7F7F", "marker": "o", "linestyle": "-"},
        "DPPO": {"color": "#9467BD", "marker": "D", "linestyle": "--"},
        "FPO": {"color": "#FF7F0E", "marker": "s", "linestyle": "-."},
        "GenPO": {"color": "#2CA02C", "marker": "^", "linestyle": ":"},
    }
)


@dataclass(frozen=True)
class TaskSpec:
    task: str
    panel_title: str
    short_title: str
    config_id: str
    log_dir: str
    log_glob: str
    genpo_reference: float


TASK_SPECS = (
    TaskSpec(
        task="CartpoleBalance",
        panel_title="Cartpole Balance",
        short_title="Cartpole\nBalance",
        config_id="gain_b",
        log_dir=(
            "run_logs/mujoco_playground_weak5_gain20_30m/"
            "playground_weak5_gain20_seed67_30m/machine_0"
        ),
        log_glob="cartpole-balance*__gain_b__seed*.log",
        genpo_reference=999.8,
    ),
    TaskSpec(
        task="FingerTurnHard",
        panel_title="Finger Turn Hard",
        short_title="Finger Turn\nHard",
        config_id="ab3_relaxed",
        log_dir=(
            "run_logs/mujoco_playground_ours60_actionbound13_30m/"
            "playground_ours_ab13_6task60_seed0to4_30m/machine_0"
        ),
        log_glob="finger-turn_hard*__ab3_relaxed__seed*.log",
        genpo_reference=479.46927614285715,
    ),
    TaskSpec(
        task="PointMass",
        panel_title="Point Mass",
        short_title="Point Mass",
        config_id="ema990_lr0",
        log_dir=(
            "run_logs/mujoco_playground_pointmass_ema_lr8_30m/"
            "playground_pointmass_ema_lr8_seed2021_30m/machine_0"
        ),
        log_glob="point_mass-easy*__ema990_lr0__seed*.log",
        genpo_reference=883.8,
    ),
    TaskSpec(
        task="ReacherEasy",
        panel_title="Reacher Easy",
        short_title="Reacher\nEasy",
        config_id="ess05",
        log_dir=(
            "run_logs/mujoco_playground_weak2_source12_30m/"
            "playground_weak2_source12_seed1617_30m/machine_0"
        ),
        log_glob="reacher-easy*__ess05__seed*.log",
        genpo_reference=943.9,
    ),
    TaskSpec(
        task="ReacherHard",
        panel_title="Reacher Hard",
        short_title="Reacher\nHard",
        config_id="candidate",
        log_dir=(
            "run_logs/mujoco_playground_weak4_fallback16_30m/"
            "playground_weak4_fallback16_seed1213_30m/machine_0"
        ),
        log_glob="reacher-hard*__candidate__seed*.log",
        genpo_reference=926.3,
    ),
)


@dataclass
class Curve:
    x: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    n: np.ndarray


@dataclass
class OursResult:
    curve: Curve
    seeds: tuple[int, ...]
    final_values: np.ndarray
    best_seed: int
    best_seed_value: float


def sample_sd(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=float)
    if len(array) < 2:
        return 0.0
    return float(np.std(array, ddof=1))


def parse_eval_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    matches = EVAL_PATTERN.findall(path.read_text(errors="replace"))
    if not matches:
        raise ValueError(f"No [EVAL] records found in {path}")
    frames = np.asarray([int(frame) for frame, _ in matches], dtype=float)
    means = np.asarray([float(value) for _, value in matches], dtype=float)
    if np.any(np.diff(frames) <= 0):
        raise ValueError(f"Evaluation frames are not strictly increasing in {path}")
    return frames / 1_000_000.0, means


def load_ours(final_root: Path, spec: TaskSpec) -> OursResult:
    log_paths = sorted((final_root / spec.log_dir).glob(spec.log_glob))
    if len(log_paths) < 2:
        raise ValueError(
            f"Expected at least two selected logs for {spec.task}, found "
            f"{len(log_paths)} under {final_root / spec.log_dir}"
        )

    seeds: list[int] = []
    curves: list[np.ndarray] = []
    common_x: np.ndarray | None = None
    for path in log_paths:
        match = SEED_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"Could not parse seed from {path.name}")
        seeds.append(int(match.group(1)))
        x, y = parse_eval_log(path)
        if common_x is None:
            common_x = x
        elif common_x.shape != x.shape or not np.allclose(common_x, x):
            raise ValueError(f"Selected seed curves are not aligned for {spec.task}")
        curves.append(y)

    assert common_x is not None
    matrix = np.maximum.accumulate(np.vstack(curves), axis=1)
    final_values = matrix[:, -1]
    best_index = int(np.argmax(final_values))
    return OursResult(
        curve=Curve(
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


def load_baselines(
    csv_path: Path,
    target_x: np.ndarray,
    task_specs: tuple[TaskSpec, ...] = TASK_SPECS,
) -> tuple[dict[tuple[str, str], Curve], dict[tuple[str, str], list[float]]]:
    task_names = {spec.task for spec in task_specs}
    run_points: dict[
        tuple[str, str, int, str], list[tuple[float, float]]
    ] = defaultdict(list)

    with csv_path.open(newline="") as file:
        for row in csv.DictReader(file):
            task = row["task"]
            method = row["method"]
            if task not in task_names or method not in BASELINE_METHODS:
                continue
            step = float(row["steps_million"])
            reward = float(row["eval_reward_mean"])
            if step <= BASELINE_INTERPOLATION_CEILING_M:
                run_points[
                    (task, method, int(row["seed"]), row["run_id"])
                ].append((step, reward))

    curves: dict[tuple[str, str], Curve] = {}
    peak_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for task in task_names:
        for method in BASELINE_METHODS:
            selected_runs = {
                key: sorted(points)
                for key, points in run_points.items()
                if key[0] == task and key[1] == method
            }
            if not selected_runs:
                raise ValueError(f"No baseline data found for {task}/{method}")

            best_so_far_rows: list[np.ndarray] = []
            for points in selected_runs.values():
                source_x = np.asarray([step for step, _ in points], dtype=float)
                source_y = np.asarray([reward for _, reward in points], dtype=float)
                valid = (target_x >= source_x[0]) & (target_x <= source_x[-1])
                row = np.full(target_x.shape, np.nan, dtype=float)
                row[valid] = np.maximum.accumulate(
                    np.interp(target_x[valid], source_x, source_y)
                )
                best_so_far_rows.append(row)
                if np.isfinite(row[-1]):
                    peak_values[(task, method)].append(float(row[-1]))

            matrix = np.vstack(best_so_far_rows)
            means: list[float] = []
            sds: list[float] = []
            counts: list[int] = []
            for index in range(matrix.shape[1]):
                values = matrix[:, index]
                values = values[np.isfinite(values)]
                if len(values) == 0:
                    raise ValueError(
                        f"No baseline run brackets {target_x[index]:g}M for "
                        f"{task}/{method}"
                    )
                means.append(float(np.mean(values)))
                sds.append(sample_sd(values))
                counts.append(len(values))
            curves[(task, method)] = Curve(
                x=target_x.copy(),
                mean=np.asarray(means, dtype=float),
                sd=np.asarray(sds, dtype=float),
                n=np.asarray(counts, dtype=int),
            )
            if not peak_values[(task, method)]:
                raise ValueError(
                    f"No {task}/{method} run brackets the 30M target grid endpoint"
                )

    return curves, peak_values


def plot_learning_curves(
    output_dir: Path,
    ours: dict[str, OursResult],
    baselines: dict[tuple[str, str], Curve],
) -> None:
    apply_style("main")
    fig, axes = plt.subplots(1, 5, figsize=(24.0, 5.2), sharex=True, sharey=True)

    for panel_index, spec in enumerate(TASK_SPECS):
        ax = axes[panel_index]
        for method in BASELINE_METHODS:
            curve = baselines[(spec.task, method)]
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

        curve = ours[spec.task].curve
        plot_learning_curve(
            ax,
            curve.x,
            curve.mean,
            method="ours",
            lower=curve.mean - curve.sd,
            upper=curve.mean + curve.sd,
            label=OURS_DISPLAY_NAME,
            profile="main",
            markevery=1,
            zorder=6,
        )
        add_panel_title(ax, panel_index, spec.panel_title, profile="main")
        configure_axes(ax, profile="main", grid_axis="both", hide_top=False)
        ax.set_xlim(-0.5, 30.5)
        ax.set_ylim(0.0, 1030.0)
        ax.set_xticks([0, 10, 20, 30])
        ax.set_yticks([0, 250, 500, 750, 1000])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        labels[::-1],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        ncol=5,
        prop={"family": "Times New Roman", "weight": "bold", "size": 18},
        handlelength=2.4,
        columnspacing=2.0,
    )

    fig.supxlabel(
        "Environment Steps (millions)",
        fontsize=20,
        fontweight="bold",
        fontfamily="Times New Roman",
        y=0.025,
    )
    fig.supylabel(
        "Return",
        fontsize=20,
        fontweight="bold",
        fontfamily="Times New Roman",
        x=0.025,
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        bottom=0.17,
        top=0.76,
        wspace=0.14,
    )
    save_figure(
        fig,
        output_dir / "best5_learning_curves.pdf",
        preview_png=output_dir / "best5_learning_curves.png",
        transparent=True,
        pad_inches=0.03,
    )
    plt.close(fig)


def build_summary_rows(
    ours: dict[str, OursResult],
    peak_values: dict[tuple[str, str], list[float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    long_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for spec in TASK_SPECS:
        ours_result = ours[spec.task]
        ours_values = ours_result.final_values
        ours_mean = float(np.mean(ours_values))
        ours_sd = sample_sd(ours_values)
        ours_frame = float(ours_result.curve.x[-1])

        baseline_stats: dict[str, tuple[float, float, int]] = {}
        for method in BASELINE_METHODS:
            values = peak_values[(spec.task, method)]
            mean = float(np.mean(values))
            sd = sample_sd(values)
            n = len(values)
            baseline_stats[method] = (mean, sd, n)
            long_rows.append(
                {
                    "task": spec.task,
                    "method": method,
                    "max_frame_million": ours_frame,
                    "mean": mean,
                    "sd": sd,
                    "n": n,
                    "selection": (
                        "per-run best-so-far at 30M after linear interpolation "
                        "to ours grid; no extrapolation"
                    ),
                }
            )

        strongest_method = max(
            BASELINE_METHODS,
            key=lambda method: baseline_stats[method][0],
        )
        strongest_mean, strongest_sd, strongest_n = baseline_stats[strongest_method]
        relative_gain = 100.0 * (ours_mean / strongest_mean - 1.0)

        long_rows.append(
            {
                "task": spec.task,
                "method": "ours",
                "max_frame_million": ours_frame,
                "mean": ours_mean,
                "sd": ours_sd,
                "n": len(ours_values),
                "selection": (
                    f"per-seed best-so-far at 30M for post-hoc config "
                    f"{spec.config_id}; "
                    f"development seeds {','.join(map(str, ours_result.seeds))}"
                ),
            }
        )
        summary_rows.append(
            {
                "task": spec.task,
                "selected_config": spec.config_id,
                "ours_seeds": ",".join(map(str, ours_result.seeds)),
                "ours_checkpoint_million": ours_frame,
                "ours_mean": ours_mean,
                "ours_sd": ours_sd,
                "ours_best_seed": ours_result.best_seed,
                "ours_best_seed_value": ours_result.best_seed_value,
                "strongest_baseline_peak_at_30m": strongest_method,
                "strongest_baseline_mean": strongest_mean,
                "strongest_baseline_sd": strongest_sd,
                "strongest_baseline_n": strongest_n,
                "relative_gain_percent": relative_gain,
                "frozen_genpo_reference": spec.genpo_reference,
                "best_seed_gain_vs_genpo_percent": (
                    100.0
                    * (ours_result.best_seed_value / spec.genpo_reference - 1.0)
                ),
            }
        )

    return long_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_normalized_summary(
    output_dir: Path,
    long_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    apply_style("compact")
    fig, ax = plt.subplots(figsize=(10.8, 4.1))

    task_order = [spec.task for spec in TASK_SPECS]
    task_labels = [spec.short_title for spec in TASK_SPECS]
    methods = ("ours", "PPO", "DPPO", "FPO", "GenPO")
    row_lookup = {
        (str(row["task"]), str(row["method"])): row for row in long_rows
    }
    strongest = {
        str(row["task"]): float(row["strongest_baseline_mean"])
        for row in summary_rows
    }

    x = np.arange(len(task_order), dtype=float)
    width = 0.15
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    colors = {
        "PPO": "#D9D9D9",
        "DPPO": "#BDBDBD",
        "FPO": "#969696",
        "GenPO": "#737373",
        "ours": BAR_BLUE,
    }
    hatches = {
        "PPO": "",
        "DPPO": "..",
        "FPO": "xx",
        "GenPO": "--",
        "ours": "///",
    }

    for method, offset in zip(methods, offsets, strict=True):
        means = []
        sds = []
        for task in task_order:
            row = row_lookup[(task, method)]
            denominator = strongest[task]
            means.append(float(row["mean"]) / denominator)
            sds.append(float(row["sd"]) / denominator)
        plot_bar_series(
            ax,
            x,
            means,
            label=OURS_DISPLAY_NAME if method == "ours" else method,
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
    ax.set_xticklabels(task_labels)
    ax.set_ylim(0.0, 1.38)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    style_labels(
        ax,
        profile="compact",
        ylabel="Return",
    )
    configure_axes(ax, profile="compact", grid_axis="y", hide_top=False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=5,
        frameon=False,
        prop={"family": "Times New Roman", "weight": "bold", "size": 10},
        columnspacing=1.1,
        handlelength=1.7,
    )
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.18, top=0.82)
    save_figure(
        fig,
        output_dir / "best5_sample_efficiency.pdf",
        preview_png=output_dir / "best5_sample_efficiency.png",
        transparent=True,
        pad_inches=0.03,
    )
    plt.close(fig)


def write_caption(
    output_dir: Path,
    summary_rows: list[dict[str, object]],
) -> None:
    task_text = ", ".join(spec.panel_title for spec in TASK_SPECS)
    selection_text = "; ".join(
        (
            f"{spec.panel_title}: `{spec.config_id}` "
            f"(seeds {summary_rows[index]['ours_seeds']})"
        )
        for index, spec in enumerate(TASK_SPECS)
    )
    lines = [
        "# Figure captions",
        "",
        (
            "**Best-so-far learning curves.** Maximum evaluation return observed "
            "up to each checkpoint on the five selected MuJoCo Playground "
            f"development tasks ({task_text}). For every seed/run, the plotted "
            "trajectory is the cumulative maximum after linear interpolation onto "
            "the exact `ours` evaluation grid (approximately one evaluation every "
            "3M frames). Only 0–30M is displayed. Baselines are interpolated "
            "run-by-run from their original checkpoints; no run is extrapolated "
            "past its final recorded checkpoint. Curves show the cross-seed/run "
            "mean with shaded ±1 sample standard deviation. `ours` uses two "
            "development seeds for four tasks and five for Finger Turn Hard; "
            "each baseline contributes up to five runs."
        ),
        "",
        (
            "**Selected `ours` runs.** "
            f"{selection_text}. Configurations are selected post hoc by the "
            "cross-seed mean best-so-far return within 30M frames."
        ),
        "",
        (
            "**Peak-return summary.** Each value is the mean of the per-seed/run "
            "best-so-far return at the common 30M endpoint after interpolation, "
            "normalized by the strongest baseline mean for that task. Error bars "
            "show ±1 sample standard deviation using the same normalization. Runs "
            "that do not bracket 30M are excluded rather than extrapolated. Higher "
            "is better."
        ),
        "",
        (
            "**Scope.** This is a post-hoc development-set, 30M-frame "
            "sample-efficiency comparison. It should not be described as a frozen "
            "held-out or 60M-frame final benchmark."
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
        help="Path to the flow-rl-final checkout containing Playground logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace_root / "artifacts" / "mujoco_playground_best5_smfp",
        help="Directory for PDFs, preview PNGs, and summary CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_root = args.final_root.resolve()
    output_dir = args.output_dir.resolve()
    baseline_csv = final_root / "playground_baselines_10tasks_raw.csv"
    if not baseline_csv.is_file():
        raise FileNotFoundError(baseline_csv)

    output_dir.mkdir(parents=True, exist_ok=True)
    ours = {spec.task: load_ours(final_root, spec) for spec in TASK_SPECS}
    target_x = next(iter(ours.values())).curve.x.copy()
    for task, result in ours.items():
        if (
            result.curve.x.shape != target_x.shape
            or not np.allclose(result.curve.x, target_x)
        ):
            raise ValueError(f"ours evaluation grid differs for {task}")
    baselines, peak_values = load_baselines(baseline_csv, target_x)
    long_rows, summary_rows = build_summary_rows(ours, peak_values)

    write_csv(output_dir / "best5_sample_efficiency_long.csv", long_rows)
    write_csv(output_dir / "best5_sample_efficiency_summary.csv", summary_rows)
    plot_learning_curves(output_dir, ours, baselines)
    plot_normalized_summary(output_dir, long_rows, summary_rows)
    write_caption(output_dir, summary_rows)

    print(f"Wrote peak-return figures and tables to {output_dir}")
    for row in summary_rows:
        print(
            f"{row['task']}: ours={float(row['ours_mean']):.3f} "
            f"vs {row['strongest_baseline_peak_at_30m']}="
            f"{float(row['strongest_baseline_mean']):.3f} "
            f"({float(row['relative_gain_percent']):+.2f}%)"
        )


if __name__ == "__main__":
    main()
