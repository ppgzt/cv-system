#!/usr/bin/env python3
"""Auditable reconstruction of post-capture latency and inference drain time.

The complete run is the experimental replicate (n=5). Passage-level values are
retained for audit and visualization, but are never used as independent FPS
replicates for confidence intervals.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


FIXED_FPS = [1, 2, 3, 4, 5, 10, 15, 20, 30]
RUN_NUMBERS = [1, 2, 3, 4, 5]
CONFIG_ORDER = ["Original", *map(str, FIXED_FPS)]
EXPECTED_RUNS = 50
EXPECTED_FIXED_RUNS = 45
EXPECTED_PASSAGES_PER_RUN = 184
EXPECTED_OBSERVATIONS = 9_200
EXPECTED_FIXED_OBSERVATIONS = 8_280
DECOMPOSITION_ATOL_S = 1e-6
T_CRIT_DF4 = float(stats.t.ppf(0.975, 4))

FIXED_RE = re.compile(r"mas-single_(?P<fps>\d+)fps_r(?P<run>\d+)$")
ORIGINAL_RE = re.compile(r"mas-single_native_r(?P<run>\d+)$")

COLORS = {
    "mean": "#0072B2",
    "p95": "#D55E00",
    "residual": "#009E73",
    "points": "#666666",
    "grid": "#D9D9D9",
}


def find_repo_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "thread_pipeline.py").is_file() and (
            candidate / "power_runs" / "battery_mas-single_20260708_104924"
        ).is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the cv-system repository root")


REPO_ROOT = find_repo_root(Path.cwd())
FIXED_ROOT = REPO_ROOT / "power_runs" / "battery_mas-single_20260708_104924"
ORIGINAL_ROOT = REPO_ROOT / "power_runs" / "battery_mas-single_native_20260713_185901"
OUTPUT_DIR = REPO_ROOT / "article_artifacts" / "drain_time_audit"
COMPACT_FIGURE_DIR = REPO_ROOT / "article_artifacts_compact" / "figures"


def unique_metrics(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("*/metrics.json"))
    if len(matches) != 1:
        raise AssertionError(f"Expected one metrics.json under {run_dir}, found {len(matches)}")
    return matches[0]


def discover_runs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(FIXED_ROOT.glob("mas-single_*fps_r*")):
        match = FIXED_RE.fullmatch(run_dir.name)
        if not match:
            continue
        fps, run_number = int(match.group("fps")), int(match.group("run"))
        if fps in FIXED_FPS and run_number in RUN_NUMBERS:
            rows.append(
                {
                    "campaign": "fixed-FPS",
                    "configuration": str(fps),
                    "configured_fps": float(fps),
                    "run": f"r{run_number}",
                    "run_number": run_number,
                    "metrics_path": unique_metrics(run_dir),
                }
            )
    for run_dir in sorted(ORIGINAL_ROOT.glob("mas-single_native_r*")):
        match = ORIGINAL_RE.fullmatch(run_dir.name)
        if not match:
            continue
        run_number = int(match.group("run"))
        if run_number in RUN_NUMBERS:
            rows.append(
                {
                    "campaign": "original-timestamp",
                    "configuration": "Original",
                    "configured_fps": math.nan,
                    "run": f"r{run_number}",
                    "run_number": run_number,
                    "metrics_path": unique_metrics(run_dir),
                }
            )
    result = pd.DataFrame(rows)
    result["configuration"] = pd.Categorical(result["configuration"], CONFIG_ORDER, ordered=True)
    result = result.sort_values(["configuration", "run_number"]).reset_index(drop=True)
    assert len(result) == EXPECTED_RUNS, f"runs={len(result)}"
    assert result["metrics_path"].astype(str).nunique() == EXPECTED_RUNS
    assert not result.duplicated(["configuration", "run"]).any()
    per_configuration = result.groupby("configuration", observed=True)["run"].nunique()
    assert per_configuration.reindex(CONFIG_ORDER).eq(5).all(), per_configuration.to_dict()
    assert result["campaign"].eq("fixed-FPS").sum() == EXPECTED_FIXED_RUNS
    return result


def timestamp(value: Any, context: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise AssertionError(f"Invalid timestamp at {context}: {value!r}") from exc
    if pd.isna(parsed):
        raise AssertionError(f"Missing timestamp at {context}")
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.tz_localize(None)
    return parsed


def extract_passages(run_specs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in run_specs.itertuples(index=False):
        path = Path(spec.metrics_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        animals = payload.get("animals")
        assert isinstance(animals, dict), f"animals is not an object: {path}"
        assert len(animals) == EXPECTED_PASSAGES_PER_RUN, f"{path}: passages={len(animals)}"

        for animal_id, passage in animals.items():
            context = f"{spec.configuration}/{spec.run}/{animal_id}"
            last_capture = timestamp(passage.get("last_image_capture_time"), context + "/last_capture")
            passage_final = timestamp(passage.get("weight_prediction_final"), context + "/passage_final")
            frames_captured = int(passage.get("total_of_images"))
            frames_accepted = int(passage.get("suitable_images"))
            imgs = passage.get("imgs", {})
            assert isinstance(imgs, dict), f"invalid imgs object: {context}"
            assert frames_accepted == len(imgs), (
                f"accepted/inference mismatch at {context}: {frames_accepted}/{len(imgs)}"
            )

            starts: list[pd.Timestamp] = []
            finals: list[pd.Timestamp] = []
            for image_id, image in imgs.items():
                assert isinstance(image, dict), f"invalid image record: {context}/{image_id}"
                start = timestamp(image.get("weight_prediction_start"), context + f"/{image_id}/start")
                final = timestamp(image.get("weight_prediction_final"), context + f"/{image_id}/final")
                assert start <= final, f"prediction start after final: {context}/{image_id}"
                starts.append(start)
                finals.append(final)

            inference_count = len(finals)
            has_valid_prediction = inference_count > 0
            last_prediction = max(finals) if finals else pd.NaT
            residual_active = sum(start <= last_capture < final for start, final in zip(starts, finals))
            residual_not_started = sum(start > last_capture for start in starts)
            residual_inferences = residual_active + residual_not_started
            residual_by_completion = sum(final > last_capture for final in finals)
            assert residual_inferences == residual_by_completion, context

            current_latency = (passage_final - last_capture).total_seconds()
            if has_valid_prediction:
                drain = max(0.0, (last_prediction - last_capture).total_seconds())
                comparison_time = max(last_capture, last_prediction)
            else:
                # No accepted frame means zero inference work to drain, but it is
                # explicitly not evidence of a valid weight estimate.
                drain = 0.0
                comparison_time = last_capture
            finalization_gap = (passage_final - comparison_time).total_seconds()
            decomposition_error = current_latency - (drain + finalization_gap)

            assert frames_captured >= frames_accepted >= 0, context
            assert current_latency >= 0.0, context
            assert drain >= 0.0, context
            assert finalization_gap >= 0.0, context
            assert abs(decomposition_error) <= DECOMPOSITION_ATOL_S, (
                f"decomposition failed at {context}: {decomposition_error}"
            )

            rows.append(
                {
                    "campaign": spec.campaign,
                    "configuration": str(spec.configuration),
                    "configured_fps": spec.configured_fps,
                    "run": spec.run,
                    "run_number": spec.run_number,
                    "passage_id": str(animal_id),
                    "animal_id": str(animal_id),
                    "frames_captured": frames_captured,
                    "frames_accepted": frames_accepted,
                    "inference_count": inference_count,
                    "last_image_capture_time": last_capture.isoformat(),
                    "last_prediction_completion_time": (
                        last_prediction.isoformat() if pd.notna(last_prediction) else ""
                    ),
                    "weight_prediction_final": passage_final.isoformat(),
                    "current_post_capture_latency_s": current_latency,
                    "clean_pipeline_drain_time_s": drain,
                    "post_inference_finalization_gap_s": finalization_gap,
                    "total_after_capture_decomposition_error_s": decomposition_error,
                    "active_inferences_at_last_capture": residual_active,
                    "not_started_inferences_at_last_capture": residual_not_started,
                    "residual_inferences": residual_inferences,
                    "has_residual_workload": residual_inferences > 0,
                    "has_valid_prediction": has_valid_prediction,
                    "passage_status": "valid_prediction" if has_valid_prediction else "no_valid_prediction",
                    "metrics_path": str(path.relative_to(REPO_ROOT)),
                }
            )

    result = pd.DataFrame(rows)
    result["configuration"] = pd.Categorical(result["configuration"], CONFIG_ORDER, ordered=True)
    result = result.sort_values(["configuration", "run_number", "passage_id"]).reset_index(drop=True)
    assert len(result) == EXPECTED_OBSERVATIONS, f"observations={len(result)}"
    assert result["campaign"].eq("fixed-FPS").sum() == EXPECTED_FIXED_OBSERVATIONS
    counts = result.groupby(["configuration", "run"], observed=True).size()
    assert len(counts) == EXPECTED_RUNS and counts.eq(EXPECTED_PASSAGES_PER_RUN).all()
    assert result["passage_id"].nunique() == EXPECTED_PASSAGES_PER_RUN
    assert result.groupby(["configuration", "passage_id"], observed=True)["run"].nunique().eq(5).all()
    assert result["has_valid_prediction"].eq(result["inference_count"].gt(0)).all()
    assert result["has_residual_workload"].eq(result["residual_inferences"].gt(0)).all()
    assert result.loc[~result["has_valid_prediction"], "clean_pipeline_drain_time_s"].eq(0).all()
    max_error = result["total_after_capture_decomposition_error_s"].abs().max()
    assert max_error <= DECOMPOSITION_ATOL_S, max_error
    return result


def p95(values: Iterable[float]) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.quantile(0.95)) if len(series) else math.nan


def validate_existing_residual_analysis(passages: pd.DataFrame) -> None:
    """Cross-check the fixed-FPS reconstruction against the prior audit."""
    path = FIXED_ROOT / "residual_analysis" / "residual_by_passage.csv"
    assert path.is_file(), f"Missing existing residual audit: {path}"
    old = pd.read_csv(path)
    new = passages[passages["campaign"].eq("fixed-FPS")].copy()
    new["fps"] = new["configured_fps"].astype(int)
    new["animal_id"] = new["animal_id"].astype(str).str.zfill(4)
    old["animal_id"] = old["animal_id"].astype(str).str.zfill(4)
    merged = new.merge(old, on=["fps", "run", "animal_id"], suffixes=("_new", "_old"), validate="one_to_one")
    assert len(merged) == EXPECTED_FIXED_OBSERVATIONS
    for reconstructed, existing in [
        ("clean_pipeline_drain_time_s", "prediction_drain_s"),
        ("current_post_capture_latency_s", "post_capture_latency_s"),
        ("post_inference_finalization_gap_s", "finalization_overhead_s"),
    ]:
        np.testing.assert_allclose(
            merged[reconstructed].to_numpy(), merged[existing].to_numpy(), rtol=0, atol=1e-12
        )
    np.testing.assert_array_equal(
        merged["residual_inferences_new"].to_numpy(), merged["residual_inferences_old"].to_numpy()
    )
    np.testing.assert_array_equal(
        merged["has_residual_workload"].astype(bool).to_numpy(), merged["has_residual"].astype(bool).to_numpy()
    )


def summarize_by_run(passages: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (configuration, run), group in passages.groupby(
        ["configuration", "run"], observed=True, sort=False
    ):
        rows.append(
            {
                "configuration": str(configuration),
                "configured_fps": group["configured_fps"].iloc[0],
                "run": run,
                "passages": len(group),
                "frames_per_passage_mean": float(group["frames_captured"].mean()),
                "accepted_per_passage_mean": float(group["frames_accepted"].mean()),
                "residual_passages_pct": 100.0 * float(group["has_residual_workload"].mean()),
                "residual_inferences_mean": float(group["residual_inferences"].mean()),
                "residual_inferences_p95": p95(group["residual_inferences"]),
                "drain_mean_s": float(group["clean_pipeline_drain_time_s"].mean()),
                "drain_p95_s": p95(group["clean_pipeline_drain_time_s"]),
                "drain_peak_s": float(group["clean_pipeline_drain_time_s"].max()),
                "drain_positive_passages_pct": 100.0 * float(group["clean_pipeline_drain_time_s"].gt(0).mean()),
                "no_valid_prediction_pct": 100.0 * float((~group["has_valid_prediction"]).mean()),
                "finalization_gap_mean_s": float(group["post_inference_finalization_gap_s"].mean()),
                "finalization_gap_p95_s": p95(group["post_inference_finalization_gap_s"]),
                "current_post_capture_latency_mean_s": float(group["current_post_capture_latency_s"].mean()),
                "current_post_capture_latency_p95_s": p95(group["current_post_capture_latency_s"]),
                "current_post_capture_latency_peak_s": float(group["current_post_capture_latency_s"].max()),
                "decomposition_max_abs_error_s": float(
                    group["total_after_capture_decomposition_error_s"].abs().max()
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["configuration"] = pd.Categorical(result["configuration"], CONFIG_ORDER, ordered=True)
    result = result.sort_values(["configuration", "run"]).reset_index(drop=True)
    assert len(result) == EXPECTED_RUNS
    assert result["passages"].eq(EXPECTED_PASSAGES_PER_RUN).all()
    return result


RUN_METRICS = [
    "frames_per_passage_mean",
    "accepted_per_passage_mean",
    "residual_passages_pct",
    "residual_inferences_mean",
    "residual_inferences_p95",
    "drain_mean_s",
    "drain_p95_s",
    "drain_peak_s",
    "drain_positive_passages_pct",
    "no_valid_prediction_pct",
    "finalization_gap_mean_s",
    "finalization_gap_p95_s",
    "current_post_capture_latency_mean_s",
    "current_post_capture_latency_p95_s",
    "current_post_capture_latency_peak_s",
]


def run_stat(values: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    assert len(x) == 5
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    half = T_CRIT_DF4 * sd / math.sqrt(5)
    return {"mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half, "ci95_half": half}


def summarize_by_configuration(run_df: pd.DataFrame, passages: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for configuration in CONFIG_ORDER:
        group = run_df[run_df["configuration"].astype(str).eq(configuration)]
        raw = passages[passages["configuration"].astype(str).eq(configuration)]
        assert len(group) == 5 and len(raw) == 920
        row: dict[str, Any] = {
            "configuration": configuration,
            "configured_fps": raw["configured_fps"].iloc[0],
            "n_runs": 5,
            "passage_run_observations": len(raw),
            "drain_absolute_peak_s": float(raw["clean_pipeline_drain_time_s"].max()),
            "current_post_capture_latency_absolute_peak_s": float(raw["current_post_capture_latency_s"].max()),
        }
        for metric in RUN_METRICS:
            summary = run_stat(group[metric])
            for suffix, value in summary.items():
                row[f"{metric}_{suffix}"] = value
        rows.append(row)
    result = pd.DataFrame(rows)
    assert result["configuration"].tolist() == CONFIG_ORDER
    return result


def tick_hypothesis(passages: pd.DataFrame, run_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed = passages[passages["campaign"].eq("fixed-FPS")].copy()
    rows: list[dict[str, Any]] = []
    for fps in FIXED_FPS:
        group = fixed[fixed["configured_fps"].eq(float(fps))]
        no_residual = group[~group["has_residual_workload"]].copy()
        period = 1.0 / fps
        delta = no_residual["current_post_capture_latency_s"] - period
        abs_delta = delta.abs()
        rows.append(
            {
                "configured_fps": fps,
                "tick_period_s": period,
                "passage_run_observations": len(group),
                "no_residual_observations": len(no_residual),
                "current_latency_mean_s_direct": float(group["current_post_capture_latency_s"].mean()),
                "current_latency_p95_s_direct": p95(group["current_post_capture_latency_s"]),
                "drain_mean_s_direct": float(group["clean_pipeline_drain_time_s"].mean()),
                "drain_p95_s_direct": p95(group["clean_pipeline_drain_time_s"]),
                "finalization_gap_mean_s_direct": float(group["post_inference_finalization_gap_s"].mean()),
                "finalization_gap_p95_s_direct": p95(group["post_inference_finalization_gap_s"]),
                "no_residual_latency_over_tick_mean": float(
                    (no_residual["current_post_capture_latency_s"] / period).mean()
                ),
                "no_residual_latency_minus_tick_mean_s": float(delta.mean()),
                "no_residual_latency_minus_tick_p95_s": p95(delta),
                "no_residual_within_1ms_pct": 100.0 * float(abs_delta.le(0.001).mean()),
                "no_residual_within_5ms_pct": 100.0 * float(abs_delta.le(0.005).mean()),
                "no_residual_within_10ms_pct": 100.0 * float(abs_delta.le(0.010).mean()),
                "no_residual_within_5pct_pct": 100.0 * float(abs_delta.le(0.05 * period).mean()),
            }
        )
    tick_df = pd.DataFrame(rows)

    run_fixed = run_df[run_df["configuration"].astype(str).ne("Original")].copy()
    run_fixed["inverse_fps_s"] = 1.0 / pd.to_numeric(run_fixed["configured_fps"])
    config_means = run_fixed.groupby("configured_fps", as_index=False).agg(
        inverse_fps_s=("inverse_fps_s", "first"),
        finalization_gap_mean_s=("finalization_gap_mean_s", "mean"),
    )
    correlations = []
    for level, x, y in [
        ("passage-run (n=8280; descriptive only)", 1.0 / fixed["configured_fps"], fixed["post_inference_finalization_gap_s"]),
        ("run (n=45)", run_fixed["inverse_fps_s"], run_fixed["finalization_gap_mean_s"]),
        ("configuration mean (n=9)", config_means["inverse_fps_s"], config_means["finalization_gap_mean_s"]),
    ]:
        pearson = stats.pearsonr(x, y)
        spearman = stats.spearmanr(x, y)
        correlations.append(
            {
                "aggregation_level": level,
                "n": len(x),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
            }
        )
    return tick_df, pd.DataFrame(correlations)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def style_axis(ax: mpl.axes.Axes) -> None:
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", format="pdf", facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", format="png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def metric_arrays(config_df: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    fixed = config_df[config_df["configuration"].ne("Original")].set_index("configuration")
    mean = np.array([fixed.loc[str(fps), f"{metric}_mean"] for fps in FIXED_FPS], dtype=float)
    half = np.array([fixed.loc[str(fps), f"{metric}_ci95_half"] for fps in FIXED_FPS], dtype=float)
    return mean, half


def make_figures(passages: pd.DataFrame, config_df: pd.DataFrame) -> None:
    configure_matplotlib()
    # Categorical spacing prevents the dense 1-5 FPS labels from colliding in
    # a one-column IEEE figure while preserving the configured values as labels.
    x = np.arange(len(FIXED_FPS), dtype=float)

    drain_mean, drain_mean_ci = metric_arrays(config_df, "drain_mean_s")
    drain_p95, drain_p95_ci = metric_arrays(config_df, "drain_p95_s")
    fig, ax = plt.subplots(figsize=(3.50, 2.35))
    ax.errorbar(x, drain_mean, yerr=drain_mean_ci, color=COLORS["mean"], marker="o", capsize=2.5, label="Mean")
    ax.errorbar(x, drain_p95, yerr=drain_p95_ci, color=COLORS["p95"], marker="s", linestyle="--", capsize=2.5, label="P95")
    ax.set(xlabel="Configured FPS", ylabel="Inference drain time (s)")
    ax.set_xticks(x, [str(fps) for fps in FIXED_FPS])
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    style_axis(ax)
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.24, top=0.98)
    save_figure(fig, "fig_drain_time_mean_p95")

    # Survival ECDF is more informative than boxplots when many passage means
    # are exactly zero. The full panel retains every outlier; the zoomed panel
    # makes 10, 15 and 20 FPS readable without a logarithmic transformation.
    per_passage = (
        passages[passages["campaign"].eq("fixed-FPS")]
        .groupby(["configured_fps", "passage_id"], as_index=False)["clean_pipeline_drain_time_s"]
        .mean()
    )
    assert per_passage.groupby("configured_fps")["passage_id"].nunique().eq(184).all()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))
    cmap = plt.get_cmap("viridis")
    for idx, fps in enumerate(FIXED_FPS):
        values = np.sort(per_passage.loc[per_passage["configured_fps"].eq(fps), "clean_pipeline_drain_time_s"].to_numpy(float))
        survival = 1.0 - np.arange(1, len(values) + 1) / len(values)
        for ax in axes:
            ax.step(values, survival, where="post", color=cmap(idx / 8), linewidth=1.15, label=str(fps))
    full_max = float(per_passage["clean_pipeline_drain_time_s"].max())
    non30 = per_passage[per_passage["configured_fps"].ne(30)]["clean_pipeline_drain_time_s"]
    zoom_max = max(0.5, float(non30.quantile(0.995)) * 1.08)
    axes[0].set_xlim(0, full_max * 1.02)
    axes[0].set_title("(a) Full absolute range")
    axes[1].set_xlim(0, zoom_max)
    axes[1].set_title("(b) Near-origin detail")
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean drain time per passage across 5 runs (s)")
        style_axis(ax)
    axes[0].set_ylabel("Fraction of passages above x")
    axes[1].legend(title="FPS", frameon=False, ncol=3, loc="upper right")
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.88, wspace=0.22)
    save_figure(fig, "fig_drain_time_distribution")

    residual, residual_ci = metric_arrays(config_df, "residual_passages_pct")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45))
    axes[0].errorbar(x, residual, yerr=residual_ci, color=COLORS["residual"], marker="o", capsize=2.5)
    axes[0].set(xlabel="Configured FPS", ylabel="Passages with residual workload (%)")
    axes[0].set_xticks(x, [str(fps) for fps in FIXED_FPS])
    axes[0].set_ylim(0, 100)
    axes[0].set_title("(a) Residual workload")
    axes[1].errorbar(x, drain_mean, yerr=drain_mean_ci, color=COLORS["mean"], marker="o", capsize=2.5, label="Mean")
    axes[1].errorbar(x, drain_p95, yerr=drain_p95_ci, color=COLORS["p95"], marker="s", linestyle="--", capsize=2.5, label="P95")
    axes[1].set(xlabel="Configured FPS", ylabel="Inference drain time (s)")
    axes[1].set_xticks(x, [str(fps) for fps in FIXED_FPS])
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("(b) Drain time")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=45)
        style_axis(ax)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.23, top=0.87, wspace=0.25)
    save_figure(fig, "fig_residual_and_drain")

    # One-column replacement for the previous post-capture-latency boxplot.
    # The unit of visualization is the passage after averaging its five runs.
    compact_data = [
        per_passage.loc[
            per_passage["configured_fps"].eq(float(fps)),
            "clean_pipeline_drain_time_s",
        ].to_numpy(float)
        for fps in FIXED_FPS
    ]
    assert all(len(values) == EXPECTED_PASSAGES_PER_RUN for values in compact_data)
    compact_positions = np.arange(1, len(FIXED_FPS) + 1, dtype=float)
    compact_means = np.asarray([values.mean() for values in compact_data])
    compact_max = max(float(values.max()) for values in compact_data)
    assert compact_max < 100.0, compact_max

    fig, ax = plt.subplots(figsize=(3.50, 1.85))
    artists = ax.boxplot(
        compact_data,
        positions=compact_positions,
        widths=0.58,
        whis=1.5,
        patch_artist=True,
        showfliers=True,
        boxprops={"facecolor": "#DCEAF4", "edgecolor": COLORS["mean"], "linewidth": 0.95},
        whiskerprops={"color": COLORS["mean"], "linewidth": 0.85},
        capprops={"color": COLORS["mean"], "linewidth": 0.85},
        medianprops={"color": "#222222", "linewidth": 1.05},
        flierprops={
            "marker": "o",
            "markersize": 2.0,
            "markerfacecolor": "none",
            "markeredgecolor": COLORS["points"],
            "markeredgewidth": 0.50,
            "linestyle": "none",
        },
    )
    assert len(artists["boxes"]) == len(FIXED_FPS)
    ax.plot(
        compact_positions,
        compact_means,
        linestyle="none",
        marker="o",
        markersize=4.2,
        color=COLORS["mean"],
        zorder=4,
    )
    ax.set_xticks(compact_positions, [str(fps) for fps in FIXED_FPS])
    ax.set_xlim(0.55, 9.35)
    # The largest plotted passage mean is ~97.98 s. Use 110 s to provide
    # visible headroom; the 105.41 s passage-run absolute peak is intentionally
    # reported in the table rather than mixed into this passage-mean boxplot.
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Configured FPS", labelpad=2.0)
    ax.set_ylabel("Inference drain time (s)", labelpad=2.0)
    ax.tick_params(axis="x", labelsize=7.5, pad=1.5)
    ax.tick_params(axis="y", labelsize=7.5, pad=1.5)
    style_axis(ax)
    fig.subplots_adjust(left=0.185, right=0.985, bottom=0.245, top=0.995)
    COMPACT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    compact_stem = "fig_drain_time_box_absolute_single_compact"
    fig.savefig(
        COMPACT_FIGURE_DIR / f"{compact_stem}.pdf",
        format="pdf",
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.01,
    )
    fig.savefig(
        COMPACT_FIGURE_DIR / f"{compact_stem}.png",
        format="png",
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close(fig)


def fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def latex_tables(config_df: pd.DataFrame) -> None:
    rows_full = []
    rows_short = []
    for row in config_df.itertuples(index=False):
        values = [
            str(row.configuration),
            fmt(row.frames_per_passage_mean_mean, 2),
            fmt(row.accepted_per_passage_mean_mean, 2),
            fmt(row.residual_passages_pct_mean, 1),
            fmt(row.residual_inferences_mean_mean, 2),
            fmt(row.drain_mean_s_mean, 3),
            fmt(row.drain_p95_s_mean, 3),
        ]
        rows_full.append(" & ".join(values) + r" \\")
        rows_short.append(" & ".join([*values[:4], *values[5:]]) + r" \\")

    full = "\n".join(
        [
            r"\begin{table}[t]",
            r"\caption{Residual workload and inference drain time.}",
            r"\label{tab:drain-time}",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.0pt}",
            r"\renewcommand{\arraystretch}{1.05}",
            r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}",
            r"\toprule",
            r"Input & \shortstack{Frames/\\pass.} & \shortstack{Accepted/\\pass.} & \shortstack{Residual\\pass. (\%)} & \shortstack{Residual\\inf. mean} & \shortstack{Drain\\mean (s)} & \shortstack{Drain\\P95 (s)} \\",
            r"\midrule",
            *rows_full,
            r"\bottomrule",
            r"\end{tabular*}",
            r"\vspace{1pt}",
            r"\parbox{\columnwidth}{\scriptsize Original is the native-timestamp trace. Statistics are computed within each run and then averaged over five runs. Residual pass. denotes passages with at least one inference unfinished at the last capture.}",
            r"\end{table}",
            "",
        ]
    )
    short = "\n".join(
        [
            r"\begin{table}[t]",
            r"\caption{Residual workload and inference drain time (compact variant).}",
            r"\label{tab:drain-time-compact}",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{2.2pt}",
            r"\renewcommand{\arraystretch}{1.05}",
            r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrr@{}}",
            r"\toprule",
            r"Input & \shortstack{Frames/\\pass.} & \shortstack{Accepted/\\pass.} & \shortstack{Residual\\pass. (\%)} & \shortstack{Drain\\mean (s)} & \shortstack{Drain\\P95 (s)} \\",
            r"\midrule",
            *rows_short,
            r"\bottomrule",
            r"\end{tabular*}",
            r"\vspace{1pt}",
            r"\parbox{\columnwidth}{\scriptsize Original is the native-timestamp trace. Statistics are computed within each run and then averaged over five runs.}",
            r"\end{table}",
            "",
        ]
    )
    assert "resizebox" not in full + short
    assert len(rows_full) == len(rows_short) == 10
    (OUTPUT_DIR / "table_drain_time_candidate.tex").write_text(full, encoding="utf-8")
    (OUTPUT_DIR / "table_drain_time_candidate_compact.tex").write_text(short, encoding="utf-8")


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    data = frame[columns].copy()
    for column in data.select_dtypes(include=["number"]).columns:
        data[column] = data[column].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in data.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    passages: pd.DataFrame,
    run_df: pd.DataFrame,
    config_df: pd.DataFrame,
    tick_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
) -> None:
    no_valid = int((~passages["has_valid_prediction"]).sum())
    fixed = passages[passages["campaign"].eq("fixed-FPS")]
    no_residual = fixed[~fixed["has_residual_workload"]]
    max_error = passages["total_after_capture_decomposition_error_s"].abs().max()
    config_view = config_df.rename(
        columns={
            "drain_mean_s_mean": "drain mean (s)",
            "drain_p95_s_mean": "drain P95 (s)",
            "drain_absolute_peak_s": "drain abs. peak (s)",
            "drain_positive_passages_pct_mean": "drain > 0 (%)",
            "no_valid_prediction_pct_mean": "no valid pred. (%)",
            "finalization_gap_mean_s_mean": "gap mean (s)",
            "finalization_gap_p95_s_mean": "gap P95 (s)",
            "current_post_capture_latency_mean_s_mean": "current mean (s)",
            "current_post_capture_latency_p95_s_mean": "current P95 (s)",
        }
    )
    config_columns = [
        "configuration", "drain mean (s)", "drain P95 (s)", "drain abs. peak (s)",
        "drain > 0 (%)", "no valid pred. (%)", "gap mean (s)", "gap P95 (s)",
        "current mean (s)", "current P95 (s)",
    ]
    tick_columns = [
        "configured_fps", "tick_period_s", "no_residual_observations",
        "no_residual_latency_over_tick_mean", "no_residual_latency_minus_tick_mean_s",
        "no_residual_within_1ms_pct", "no_residual_within_5ms_pct",
        "no_residual_within_10ms_pct", "no_residual_within_5pct_pct",
    ]
    corr = correlation_df.iloc[-1]
    report = f"""# Auditoria de post-capture latency e pipeline drain time

Gerado em {datetime.now().isoformat(timespec='seconds')}. Nenhuma implementação do pipeline, notebook existente ou resultado anterior foi alterado.

## Conclusão executiva

O efeito próximo de `1/FPS` foi localizado no código e confirmado nos dados. No capturador fixed-FPS, `last_capture` é atualizado no instante em que a imagem é carregada e imediatamente antes de ela ser colocada em Q1 (`thread_pipeline.py`, linhas 127-147). Mesmo após a última iteração válida, o laço incrementa `next_tick` e dorme até o deadline seguinte (linhas 154-158); somente depois desse sono coloca `END_ANIMAL` em Q1 (linhas 160-163). O sentinel atravessa Q1 -> Q2 -> Q3 em ordem FIFO (linhas 281-346) e só então o preditor calcula a média, registra a predição final e grava `weight_prediction_final` (linhas 368-381, 398-439).

Logo, `current_post_capture_latency_s` é uma latência formal de emissão pós-captura cujo intervalo contém: (i) inferências ainda pendentes e demais etapas FIFO; (ii) a espera do tick de encerramento fixed-FPS; e (iii) a curta finalização após a drenagem. Esses trabalhos podem se sobrepor no tempo: não se deve somar um “custo do tick” independente ao drain. Ela não é uma medida pura de backlog. `clean_pipeline_drain_time_s` responde diretamente quanto tempo levou para terminar a última inferência que ainda não havia terminado na última captura.

Recomendação editorial, considerando que o artigo precisa reduzir espaço e que sua pergunta central é overload: **B - manter apenas drain time no artigo**. Reporte mean, P95 e absolute peak: mean caracteriza o comportamento típico médio, P95 caracteriza a cauda recorrente e peak preserva o pior caso observado. A post-capture latency antiga deve permanecer somente nesta trilha de auditoria, pois acrescenta uma segunda pergunta - tempo até emissão formal - que não é necessária para sustentar os resultados de backlog e ainda exige explicar o artefato de `1/FPS`. Drain time e residual workload preservam a evidência relevante. Uma reinstrumentação futura seria necessária apenas se um trabalho posterior quiser estudar responsividade formal sem dependência do tick, registrando `sentinel_emitted`, `sentinel_received_at_predictor`, `aggregation_start/end` e um timestamp monotônico.

## 1. Semântica dos eventos

- **Última captura:** `last_capture = self._now()` ocorre após `dataset.load_depth` retornar uma imagem e imediatamente antes do `q1.put(payload)`. É tempo de parede (`datetime.now().isoformat()`), não o timestamp do dataset nem um relógio monotônico. O valor viaja dentro do sentinel e só é persistido como `metrics["animals"][animal_id]["last_image_capture_time"]` quando o preditor consome `END_ANIMAL` (`thread_pipeline.py`, linhas 402-411); portanto o valor representa a captura, não o instante tardio da escrita no JSON.
- **Sentinel fixed-FPS:** é emitido após a iteração final e após o `sleep` até o tick seguinte. Esse é o mecanismo que introduz aproximadamente `1/FPS` quando o downstream já está ocioso.
- **Sentinel Original:** no replay nativo, cada frame dorme até seu próprio deadline antes da captura, mas não existe sono periódico posterior à última captura; `END_ANIMAL` é enfileirado imediatamente após o `for` (`thread_pipeline.py`, linhas 206-251).
- **Percurso FIFO:** Frame Selection encaminha o tuple inalterado de Q1 para Q2; Depth-Image Preprocessing o encaminha de Q2 para Q3; o worker Weight Prediction o consome depois de todos os frames anteriores da passagem.
- **Início/fim de inferência:** `start_ts` é gravado imediatamente antes de `inference_adapter.predict([img])`; `final_ts`, imediatamente depois do retorno síncrono (`thread_pipeline.py`, linhas 452-473). O módulo de regressão executa `interpreter.invoke()` dentro dessa chamada (`domain/modules/predict_weight.py`, linhas 38-51).
- **`weight_prediction_final`:** o worker já consumiu o sentinel; calcula `np.mean(weights)` (ou `0.0` sem pesos), chama `ReportCollector.record_final_prediction`, então grava o timestamp e só depois emite o log `[FINAL]`. Não é o fim da última inferência, próximo tick após o sentinel, acknowledgment ou envio de mensagem externa.
- **Sem accepted frame:** `weights=[]`, a agregação produz `predicted=0.0`, o relatório registra esse valor e `weight_prediction_final` ainda é emitido. Nesta auditoria isso é `no_valid_prediction`; drain é zero porque não há inferência a drenar, sem interpretar zero como estimativa válida.
- **Timers/polling:** o caminho `thread` usa `time.sleep` deadline-based no capturador e `queue.get()` bloqueante nos workers. Não há polling, callback ou acknowledgment no caminho medido. O `reactor.callLater(1.0, reactor.stop)` existe apenas no engine PADE e ocorre depois do timestamp final; as campanhas auditadas têm PID `mas-single_thread_*`.

Sequência fixed-FPS observada:

```text
last capture timestamp + Q1.put(frame)
  -> next_tick += 1/FPS
  -> capture thread sleeps until that next tick
  -> Q1.put(END_ANIMAL)
  -> selector finishes earlier Q1 items, forwards accepted frames, then sentinel to Q2
  -> enhancer finishes earlier Q2 items, forwards enhanced frames, then sentinel to Q3
  -> predictor finishes all earlier Q3 inference calls
  -> last per-frame prediction completion timestamp
  -> predictor receives END_ANIMAL
  -> mean aggregation (or 0.0 with no accepted frame)
  -> ReportCollector.record_final_prediction
  -> weight_prediction_final timestamp
  -> [FINAL] log
```

## 2. Definições formais reconstruídas

Para passagem `p`, seja `C_p` a última captura, `F_p` o timestamp final da passagem e `I_p=max_i I_{{pi}}` o maior fim de inferência, quando existe:

- `current_post_capture_latency_s = F_p - C_p`;
- `last_prediction_completion_time = I_p` (ausente quando não há inferência);
- `clean_pipeline_drain_time_s = max(0, I_p - C_p)`, ou `0` sem inferência;
- `post_inference_finalization_gap_s = F_p - max(C_p, I_p)`, usando `C_p` quando `I_p` é ausente;
- `decomposition_error = current - (drain + finalization_gap)`.

Definição operacional em linguagem direta: **drain time é o intervalo entre a última captura da passagem e a conclusão da última inferência daquela passagem que terminou depois dessa captura**. Se todas as inferências já haviam terminado em `C_p`, o valor é zero. Se nenhuma inferência foi executada, o valor também é zero, mas a passagem recebe separadamente o status `no_valid_prediction`. O endpoint é `max(imgs[*].weight_prediction_final)`, não `passage.weight_prediction_final`; portanto a métrica exclui agregação final, trânsito final do sentinel, logging e emissão formal do resultado.

Nome recomendado no artigo: **post-capture inference drain time**, abreviado como **drain time** depois da primeira definição. `Post-capture latency` sem qualificador não é recomendado porque já designava `passage.weight_prediction_final - last_image_capture_time` e sugere latência até o resultado final. Alternativas tecnicamente corretas são `post-capture inference-completion time` e `residual inference drain time`. Se o nome `post-capture latency` for indispensável, ele só pode ser reutilizado mediante redefinição explícita do endpoint como última conclusão de inferência e declaração de que não é latência de emissão do resultado; isso cria risco de ambiguidade com os resultados anteriores.

A definição de drain coincide com `prediction_drain_s` do script residual existente. Nas 8.280 observações fixed-FPS, drain, post-capture latency e finalization overhead coincidiram com `residual_analysis/residual_by_passage.csv` até `1e-12 s`; residual count e flag residual coincidiram exatamente. A identidade foi validada em {len(passages):,} observações com tolerância absoluta `{DECOMPOSITION_ATOL_S:g} s`; erro absoluto máximo observado: `{max_error:.3g} s`.

## 3. Integridade e cobertura

- Runs: {len(run_df)} total, {EXPECTED_FIXED_RUNS} fixed-FPS e 5 Original.
- Passagens por run: {EXPECTED_PASSAGES_PER_RUN} em todas as runs.
- Observações passagem-run: {len(passages):,}; fixed-FPS: {len(fixed):,}.
- IDs distintos por configuração: {passages['passage_id'].nunique()}, cada um presente nas cinco runs.
- Passagens sem inferência válida: {no_valid:,}; todas têm drain zero e status separado `no_valid_prediction`.
- `frames_accepted == inference_count`, ordem início <= fim, residual <= inferências e todas as decomposições foram verificadas por assertion.

## 4. Teste da hipótese do tick

{markdown_table(tick_df, tick_columns, 5)}

Nas passagens fixed-FPS sem residual (n={len(no_residual):,}), qualquer atraso após a última captura não pode ser atribuído a inferência pendente. A relação forte entre o gap e `1/FPS` também aparece após agregação por configuração: Pearson r={corr.pearson_r:.5f} (p={corr.pearson_p:.3g}) e Spearman rho={corr.spearman_rho:.5f} (p={corr.spearman_p:.3g}), n=9. A causa não é inferida só da correlação: ela está explicitamente localizada no `sleep` anterior ao `END_ANIMAL`. O restante do gap inclui trânsito FIFO do sentinel, média, mutex curto do `ReportCollector`, logging/agendamento de threads e resolução do relógio.

## 5. Comparação científica das métricas

1. **Tempo até emitir formalmente o resultado após a última captura:** `current_post_capture_latency_s` responde melhor, pois termina em `weight_prediction_final`; é dependente do protocolo e, no fixed-FPS, do tick.
2. **Tempo para processar inferência pendente:** `clean_pipeline_drain_time_s` responde diretamente. Com drain zero e predição válida, todas as inferências terminaram até a captura; sem predição válida, zero significa ausência de trabalho, não sucesso.
3. **Quando o pipeline deixa de acompanhar o workload admitido:** use conjuntamente `has_residual_workload`, `residual_inferences` e drain. Drain sozinho mede duração, não tamanho da fila nem causa causal de overload.
4. **Backlog versus protocolo:** drain é o componente de inferência; `post_inference_finalization_gap_s` é o restante não sobreposto depois de `max(last_capture, last_prediction_completion)`. Ele agrega a parte ainda visível da espera do tick/sentinel e a finalização. Não é o custo contrafactual completo do protocolo, pois parte da espera do sentinel pode ocorrer simultaneamente à drenagem. Como não há timestamp do sentinel, não se pode subdividir numericamente esse gap entre tick, filas de seleção/preprocessamento e agregação, embora o código e o padrão sem residual identifiquem o tick como componente dominante em baixo FPS.

Drain time pode substituir post-capture latency **como métrica do artigo**, desde que o texto deixe de responder à pergunta de latência formal de emissão. Afirmações como “o pipeline emite o resultado X s após a última captura” devem ser removidas, não reescritas com drain. Afirmações sobre backlog, trabalho residual e incapacidade de acompanhar a carga devem usar residual workload + drain. Também deve ser retirada qualquer interpretação de latência próxima a `1/FPS` como custo computacional residual.

## 6. Agregações por configuração (run = réplica, n=5)

{markdown_table(config_view, config_columns, 4)}

Cada média e P95 acima foi calculado dentro de cada run e depois promediado entre as cinco runs. Os CSVs incluem DP e IC95% Student-t (`df=4`) para cada estatística de run. `drain abs. peak` é o maior valor de passagem-run entre as cinco runs; `drain_peak_s_mean` e seu IC, também exportados, representam a média dos cinco picos de run.

Mean, P95 e peak respondem a perguntas diferentes e devem ser preservados quando houver espaço: mean resume o custo médio por passagem; P95 descreve uma cauda alta recorrente sem depender de um único caso; absolute peak registra o pior caso efetivamente observado. O absolute peak é descritivo, sensível a uma única observação e não recebe IC como se fosse uma estimativa estável. Para inferência entre runs, use adicionalmente a média dos cinco picos de run e seu IC95%. Os peaks absolutos foram 2,402 s no Original; 0,193, 0,191, 0,191, 0,193 e 0,162 s em 1-5 FPS; 2,619 s em 10 FPS; 9,430 s em 15 FPS; 23,050 s em 20 FPS; e 105,412 s em 30 FPS.

## 7. Visualização da distribuição

Boxplots não são inválidos, mas têm uma limitação nas configurações com muitos zeros: quartis, mediana e whiskers podem colapsar em zero e não mostram por si só se zero significa pipeline drenado ou ausência de predição. A ECDF de sobrevivência em `fig_drain_time_distribution` é a alternativa estatisticamente mais informativa. Sob a restrição editorial de uma coluna, porém, `fig_drain_time_box_absolute_single_compact` é o compromisso recomendado: mantém escala absoluta linear, mostra todos os outliers e adiciona o ponto da média. Sua interpretação deve ser acompanhada pelo percentual de residual workload e pela taxa `no_valid_prediction`.

Na figura compacta, cada uma das 184 observações de um FPS é primeiro agregada pela média das cinco runs. Consequentemente, o maior outlier desenhado é o maior **mean por passagem**, não o absolute peak de uma observação passagem-run. Não sobreponha o absolute peak nessa figura porque isso misturaria unidades de agregação; reporte-o na tabela ou no texto.

## 8. Implicações para o artigo

- Renomear claramente a métrica existente como latência de emissão formal pós-captura.
- Apresentar drain mean/P95 e residual passages/inferences como evidência de overload.
- Explicar que o fixed-FPS agenda o sentinel no tick posterior; o Original não tem esse sono posterior à última captura.
- Não comparar a latência atual entre FPS como se fosse apenas custo computacional.
- Não afirmar que drain zero implica predição válida; reportar `no_valid_prediction` separadamente.
- Não tratar 184 passagens como réplicas independentes do FPS; os ICs usam cinco runs completas.
- Não alegar tempo de drenagem total de seleção+preprocessamento: o timestamp disponível termina apenas a última inferência, embora o início dessa inferência já pressuponha que o frame chegou ao preditor.

## 9. Artefatos

- `passage_run_latency_decomposition.csv`: trilha completa das 9.200 observações.
- `drain_time_summary_by_run.csv`: 50 réplicas consolidadas.
- `drain_time_summary_by_configuration.csv`: médias, DP e IC95% entre cinco runs.
- `tick_hypothesis_by_fps.csv` e `tick_gap_correlations.csv`: testes específicos do tick.
- Três figuras em PDF vetorial e PNG 300 dpi.
- Figura compacta de uma coluna em `article_artifacts_compact/figures/fig_drain_time_box_absolute_single_compact.pdf` e `.png`.
- Duas variantes de tabela LaTeX de uma coluna, sem `resizebox`.
- Este script reproduz integralmente os artefatos.

## Recomendação final

**B. Manter apenas drain time no artigo.** Com a restrição explícita de espaço e o foco em overload, a latência formal de emissão não acrescenta evidência necessária e traz um componente de tick dependente da implementação. Preserve post-capture latency apenas nos artefatos desta auditoria para rastreabilidade. No manuscrito, combine drain mean/P95 com residual passages e residual inferences; remova afirmações sobre o tempo até emissão formal do resultado. Essa decisão não enfraquece a discussão de overload, mas estreita corretamente o escopo daquilo que o artigo afirma medir.
"""
    (OUTPUT_DIR / "latency_decomposition_audit_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_specs = discover_runs()
    passages = extract_passages(run_specs)
    validate_existing_residual_analysis(passages)
    run_df = summarize_by_run(passages)
    config_df = summarize_by_configuration(run_df, passages)
    tick_df, correlation_df = tick_hypothesis(passages, run_df)

    passages.to_csv(OUTPUT_DIR / "passage_run_latency_decomposition.csv", index=False, float_format="%.9f")
    run_df.to_csv(OUTPUT_DIR / "drain_time_summary_by_run.csv", index=False, float_format="%.9f")
    config_df.to_csv(OUTPUT_DIR / "drain_time_summary_by_configuration.csv", index=False, float_format="%.9f")
    tick_df.to_csv(OUTPUT_DIR / "tick_hypothesis_by_fps.csv", index=False, float_format="%.9f")
    correlation_df.to_csv(OUTPUT_DIR / "tick_gap_correlations.csv", index=False, float_format="%.9f")

    make_figures(passages, config_df)
    latex_tables(config_df)
    write_report(passages, run_df, config_df, tick_df, correlation_df)

    print(f"Wrote audit artifacts to {OUTPUT_DIR}")
    print(f"runs={len(run_df)}, observations={len(passages)}, fixed={passages['campaign'].eq('fixed-FPS').sum()}")
    print(f"maximum decomposition error={passages['total_after_capture_decomposition_error_s'].abs().max():.3g} s")


if __name__ == "__main__":
    main()
