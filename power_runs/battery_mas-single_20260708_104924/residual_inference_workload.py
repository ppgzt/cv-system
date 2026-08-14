"""Residual inference workload at the end of capture.

This module is the executable implementation used by
``residual_inference_workload_analysis.ipynb``.  It reads only the 45
``metrics.json`` files from the fixed-FPS campaign.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


FPS_LEVELS = [1, 2, 3, 4, 5, 10, 15, 20, 30]
EXPECTED_RUNS = [1, 2, 3, 4, 5]
EXPECTED_PASSAGES_PER_RUN = 184
LATENCY_IDENTITY_ATOL_S = 1e-6
RUN_DIR_RE = re.compile(r"mas-single_(\d+)fps_r(\d+)$")

PASSAGE_COLUMNS = [
    "fps",
    "run",
    "animal_id",
    "suitable_images",
    "active_inferences",
    "not_started_inferences",
    "residual_inferences",
    "residual_fraction",
    "has_residual",
    "prediction_drain_s",
    "post_capture_latency_s",
    "finalization_overhead_s",
]

RUN_METRICS = [
    "proportion_passages_with_residual",
    "proportion_passages_with_active_inference",
    "proportion_passages_with_not_started_inference",
    "residual_inferences_mean",
    "residual_inferences_median",
    "residual_inferences_p95",
    "active_inferences_mean",
    "active_inferences_p95",
    "active_inferences_max",
    "not_started_inferences_mean",
    "not_started_inferences_median",
    "not_started_inferences_p95",
    "not_started_share_of_residual",
    "prediction_drain_s_mean",
    "prediction_drain_s_median",
    "prediction_drain_s_p95",
    "post_capture_latency_s_mean",
    "post_capture_latency_s_p95",
]


def _p95(values: pd.Series) -> float:
    """Linear empirical 95th percentile, ignoring unavailable values."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.quantile(0.95)) if len(values) else math.nan


def _timestamp(
    value: Any,
    field: str,
    context: dict[str, Any],
    issues: list[dict[str, Any]],
) -> pd.Timestamp:
    """Parse one ISO timestamp and register missing/invalid values."""
    if value is None or (isinstance(value, str) and not value.strip()):
        issues.append({**context, "issue": "missing_timestamp", "field": field, "value": value})
        return pd.NaT
    try:
        parsed = pd.Timestamp(value)
    except Exception:
        parsed = pd.NaT
    if pd.isna(parsed):
        issues.append({**context, "issue": "invalid_timestamp", "field": field, "value": value})
    return parsed


def discover_metrics(campaign_dir: Path) -> pd.DataFrame:
    """Discover exactly one metrics file for every requested FPS/run pair."""
    rows: list[dict[str, Any]] = []
    for path in sorted(campaign_dir.glob("mas-single_*fps_r*/*/metrics.json")):
        match = RUN_DIR_RE.fullmatch(path.parents[1].name)
        if not match:
            continue
        fps, run = map(int, match.groups())
        if fps in FPS_LEVELS and run in EXPECTED_RUNS:
            rows.append({"fps": fps, "run_number": run, "run": f"r{run}", "metrics_path": path})

    files = pd.DataFrame(rows)
    expected = pd.MultiIndex.from_product([FPS_LEVELS, EXPECTED_RUNS], names=["fps", "run_number"])
    found = pd.MultiIndex.from_frame(files[["fps", "run_number"]]) if len(files) else pd.MultiIndex.from_tuples([])
    missing = expected.difference(found)
    duplicates = files.groupby(["fps", "run_number"]).size() if len(files) else pd.Series(dtype=int)
    duplicates = duplicates[duplicates.ne(1)]
    if len(files) != 45 or len(missing) or len(duplicates):
        raise ValueError(
            f"Expected 45 unique metrics.json files; found={len(files)}, "
            f"missing={list(missing)}, duplicate_counts={duplicates.to_dict()}"
        )
    return files.sort_values(["fps", "run_number"], ignore_index=True)


def extract_passages(files: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract passage metrics and retain a separate audit trail."""
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    run_counts: list[dict[str, Any]] = []

    for spec in files.itertuples(index=False):
        with Path(spec.metrics_path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        animals = payload.get("animals")
        if not isinstance(animals, dict):
            raise ValueError(f"animals is not an object in {spec.metrics_path}")
        run_counts.append({"fps": spec.fps, "run": spec.run, "passages": len(animals)})

        for animal_id, passage in animals.items():
            base = {"fps": spec.fps, "run": spec.run, "animal_id": str(animal_id)}
            last_capture = _timestamp(
                passage.get("last_image_capture_time"), "last_image_capture_time", base, issues
            )
            passage_final = _timestamp(
                passage.get("weight_prediction_final"), "passage.weight_prediction_final", base, issues
            )

            suitable_raw = passage.get("suitable_images")
            try:
                suitable_images = int(suitable_raw)
                if suitable_images < 0:
                    raise ValueError
            except (TypeError, ValueError):
                suitable_images = pd.NA
                issues.append(
                    {**base, "issue": "invalid_suitable_images", "field": "suitable_images", "value": suitable_raw}
                )

            imgs = passage.get("imgs", {})
            if not isinstance(imgs, dict):
                issues.append({**base, "issue": "invalid_imgs", "field": "imgs", "value": type(imgs).__name__})
                imgs = {}

            starts: list[pd.Timestamp] = []
            finals: list[pd.Timestamp] = []
            image_timestamps_valid = True
            for image_id, image in imgs.items():
                image_context = {**base, "image_id": str(image_id)}
                if not isinstance(image, dict):
                    issues.append(
                        {**image_context, "issue": "invalid_image_record", "field": "imgs", "value": type(image).__name__}
                    )
                    image_timestamps_valid = False
                    continue
                start = _timestamp(
                    image.get("weight_prediction_start"),
                    "imgs.weight_prediction_start",
                    image_context,
                    issues,
                )
                final = _timestamp(
                    image.get("weight_prediction_final"),
                    "imgs.weight_prediction_final",
                    image_context,
                    issues,
                )
                if pd.isna(start) or pd.isna(final):
                    image_timestamps_valid = False
                    continue
                if start > final:
                    issues.append(
                        {
                            **image_context,
                            "issue": "prediction_start_after_final",
                            "field": "imgs",
                            "value": f"{start.isoformat()} > {final.isoformat()}",
                        }
                    )
                    image_timestamps_valid = False
                starts.append(start)
                finals.append(final)

            if suitable_images is not pd.NA and suitable_images != len(imgs):
                issues.append(
                    {
                        **base,
                        "issue": "suitable_images_imgs_mismatch",
                        "field": "suitable_images/imgs",
                        "value": f"{suitable_images}/{len(imgs)}",
                    }
                )

            valid_for_residual = pd.notna(last_capture) and image_timestamps_valid
            if valid_for_residual:
                active = sum(start <= last_capture < final for start, final in zip(starts, finals))
                not_started = sum(start > last_capture for start in starts)
                residual = active + not_started
                residual_by_final = sum(final > last_capture for final in finals)
                identity_ok: Any = residual == residual_by_final
                max_image_final = max(finals) if finals else last_capture
                raw_drain_s = (max_image_final - last_capture).total_seconds()
                prediction_drain_s = max(0.0, raw_drain_s)
            else:
                active = not_started = residual = residual_by_final = pd.NA
                identity_ok = pd.NA
                max_image_final = pd.NaT
                prediction_drain_s = math.nan

            if pd.notna(last_capture) and pd.notna(passage_final):
                post_capture_latency_s = (passage_final - last_capture).total_seconds()
                comparison_time = last_capture
                if pd.notna(max_image_final):
                    comparison_time = max(last_capture, max_image_final)
                finalization_overhead_s = (passage_final - comparison_time).total_seconds()
            else:
                post_capture_latency_s = finalization_overhead_s = math.nan

            if suitable_images is not pd.NA and suitable_images > 0 and residual is not pd.NA:
                residual_fraction = residual / suitable_images
            else:
                # 0/0 is undefined; export it as an empty CSV field rather than as zero.
                residual_fraction = math.nan

            rows.append(
                {
                    **base,
                    "suitable_images": suitable_images,
                    "active_inferences": active,
                    "not_started_inferences": not_started,
                    "residual_inferences": residual,
                    "residual_fraction": residual_fraction,
                    "has_residual": bool(residual > 0) if residual is not pd.NA else pd.NA,
                    "prediction_drain_s": prediction_drain_s,
                    "post_capture_latency_s": post_capture_latency_s,
                    "finalization_overhead_s": finalization_overhead_s,
                    "_residual_by_prediction_final": residual_by_final,
                    "_residual_identity_ok": identity_ok,
                }
            )

    passage_df = pd.DataFrame(rows)
    for column in ["suitable_images", "active_inferences", "not_started_inferences", "residual_inferences"]:
        passage_df[column] = passage_df[column].astype("Int64")
    passage_df["has_residual"] = passage_df["has_residual"].astype("boolean")
    issue_df = pd.DataFrame(issues, columns=["fps", "run", "animal_id", "image_id", "issue", "field", "value"])
    return passage_df.sort_values(["fps", "run", "animal_id"], ignore_index=True), issue_df, pd.DataFrame(run_counts)


def summarize_by_run(passage_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate passages without treating them as treatment replicates."""
    rows: list[dict[str, Any]] = []
    for (fps, run), group in passage_df.groupby(["fps", "run"], sort=True):
        residual_sum = group["residual_inferences"].sum(min_count=1)
        not_started_sum = group["not_started_inferences"].sum(min_count=1)
        not_started_share = (
            float(not_started_sum / residual_sum)
            if pd.notna(residual_sum) and residual_sum != 0
            else math.nan
        )
        rows.append(
            {
                "fps": fps,
                "run": run,
                "passages": len(group),
                "proportion_passages_with_residual": float(group["has_residual"].mean()),
                "proportion_passages_with_active_inference": float(
                    group["active_inferences"].gt(0).mean()
                ),
                "proportion_passages_with_not_started_inference": float(
                    group["not_started_inferences"].gt(0).mean()
                ),
                "residual_inferences_mean": float(group["residual_inferences"].mean()),
                "residual_inferences_median": float(group["residual_inferences"].median()),
                "residual_inferences_p95": _p95(group["residual_inferences"]),
                "active_inferences_mean": float(group["active_inferences"].mean()),
                "active_inferences_p95": _p95(group["active_inferences"]),
                "active_inferences_max": float(group["active_inferences"].max()),
                "not_started_inferences_mean": float(group["not_started_inferences"].mean()),
                "not_started_inferences_median": float(group["not_started_inferences"].median()),
                "not_started_inferences_p95": _p95(group["not_started_inferences"]),
                "not_started_share_of_residual": not_started_share,
                "prediction_drain_s_mean": float(group["prediction_drain_s"].mean()),
                "prediction_drain_s_median": float(group["prediction_drain_s"].median()),
                "prediction_drain_s_p95": _p95(group["prediction_drain_s"]),
                "post_capture_latency_s_mean": float(group["post_capture_latency_s"].mean()),
                "post_capture_latency_s_p95": _p95(group["post_capture_latency_s"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["fps", "run"], ignore_index=True)


def summarize_by_fps(run_df: pd.DataFrame) -> pd.DataFrame:
    """Mean, sample SD and t-based 95% CI over five complete runs per FPS."""
    rows: list[dict[str, Any]] = []
    for fps, group in run_df.groupby("fps", sort=True):
        for metric in RUN_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            n = len(values)
            mean = float(values.mean()) if n else math.nan
            std = float(values.std(ddof=1)) if n > 1 else math.nan
            margin = float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n)) if n > 1 else math.nan
            rows.append(
                {
                    "fps": fps,
                    "metric": metric,
                    "n_runs": n,
                    "mean": mean,
                    "std": std,
                    "ci95_low": mean - margin if n > 1 else math.nan,
                    "ci95_high": mean + margin if n > 1 else math.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["fps", "metric"], ignore_index=True)


def decomposition_summary_by_fps(fps_df: pd.DataFrame) -> pd.DataFrame:
    """Return the compact FPS table for the two residual-workload components."""
    summary = pd.DataFrame({"FPS": FPS_LEVELS})
    for metric in [
        "active_inferences_mean",
        "not_started_inferences_mean",
        "not_started_inferences_p95",
        "not_started_share_of_residual",
    ]:
        values = _metric_rows(fps_df, metric).set_index("fps")["mean"]
        summary[metric] = summary["FPS"].map(values)
    return summary


def integrity_checks(
    files: pd.DataFrame,
    passage_df: pd.DataFrame,
    issue_df: pd.DataFrame,
    run_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Return the requested integrity checks as a compact table."""
    timestamp_issues = issue_df[issue_df["issue"].isin(["missing_timestamp", "invalid_timestamp"])]
    counts_ok = bool(run_counts["passages"].eq(EXPECTED_PASSAGES_PER_RUN).all() and len(run_counts) == 45)
    runs_per_fps = files.groupby("fps")["run"].nunique().reindex(FPS_LEVELS, fill_value=0)
    comparable = passage_df.dropna(subset=["residual_inferences", "suitable_images"])
    residual_bound_violations = comparable[comparable["residual_inferences"] > comparable["suitable_images"]]
    identity_violations = passage_df[passage_df["_residual_identity_ok"].eq(False)]
    active_worker_violations = passage_df[passage_df["active_inferences"].gt(1)]
    decomposition_comparable = passage_df.dropna(
        subset=["active_inferences", "not_started_inferences", "residual_inferences"]
    )
    decomposition_violations = decomposition_comparable[
        decomposition_comparable["active_inferences"]
        + decomposition_comparable["not_started_inferences"]
        != decomposition_comparable["residual_inferences"]
    ]
    prediction_order_violations = issue_df[
        issue_df["issue"].eq("prediction_start_after_final")
    ]
    negative_drain = passage_df[passage_df["prediction_drain_s"] < 0]
    negative_post = passage_df[passage_df["post_capture_latency_s"] < 0]
    negative_finalization_overhead = passage_df[
        passage_df["finalization_overhead_s"] < 0
    ]
    latency_comparable = passage_df.dropna(
        subset=["post_capture_latency_s", "prediction_drain_s", "finalization_overhead_s"]
    ).copy()
    latency_error_s = (
        latency_comparable["post_capture_latency_s"]
        - latency_comparable["prediction_drain_s"]
        - latency_comparable["finalization_overhead_s"]
    ).abs()
    latency_identity_violations = latency_comparable[
        latency_error_s.gt(LATENCY_IDENTITY_ATOL_S)
    ]

    checks = [
        ("45 metrics.json exclusivos", len(files) == 45, 45 - len(files), f"encontrados={len(files)}"),
        (
            "cinco runs por FPS",
            bool(runs_per_fps.eq(5).all()),
            int((runs_per_fps != 5).sum()),
            ", ".join(f"{fps} FPS={count}" for fps, count in runs_per_fps.items()),
        ),
        (
            "184 passagens por run",
            counts_ok,
            int((run_counts["passages"] != EXPECTED_PASSAGES_PER_RUN).sum()),
            f"mín={run_counts['passages'].min()}, máx={run_counts['passages'].max()}",
        ),
        (
            "residual_inferences <= suitable_images",
            residual_bound_violations.empty,
            len(residual_bound_violations),
            "comparações válidas=" + str(len(comparable)),
        ),
        (
            "active_inferences <= 1",
            active_worker_violations.empty,
            len(active_worker_violations),
            "um único worker de predição",
        ),
        (
            "active_inferences + not_started_inferences == residual_inferences",
            decomposition_violations.empty,
            len(decomposition_violations),
            f"passagens verificadas={len(decomposition_comparable)}",
        ),
        (
            "prediction_start <= prediction_final",
            prediction_order_violations.empty,
            len(prediction_order_violations),
            "comparação por imagem",
        ),
        (
            "timestamps ausentes ou inválidos",
            timestamp_issues.empty,
            len(timestamp_issues),
            "campos de passagem e de imagem",
        ),
        (
            "prediction_drain_s não negativo",
            negative_drain.empty,
            len(negative_drain),
            "após aplicação de max(0, ...)",
        ),
        (
            "post_capture_latency_s não negativo",
            negative_post.empty,
            len(negative_post),
            "weight_prediction_final da passagem - last_capture",
        ),
        (
            "finalization_overhead_s >= 0",
            negative_finalization_overhead.empty,
            len(negative_finalization_overhead),
            "weight_prediction_final da passagem - fim da drenagem",
        ),
        (
            "post_capture_latency_s ≈ prediction_drain_s + finalization_overhead_s",
            latency_identity_violations.empty,
            len(latency_identity_violations),
            (
                f"tolerância absoluta={LATENCY_IDENTITY_ATOL_S:g} s; "
                f"erro máximo={latency_error_s.max():.3g} s"
            ),
        ),
        (
            "identidade residual por prediction_final > last_capture",
            identity_violations.empty,
            len(identity_violations),
            f"passagens verificadas={passage_df['_residual_identity_ok'].notna().sum()}",
        ),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "violations", "details"])


def kruskal_wallis_by_run(run_df: pd.DataFrame) -> pd.DataFrame:
    """Global tests only when run-level values have non-trivial variation."""
    rows: list[dict[str, Any]] = []
    for metric in RUN_METRICS:
        groups = [pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy() for _, g in run_df.groupby("fps")]
        groups = [g for g in groups if len(g)]
        all_values = np.concatenate(groups) if groups else np.array([])
        overall_variable = len(all_values) > 1 and not np.allclose(all_values, all_values[0], rtol=1e-10, atol=1e-12)
        within_group_variable = any(
            len(g) > 1 and not np.allclose(g, g[0], rtol=1e-10, atol=1e-12) for g in groups
        )
        enough_groups = len(groups) == len(FPS_LEVELS) and all(len(g) >= 2 for g in groups)

        if enough_groups and overall_variable and within_group_variable:
            result = stats.kruskal(*groups, nan_policy="omit")
            rows.append(
                {
                    "metric": metric,
                    "status": "executed",
                    "n_runs": len(all_values),
                    "groups": len(groups),
                    "H": float(result.statistic),
                    "df": len(groups) - 1,
                    "p_value": float(result.pvalue),
                    "reason": "variação observada em métricas consolidadas por run",
                }
            )
        else:
            reasons = []
            if not enough_groups:
                reasons.append("grupos insuficientes")
            if not overall_variable:
                reasons.append("sem variação global")
            if not within_group_variable:
                reasons.append("valores determinísticos dentro de todos os FPS")
            rows.append(
                {
                    "metric": metric,
                    "status": "not_run",
                    "n_runs": len(all_values),
                    "groups": len(groups),
                    "H": math.nan,
                    "df": math.nan,
                    "p_value": math.nan,
                    "reason": "; ".join(reasons),
                }
            )
    return pd.DataFrame(rows)


def _metric_rows(fps_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return fps_df[fps_df["metric"].eq(metric)].sort_values("fps")


def _asymmetric_error(frame: pd.DataFrame) -> np.ndarray:
    mean = frame["mean"].to_numpy(dtype=float)
    return np.vstack([mean - frame["ci95_low"].to_numpy(), frame["ci95_high"].to_numpy() - mean])


def make_figures(fps_df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Create the requested compact two-panel figure and the drain figure."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
        }
    )
    colors = {"mean": "#1f77b4", "p95": "#d95f02"}

    proportion = _metric_rows(fps_df, "proportion_passages_with_residual")
    residual_mean = _metric_rows(fps_df, "residual_inferences_mean")
    residual_p95 = _metric_rows(fps_df, "residual_inferences_p95")

    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.25), constrained_layout=True)
    axes[0].errorbar(
        proportion["fps"],
        proportion["mean"],
        yerr=_asymmetric_error(proportion),
        fmt="o-",
        color=colors["mean"],
        capsize=3,
        lw=1.5,
    )
    axes[0].set(
        title="(a) Passagens com trabalho residual",
        xlabel="FPS configurado",
        ylabel="Proporção de passagens",
        xticks=FPS_LEVELS,
        ylim=(-0.03, 1.03),
    )
    axes[0].grid(axis="y", alpha=0.25)

    for frame, label, color, marker in [
        (residual_mean, "Média", colors["mean"], "o"),
        (residual_p95, "P95", colors["p95"], "s"),
    ]:
        axes[1].errorbar(
            frame["fps"],
            frame["mean"],
            yerr=_asymmetric_error(frame),
            fmt=marker + "-",
            label=label,
            color=color,
            capsize=3,
            lw=1.5,
        )
    axes[1].set(
        title="(b) Inferências residuais por passagem",
        xlabel="FPS configurado",
        ylabel="Inferências",
        xticks=FPS_LEVELS,
    )
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"residual_workload_by_fps.{suffix}", bbox_inches="tight")
    plt.close(fig)

    drain_mean = _metric_rows(fps_df, "prediction_drain_s_mean")
    drain_p95 = _metric_rows(fps_df, "prediction_drain_s_p95")
    positive = np.concatenate(
        [drain_mean.loc[drain_mean["mean"] > 0, "mean"].to_numpy(), drain_p95.loc[drain_p95["mean"] > 0, "mean"].to_numpy()]
    )
    use_symlog = len(positive) > 1 and positive.max() / positive.min() >= 50

    fig, ax = plt.subplots(figsize=(5.5, 3.35), constrained_layout=True)
    for frame, label, color, marker in [
        (drain_mean, "Média", colors["mean"], "o"),
        (drain_p95, "P95", colors["p95"], "s"),
    ]:
        ax.errorbar(
            frame["fps"],
            frame["mean"],
            yerr=_asymmetric_error(frame),
            fmt=marker + "-",
            label=label,
            color=color,
            capsize=3,
            lw=1.5,
        )
    if use_symlog:
        ax.set_yscale("symlog", linthresh=0.01, linscale=0.8)
    ax.set(
        title="Drenagem das inferências após a última captura",
        xlabel="FPS configurado",
        ylabel="prediction_drain_s (s)",
        xticks=FPS_LEVELS,
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", which="both", alpha=0.25)
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"prediction_drain_by_fps.{suffix}", bbox_inches="tight")
    plt.close(fig)
    return {"prediction_drain_scale": "symlog" if use_symlog else "linear"}


def _format_number(value: Any, digits: int = 4) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _markdown_table(frame: pd.DataFrame, formats: dict[str, int] | None = None) -> str:
    """Render a small DataFrame without depending on the tabulate package."""
    formats = formats or {}
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            if column in formats:
                values.append(_format_number(value, formats[column]))
            elif isinstance(value, (float, np.floating)):
                values.append(_format_number(value, 4))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    passage_df: pd.DataFrame,
    run_df: pd.DataFrame,
    fps_df: pd.DataFrame,
    decomposition_summary: pd.DataFrame,
    integrity_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    figure_info: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Write a concise Markdown report from run-level inference."""
    summary = pd.DataFrame({"FPS": FPS_LEVELS})
    metrics_for_summary = {
        "proportion_passages_with_residual": "Proporção com residual",
        "residual_inferences_mean": "Residual médio",
        "residual_inferences_p95": "Residual P95",
        "prediction_drain_s_mean": "Drenagem média (s)",
        "post_capture_latency_s_mean": "Latência pós-captura média (s)",
    }
    for metric, label in metrics_for_summary.items():
        values = _metric_rows(fps_df, metric).set_index("fps")["mean"]
        summary[label] = summary["FPS"].map(values)

    executed = tests_df[tests_df["status"].eq("executed")].copy()
    skipped = tests_df[tests_df["status"].ne("executed")].copy()
    zero_suitable = int(passage_df["suitable_images"].eq(0).sum())
    highest = summary.loc[summary["Proporção com residual"].idxmax()]
    minimum_proportion = summary["Proporção com residual"].min()
    lowest_fps = summary.loc[
        np.isclose(summary["Proporção com residual"], minimum_proportion), "FPS"
    ].astype(int)
    lowest_fps_text = ", ".join(map(str, lowest_fps))
    passage_count_text = f"{len(passage_df):,}".replace(",", ".")

    integrity_report = integrity_df.copy()
    integrity_report["passed"] = integrity_report["passed"].map({True: "OK", False: "FALHA"})
    test_columns = ["metric", "H", "df", "p_value"]

    report = f"""# Trabalho residual de inferência — battery_mas-single_20260708_104924

## Escopo e definição

A análise usa exclusivamente os 45 arquivos `metrics.json` das configurações 1, 2, 3, 4, 5, 10, 15, 20 e 30 FPS, com cinco execuções completas por FPS. Foram analisadas {passage_count_text} passagens.

`residual_inferences` representa **residual inference workload at the end of capture**: inferências de imagens adequadas cuja finalização ocorreu depois da última captura. `not_started_inferences` representa **inferences not yet started at the prediction stage** no instante da última captura.

Nas {zero_suitable} passagens com `suitable_images = 0`, `residual_fraction` é indefinida e foi exportada como campo vazio; `has_residual` é falso.

## Principais resultados

{_markdown_table(summary, {column: (0 if column == 'FPS' else 4) for column in summary.columns})}

A maior proporção média de passagens com trabalho residual ocorreu em {int(highest['FPS'])} FPS ({highest['Proporção com residual']:.3f}); a menor ocorreu, com empate, em {lowest_fps_text} FPS ({minimum_proportion:.3f}). A figura principal apresenta IC95% calculados sobre as cinco runs completas de cada FPS.

### Decomposição do trabalho residual por FPS

{_markdown_table(decomposition_summary, {column: (0 if column == 'FPS' else 4) for column in decomposition_summary.columns})}

Cada valor é a média das cinco métricas consolidadas por run no FPS correspondente. A estatística `not_started_share_of_residual` é calculada primeiro em cada run como a soma de `not_started_inferences` dividida pela soma de `residual_inferences`; quando a soma residual é zero, o valor da run é `NaN`.

## Integridade

{_markdown_table(integrity_report)}

Além das verificações de cobertura e esquema, foram testados o limite de uma inferência ativa, a decomposição exata do trabalho residual, a ordem dos timestamps de predição, a não negatividade de `finalization_overhead_s` e a identidade temporal com tolerância absoluta de 1e-6 s.

## Inferência no nível da run

A unidade experimental é a execução completa (n=5 por FPS). As 184 passagens de uma run não foram tratadas como réplicas independentes do tratamento. Os IC95% usam a distribuição t de Student sobre as cinco métricas consolidadas por run (média ± t(0,975; 4) × erro-padrão).

O Kruskal–Wallis foi aplicado apenas quando havia nove grupos completos e variação observada nas métricas consolidadas por run. Nenhum pós-hoc foi executado automaticamente.

### Testes globais executados

{_markdown_table(executed[test_columns], {'H': 4, 'df': 0, 'p_value': 6}) if len(executed) else 'Nenhum teste atendeu ao critério de variação.'}

### Métricas sem teste global

{_markdown_table(skipped[['metric', 'reason']]) if len(skipped) else 'Todas as métricas apresentaram variação suficiente.'}

## Figuras

- `residual_workload_by_fps.png` e `.pdf`: proporção de passagens com trabalho residual e média/P95 de inferências residuais por passagem.
- `prediction_drain_by_fps.png` e `.pdf`: média/P95 de `prediction_drain_s`; escala {figure_info['prediction_drain_scale']} para evitar compressão das configurações abaixo de 30 FPS.

## Arquivos tabulares

- `residual_by_passage.csv`: uma linha por passagem.
- `residual_by_run.csv`: uma linha por execução completa.
- `residual_by_fps.csv`: formato longo, com média, desvio padrão amostral e IC95% das métricas no nível da run.
- `residual_components_by_fps.csv`: tabela compacta da decomposição por FPS.

## Observações metodológicas

- As comparações de timestamp usam `prediction_start <= last_capture < prediction_final` para inferências ativas e `prediction_start > last_capture` para inferências ainda não iniciadas.
- `prediction_drain_s` é truncado inferiormente em zero conforme a definição solicitada.
- O P95 usa quantil empírico com interpolação linear.
- `finalization_overhead_s` pode ser interpretado separadamente do tempo de drenagem; quando não há imagens adequadas, o máximo por imagem é substituído por `last_capture`.
"""
    path = output_dir / "residual_inference_workload_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def run_analysis(campaign_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Run the complete analysis and return tables plus artifact paths."""
    campaign_dir = Path(campaign_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else campaign_dir / "residual_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_metrics(campaign_dir)
    passage_df, issue_df, run_counts = extract_passages(files)
    run_df = summarize_by_run(passage_df)
    fps_df = summarize_by_fps(run_df)
    decomposition_summary = decomposition_summary_by_fps(fps_df)
    integrity_df = integrity_checks(files, passage_df, issue_df, run_counts)
    tests_df = kruskal_wallis_by_run(run_df)

    if not integrity_df["passed"].all():
        failures = integrity_df.loc[~integrity_df["passed"], ["check", "violations", "details"]]
        raise ValueError("Integrity checks failed:\n" + failures.to_string(index=False))

    passage_path = output_dir / "residual_by_passage.csv"
    run_path = output_dir / "residual_by_run.csv"
    fps_path = output_dir / "residual_by_fps.csv"
    decomposition_path = output_dir / "residual_components_by_fps.csv"
    passage_df[PASSAGE_COLUMNS].to_csv(passage_path, index=False, float_format="%.9f")
    run_df.to_csv(run_path, index=False, float_format="%.9f")
    fps_df.to_csv(fps_path, index=False, float_format="%.9f")
    decomposition_summary.to_csv(decomposition_path, index=False, float_format="%.9f")

    figure_info = make_figures(fps_df, output_dir)
    report_path = write_report(
        passage_df,
        run_df,
        fps_df,
        decomposition_summary,
        integrity_df,
        tests_df,
        figure_info,
        output_dir,
    )
    return {
        "passage": passage_df[PASSAGE_COLUMNS].copy(),
        "run": run_df,
        "fps": fps_df,
        "decomposition_summary": decomposition_summary,
        "integrity": integrity_df,
        "timestamp_and_schema_issues": issue_df,
        "kruskal_wallis": tests_df,
        "figure_info": figure_info,
        "paths": {
            "passage_csv": passage_path,
            "run_csv": run_path,
            "fps_csv": fps_path,
            "decomposition_csv": decomposition_path,
            "report": report_path,
            "residual_png": output_dir / "residual_workload_by_fps.png",
            "residual_pdf": output_dir / "residual_workload_by_fps.pdf",
            "drain_png": output_dir / "prediction_drain_by_fps.png",
            "drain_pdf": output_dir / "prediction_drain_by_fps.pdf",
        },
    }


if __name__ == "__main__":
    result = run_analysis(Path(__file__).resolve().parent)
    print(result["decomposition_summary"].to_string(index=False))
    print()
    print(result["integrity"].to_string(index=False))
    print("\nArtifacts:")
    for name, path in result["paths"].items():
        print(f"- {name}: {path}")
