"""Geração de todos os artefatos de saída do benchmark (§12, §13, §16, §17).

Escreve, num diretório de relatório por execução:
- *_measurements.csv  (uma linha por medição válida)
- system_monitor.csv  (amostras 1 Hz da thread de monitoramento)
- failures.csv        (falhas de validação/runtime, separadas)
- warmup_<comp>.csv   (medições de warm-up, separadas das válidas)
- metadata.json       (config + hw + sw + modelos + pools + bootstrap)
- summary.json        (estatística completa por componente/métrica)
- summary.csv         (versão tabular compacta)
- report.md           (relatório legível)
- article_table.{csv,md,tex}  (tabela compacta para o artigo)
- article_table_detailed.{csv,md,tex}
- plots/*.png + *.pdf (gráficos; opcionais se matplotlib ausente)

Todos os dados ficam nos CSVs — os gráficos são conveniência.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime

import numpy as np

from . import statistics as st

# Campos (ordem) de cada CSV de medições, conforme especificação.
CSV_FIELDS = {
    "selector": [
        "iteration", "image_id", "animal_id", "true_class", "predicted_class",
        "predicted_argmax", "score", "total_stage_ns", "total_stage_ms",
        "tflite_total_ns", "tflite_total_ms", "invoke_ns", "invoke_ms",
        "input_shape", "input_dtype", "timestamp_monotonic_ns", "timestamp_utc",
    ],
    "enhancer": [
        "iteration", "image_id", "animal_id", "total_stage_ns", "total_stage_ms",
        "input_shape", "input_dtype", "output_shape", "output_dtype",
        "output_min", "output_max", "output_mean", "output_std",
        "noise_removal_ns", "noise_removal_ms", "adjust_scale_ns",
        "adjust_scale_ms", "replicate_ns", "replicate_ms",
        "resize_pad_ns", "resize_pad_ms",
        "timestamp_monotonic_ns", "timestamp_utc",
    ],
    "predictor": [
        "iteration", "image_id", "animal_id", "prediction",
        "total_stage_ns", "total_stage_ms", "tflite_total_ns", "tflite_total_ms",
        "invoke_ns", "invoke_ms", "input_shape", "input_dtype",
        "output_shape", "output_dtype", "timestamp_monotonic_ns", "timestamp_utc",
    ],
    "aggregation": [
        "iteration", "num_predictions", "aggregation_ns", "aggregation_ms",
        "result", "timestamp_monotonic_ns", "timestamp_utc",
    ],
}

MONITOR_FIELDS = [
    "timestamp_utc", "timestamp_monotonic_ns", "component",
    "cpu_total_percent", "process_cpu_percent", "process_rss_bytes",
    "mem_available_bytes", "temperature_celsius", "cpu_freq_cur_hz",
    "cpu_freq_min_hz", "cpu_freq_max_hz", "cpu_governor", "throttled_raw",
    "throttled_now", "undervoltage_now", "freq_capped_now",
]

FAILURE_FIELDS = [
    "component", "iteration", "image_id", "animal_id",
    "exception_type", "message", "timestamp_utc",
]

# Rótulos amigáveis para o artigo.
STAGE_LABEL = {
    "selector": ("Frame selection", "TFLite (MobileNetV2)"),
    "enhancer": ("Data enhancement", "NumPy + TF (CPU)"),
    "predictor": ("Weight estimation", "TFLite (EfficientNet-B3)"),
    "aggregation": ("Final aggregation", "NumPy (CPU)"),
}


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "not_available"


def file_size_mb(path: str):
    try:
        return round(os.path.getsize(path) / 1e6, 3)
    except Exception:
        return None


def write_csv(path: str, rows: list[dict], fieldnames=None):
    if not rows:
        # cria o arquivo vazio (com cabeçalho) mesmo sem dados
        fieldnames = fieldnames or []
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        return
    if fieldnames is None:
        # deriva ordem estável: chaves do primeiro + extras não vistos
        fieldnames = list(rows[0].keys())
        for r in rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _monitor_arrays(monitor_rows: list[dict], key: str):
    """Extrai séries (times_ns, vals) do monitor para uma dada chave."""
    times, vals = [], []
    for r in monitor_rows:
        v = r.get(key)
        t = r.get("timestamp_monotonic_ns")
        if t is None or v is None or v == "not_available":
            continue
        try:
            vals.append(float(v))
            times.append(float(t))
        except (TypeError, ValueError):
            continue
    return np.asarray(times), np.asarray(vals)


# --------------------------------------------------------------------------- #
# Resumo por métrica
# --------------------------------------------------------------------------- #
def summarize_values(arr: np.ndarray, monitor_rows: list[dict],
                     lat_times: np.ndarray, seed: int, n_boot: int,
                     field: str = "") -> dict:
    """Resumo completo de uma série temporal (nanossegundos)."""
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"field": field, "n_valid": 0}
    desc = st.describe_ns(arr)
    bootstrap_stats = ["median"] + [f"p{q}" for q in st._PERCENTILES
                                      if q != 50]
    bootstrap_all = st.bootstrap_cis(
        arr, statistics=bootstrap_stats, n_boot=n_boot, seed=seed)
    thr = st.throughput_ops_per_sec(desc.get("mean_ns"),
                                    desc.get("median_ns"))
    slope = st.temporal_slope_ns_per_iter(arr)
    splits = st.split_halves(arr)
    fl = st.first_last_n(arr, 100)
    blocks = st.block_stats(arr, 100)

    corr = {}
    for env_key, label in [("temperature_celsius", "temperature"),
                           ("cpu_freq_cur_hz", "cpu_freq"),
                           ("process_cpu_percent", "process_cpu"),
                           ("cpu_total_percent", "cpu_total")]:
        et, ev = _monitor_arrays(monitor_rows, env_key)
        corr[label] = (st.correlation(lat_times, arr, et, ev)
                       if et.size else
                       {"pearson_r": None, "pvalue": None, "n": 0})

    mask = st.iqr_outlier_mask(arr)
    desc_no_out = st.describe_ns(arr[mask]) if mask.size else {"n_valid": 0}

    return {
        "field": field,
        "describe": desc,
        "bootstrap_median_ci": bootstrap_all.get("median"),
        "bootstrap_percentile_cis": {
            key: value for key, value in bootstrap_all.items()
            if key != "median"
        },
        "throughput": thr,
        "temporal_slope": slope,
        "split_halves": splits,
        "first_last_100": fl,
        "blocks_of_100": blocks,
        "correlations": corr,
        "iqr_outliers_removed_describe": desc_no_out,
        "n_outliers_iqr": int(desc.get("n_outliers_iqr", 0)),
    }


def summarize_metric(measurements: list[dict], field: str,
                     monitor_rows: list[dict], seed: int,
                     n_boot: int) -> dict:
    vals = [m[field] for m in measurements
            if isinstance(m.get(field), (int, float))]
    lat_times = np.asarray([m.get("timestamp_monotonic_ns", np.nan)
                            for m in measurements], dtype=np.float64)
    return summarize_values(np.asarray(vals, dtype=np.float64),
                            monitor_rows, lat_times, seed, n_boot, field=field)


# --------------------------------------------------------------------------- #
# Reporter
# --------------------------------------------------------------------------- #
class Reporter:
    def __init__(self, report_dir: str, config: dict, env_before: dict,
                 env_after: dict, model_infos: dict, pool_stats: dict,
                 bootstrap_cfg: dict, plots: bool = True):
        self.dir = report_dir
        self.config = config
        self.env_before = env_before
        self.env_after = env_after
        self.model_infos = model_infos          # {component: {path,sha256,size_mb,...}}
        self.pool_stats = pool_stats            # {component: {...unique counts...}}
        self.bootstrap_cfg = bootstrap_cfg
        self.plots = plots
        os.makedirs(self.dir, exist_ok=True)
        self.summaries: dict = {}

    # ------------------------------------------------------------------ #
    def write_measurements(self, component: str, rows: list[dict]):
        write_csv(os.path.join(self.dir, f"{component}_measurements.csv"),
                  rows, CSV_FIELDS.get(component))

    def write_warmup(self, component: str, rows: list[dict]):
        if rows:
            write_csv(os.path.join(self.dir, f"warmup_{component}.csv"),
                      rows, CSV_FIELDS.get(component))

    def write_monitor(self, rows: list[dict]):
        write_csv(os.path.join(self.dir, "system_monitor.csv"),
                  rows, MONITOR_FIELDS)

    def write_failures(self, rows: list[dict]):
        write_csv(os.path.join(self.dir, "failures.csv"),
                  rows, FAILURE_FIELDS)

    # ------------------------------------------------------------------ #
    def compute_summaries(self, components: dict, all_monitor_rows: list[dict]):
        """components: {name: {benchmark, run_result, reuse_counts}}."""
        seed = self.bootstrap_cfg["seed"]
        n_boot = self.bootstrap_cfg["n_resamples"]
        out = {"bootstrap": dict(self.bootstrap_cfg),
               "components": {}}
        flat_rows = []  # para summary.csv e tabelas

        for name, info in components.items():
            bench = info["benchmark"]
            measurements = bench.measurements
            comp_monitor = [r for r in all_monitor_rows
                            if r.get("component") == name]
            metrics_for_comp = self._metric_fields(name, bench)
            metric_summaries = {}
            for field in metrics_for_comp:
                metric_summaries[field] = summarize_metric(
                    measurements, field, comp_monitor, seed, n_boot)

            # Métrica derivada: overhead de preparação/pós-processamento
            # (total - tflite) para selector/predictor (§17 tabela detalhada).
            if name in ("selector", "predictor"):
                lat_times = np.asarray(
                    [m.get("timestamp_monotonic_ns", np.nan)
                     for m in measurements], dtype=np.float64)
                prep = np.asarray([
                    (m["total_stage_ns"] - m["tflite_total_ns"])
                    for m in measurements
                    if isinstance(m.get("total_stage_ns"), (int, float))
                    and isinstance(m.get("tflite_total_ns"), (int, float))
                ], dtype=np.float64)
                metric_summaries["prep_post_ns"] = summarize_values(
                    prep, comp_monitor, lat_times, seed, n_boot,
                    field="prep_post_ns")

            # razões (selector/predictor): total vs tflite vs invoke
            ratios = self._compute_ratios(metric_summaries, name)

            comp_summary = {
                "run": info["run_result"],
                "metrics": metric_summaries,
                "ratios": ratios,
                "n_unique_images": self.pool_stats.get(name, {}).get(
                    "total_unique",
                    self.pool_stats.get(name, {}).get("suited_unique")),
                "reuse_counts": info.get("reuse_counts"),
                "pool_stats": self.pool_stats.get(name),
                "model_info": self.model_infos.get(name),
                "correctness": bench.correctness,
            }
            out["components"][name] = comp_summary
            self.summaries[name] = comp_summary
            flat_rows.extend(self._flat_rows(name, metric_summaries,
                                             comp_summary))
        self._summary_nested = out
        self._flat_rows = flat_rows

    def _metric_fields(self, name, bench):
        if name == "selector":
            return ["total_stage_ns", "tflite_total_ns", "invoke_ns"]
        if name == "predictor":
            return ["total_stage_ns", "tflite_total_ns", "invoke_ns"]
        if name == "enhancer":
            base = ["total_stage_ns"]
            if bench.decompose:
                base += ["noise_removal_ns", "adjust_scale_ns",
                         "replicate_ns", "resize_pad_ns"]
            return base
        if name == "aggregation":
            return ["aggregation_ns"]
        return []

    def _compute_ratios(self, metric_summaries, name):
        ratios = {}
        if name in ("selector", "predictor"):
            total = metric_summaries.get("total_stage_ns", {}).get(
                "describe", {}).get("mean_ns")
            tflite = metric_summaries.get("tflite_total_ns", {}).get(
                "describe", {}).get("mean_ns")
            invoke = metric_summaries.get("invoke_ns", {}).get(
                "describe", {}).get("mean_ns")
            if total and tflite:
                ratios["proportion_tflite_of_total"] = tflite / total
                ratios["proportion_prep_of_total"] = 1.0 - (tflite / total)
                ratios["prep_overhead_mean_ms"] = (total - tflite) / 1e6
                ratios["ratio_prep_over_tflite"] = (total - tflite) / tflite
            if invoke and tflite:
                ratios["proportion_invoke_of_tflite"] = invoke / tflite
        return ratios

    def _flat_rows(self, name, metric_summaries, comp_summary):
        rows = []
        label, fmt = STAGE_LABEL.get(name, (name, ""))
        model_info = self.model_infos.get(name, {})
        for field, ms in metric_summaries.items():
            d = ms.get("describe", {})
            if not d or d.get("n_valid", 0) == 0:
                continue
            rows.append({
                "component": name,
                "stage": label,
                "format": fmt,
                "metric": field,
                "model_size_mb": model_info.get("size_mb", ""),
                "num_threads": model_info.get("num_threads", ""),
                "n_valid": d.get("n_valid"),
                "mean_ms": round(d.get("mean_ms"), 6),
                "std_ms": round(d.get("std_ms"), 6),
                "median_ms": round(d.get("median_ms"), 6),
                "p95_ms": round(d.get("p95_ms"), 6),
                "p99_ms": round(d.get("p99_ms"), 6),
                "min_ms": round(d.get("min_ms"), 6),
                "max_ms": round(d.get("max_ms"), 6),
                "cv": d.get("cv"),
                "throughput_ops_per_s_median": ms.get(
                    "throughput", {}).get("ops_per_sec_by_median"),
            })
        return rows

    # ------------------------------------------------------------------ #
    def write_all(self):
        self._write_metadata()
        self._write_summary_json()
        self._write_summary_csv()
        self._write_report_md()
        self._write_article_tables()
        if self.plots:
            try:
                self._write_plots()
            except Exception as e:  # noqa: BLE001
                with open(os.path.join(self.dir, "plots_ERROR.txt"), "w") as f:
                    f.write(f"Geração de gráficos falhou: {e!r}\n")

    def _write_metadata(self):
        meta = {
            "benchmark": "component_microbenchmark",
            "generated_utc": datetime.utcnow().isoformat() + "Z",
            "config": self.config,
            "environment_before": self.env_before,
            "environment_after": self.env_after,
            "models": self.model_infos,
            "pools": self.pool_stats,
            "bootstrap": self.bootstrap_cfg,
            "notes": {
                "delegates": ("Selector/predictor: delegate CPU padrão do "
                              "TFLite, não configurado explicitamente; "
                              "enhancer/aggregation: não aplicável."),
                "timed_regions": "ver BENCHMARK_COMPONENTS.md e report.md",
                "warmup_disclaimer": ("warm-ups NÃO são misturados com as "
                                      "medições válidas"),
            },
        }
        with open(os.path.join(self.dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def _write_summary_json(self):
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(self._summary_nested, f, indent=2, default=str)

    def _write_summary_csv(self):
        fields = ["component", "stage", "format", "metric", "model_size_mb",
                  "num_threads", "n_valid", "mean_ms", "std_ms", "median_ms",
                  "p95_ms", "p99_ms", "min_ms", "max_ms", "cv",
                  "throughput_ops_per_s_median"]
        write_csv(os.path.join(self.dir, "summary.csv"),
                  self._flat_rows, fields)

    # ------------------------------------------------------------------ #
    def _write_report_md(self):
        lines = []
        lines.append("# Benchmark de componentes do pipeline Edge AI\n")
        lines.append(f"- Gerado em (UTC): {datetime.utcnow().isoformat()}Z")
        lines.append(f"- Semente: {self.config.get('seed')} | "
                     f"warm-up: {self.config.get('warmup')} | "
                     f"iterações: {self.config.get('iterations')}")
        host = self.env_before.get("hostname", "?")
        model = self.env_before.get("device_model", "?")
        lines.append(f"- Host: {host} | dispositivo: {model}")
        temp0 = self.env_before.get("temperature_celsius")
        temp1 = self.env_after.get("temperature_celsius")
        lines.append(f"- Temperatura inicial/final: {temp0} °C / {temp1} °C")
        thr = self.env_after.get("throttled", {})
        if (isinstance(thr, dict)
                and thr.get("throttled_occurred") is True):
            lines.append("\n> ⚠ **Throttling ocorreu** desde o boot "
                         f"(raw={thr.get('raw')}). Ver system_monitor.csv.")
        lines.append("\n## 1. Metodologia\n")
        lines.append("- Cada componente é medido de forma **sequencial e "
                     "isolada**, após warm-up próprio.")
        lines.append("- Reuso das **funções/interpretadores reais** do "
                     "pipeline (FrameSelection, DataEnhance, PredictWeight), "
                     "com num_threads=2. Selector/predictor usam o delegate "
                     "CPU padrão do TFLite quando disponível; enhancement e "
                     "agregação não usam TFLite.")
        lines.append("- `time.perf_counter_ns()` em todas as durações. "
                     "Nenhuma escrita em disco/print dentro das regiões "
                     "cronometradas.")
        lines.append("- Decomposição selector/predictor: mesma sequência do "
                     "`predict()` real instrumentada (verificação de "
                     "equivalência em cada componente abaixo).")
        lines.append("\n## 2. Regiões cronometradas (resumo)\n")
        lines.append("| Componente | total_stage inclui | tflite_total |"
                     " invoke |")
        lines.append("|---|---|---|---|")
        lines.append("| selector | to_single_channel + preprocess_fn + "
                     "set_tensor + invoke + get_tensor + classe | set+invoke+get"
                     " | invoke |")
        lines.append("| enhancer | `DataEnhance.run(img)` (4 transforms) |"
                     " — | — |")
        lines.append("| predictor | asarray + set_tensor + invoke + get_tensor"
                     " + copy + float | set+invoke+get | invoke |")
        lines.append("| aggregation | `float(np.mean(weights))` | — | — |")
        lines.append("\n## 3. Resultados por componente\n")
        for name, s in self.summaries.items():
            lines.append(f"### {name}\n")
            rr = s.get("run", {})
            lines.append(f"- solicitadas: {rr.get('requested')} | "
                         f"concluídas: {rr.get('completed')} | "
                         f"válidas: {rr.get('valid')} | "
                         f"falhas: {rr.get('failures')} | "
                         f"warm-up: {rr.get('warmup')}")
            c = s.get("correctness") or {}
            if c.get("checked"):
                lines.append(f"- corretude (instrumentado vs real): "
                             f"diff={c.get('abs_diff')} "
                             f"match={c.get('matches_within_1e-6') or c.get('matches_within_1e-4')}")
            ratios = s.get("ratios") or {}
            if ratios:
                lines.append("- razões: " + ", ".join(
                    f"{k}={_fmt(v)}" for k, v in ratios.items()))
            lines.append("\n| métrica | n | média ms | mediana ms | "
                         "dp ms | p95 ms | p99 ms | CV | outliers IQR |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for field, ms in s.get("metrics", {}).items():
                d = ms.get("describe", {})
                if not d or d.get("n_valid", 0) == 0:
                    continue
                lines.append(
                    f"| {field} | {d.get('n_valid')} | "
                    f"{_fmt(d.get('mean_ms'))} | {_fmt(d.get('median_ms'))} | "
                    f"{_fmt(d.get('std_ms'))} | {_fmt(d.get('p95_ms'))} | "
                    f"{_fmt(d.get('p99_ms'))} | {_fmt(d.get('cv'))} | "
                    f"{d.get('n_outliers_iqr')} |")
            # inclinação temporal
            ts = next(iter(s.get("metrics", {}).values()), {}).get(
                "temporal_slope", {})
            if ts and ts.get("slope_ns_per_iter") is not None:
                lines.append(f"- inclinação temporal: "
                             f"{_fmt(ts.get('slope_ns_per_iter'))} ns/iter "
                             f"(r={_fmt(ts.get('rvalue'))})")
            lines.append("")
        with open(os.path.join(self.dir, "report.md"), "w") as f:
            f.write("\n".join(lines))

    # ------------------------------------------------------------------ #
    def _write_article_tables(self):
        # Tabela principal (uma linha por componente; métrica principal).
        main_src = []
        for r in self._flat_rows:
            if not ((r["component"] in ("selector", "predictor")
                     and r["metric"] == "total_stage_ns")
                    or (r["component"] == "enhancer"
                        and r["metric"] == "total_stage_ns")
                    or (r["component"] == "aggregation"
                        and r["metric"] == "aggregation_ns")):
                continue
            main_src.append(r)
        # CSV principal com mean/std separados (útil p/ Pandas/R).
        main_csv_fields = ["stage", "format", "model_size_mb", "num_threads",
                           "n_valid", "mean_ms", "std_ms", "median_ms",
                           "p95_ms", "p99_ms", "min_ms", "max_ms", "cv",
                           "throughput_ops_per_s_median"]
        write_csv(os.path.join(self.dir, "article_table.csv"),
                  main_src, main_csv_fields)
        # MD/LaTeX com coluna combinada "Mean ± SD".
        main_display = [dict(r, mean_pm_sd=f"{_fmt(r.get('mean_ms'))} ± "
                                       f"{_fmt(r.get('std_ms'))}")
                        for r in main_src]
        main_disp_fields = ["stage", "format", "model_size_mb", "num_threads",
                            "n_valid", "mean_pm_sd", "median_ms", "p95_ms",
                            "p99_ms", "min_ms", "max_ms", "cv",
                            "throughput_ops_per_s_median"]
        self._md_table(os.path.join(self.dir, "article_table.md"),
                       main_display, main_disp_fields,
                       headers=["Stage", "Format", "Size MB", "Threads", "N",
                                "Mean ± SD ms", "Median ms", "P95 ms",
                                "P99 ms", "Min ms", "Max ms", "CV",
                                "Thruput (s⁻¹)"])
        self._latex_table(os.path.join(self.dir, "article_table.tex"),
                          main_display, main_disp_fields,
                          headers=["Stage", "Format", "Size MB", "Threads",
                                   "N", "Mean$\\pm$SD ms", "Median ms", "P95",
                                   "P99", "Min", "Max", "CV", "Thruput"])

        # Tabela detalhada: todas as métricas
        # (total/tflite/invoke/prep_post/per-transform/aggregation).
        det_fields = ["component", "metric", "mean_ms", "std_ms", "median_ms",
                      "p95_ms", "p99_ms", "n_valid"]
        write_csv(os.path.join(self.dir, "article_table_detailed.csv"),
                  self._flat_rows, det_fields)
        self._md_table(os.path.join(self.dir, "article_table_detailed.md"),
                       self._flat_rows, det_fields,
                       headers=["Component", "Metric", "Mean ms", "SD ms",
                                "Median ms", "P95 ms", "P99 ms", "N"])
        self._latex_table(os.path.join(self.dir, "article_table_detailed.tex"),
                          self._flat_rows, det_fields,
                          headers=["Component", "Metric", "Mean ms", "SD ms",
                                   "Median ms", "P95", "P99", "N"])

    @staticmethod
    def _md_table(path, rows, fields, headers):
        with open(path, "w") as f:
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
            for r in rows:
                cells = [_fmt(r.get(fld, "")) for fld in fields]
                f.write("| " + " | ".join(cells) + " |\n")

    @staticmethod
    def _latex_table(path, rows, fields, headers):
        col = "l" + "r" * (len(headers) - 1)
        with open(path, "w") as f:
            f.write("\\begin{tabular}{" + col + "}\n\\toprule\n")
            f.write(" & ".join(headers) + " \\\\\n\\midrule\n")
            for r in rows:
                cells = [_fmt(r.get(fld, "")).replace("_", "\\_")
                         for fld in fields]
                f.write(" & ".join(cells) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")

    # ------------------------------------------------------------------ #
    def _write_plots(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plots_dir = os.path.join(self.dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        for name, s in self.summaries.items():
            # localizamos as medições via summarize já feitas; mas para plotar
            # precisamos dos arrays — lemos do CSV de medições.
            csv_path = os.path.join(self.dir, f"{name}_measurements.csv")
            if not os.path.exists(csv_path):
                continue
            data = _read_measurements_csv(csv_path)
            if not data:
                continue
            self._plot_component(name, data, plots_dir, plt)
        plt.close("all")

    def _plot_component(self, name, data, plots_dir, plt):
        # escolhe a métrica principal
        if name == "aggregation":
            metric = "aggregation_ns"
        else:
            metric = "total_stage_ns"
        vals_ms = [d[metric] / 1e6 for d in data if metric in d]
        if not vals_ms:
            return
        arr = np.asarray(vals_ms)

        def _save(fig, base):
            fig.savefig(os.path.join(plots_dir, f"{name}_{base}.png"),
                        dpi=120, bbox_inches="tight")
            fig.savefig(os.path.join(plots_dir, f"{name}_{base}.pdf"),
                        bbox_inches="tight")
            plt.close(fig)

        # histograma
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(arr, bins=40, color="#3b82f6", edgecolor="white")
        ax.set_xlabel(f"{metric} (ms)"); ax.set_ylabel("frequência")
        ax.set_title(f"{name} — histograma de latência")
        _save(fig, "histogram")

        # latência ao longo das iterações + mediana móvel
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(arr, lw=0.6, alpha=0.6, label="latência")
        if arr.size >= 21:
            k = 21
            moving = np.convolve(arr, np.ones(k) / k, mode="same")
            ax.plot(moving, color="#ef4444", lw=1.5,
                    label=f"mediana/média móvel (k={k})")
        ax.set_xlabel("iteração"); ax.set_ylabel(f"{metric} (ms)")
        ax.set_title(f"{name} — latência ao longo das iterações")
        ax.legend()
        _save(fig, "latency_over_iterations")

        # comparação de métricas (média/mediana/p95/p99) se houver mais de uma
        metric_cols = [c for c in
                       (["total_stage_ns", "tflite_total_ns", "invoke_ns"]
                        if name in ("selector", "predictor") else
                        [metric]) if c in data[0]]
        if len(metric_cols) > 1:
            stats = []
            for c in metric_cols:
                v = np.asarray([d[c] / 1e6 for d in data if c in d])
                stats.append([v.mean(), np.median(v),
                              np.percentile(v, 95), np.percentile(v, 99)])
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.arange(len(metric_cols))
            ax.bar(x - 0.3, [s[0] for s in stats], width=0.2, label="mean")
            ax.bar(x - 0.1, [s[1] for s in stats], width=0.2, label="median")
            ax.bar(x + 0.1, [s[2] for s in stats], width=0.2, label="p95")
            ax.bar(x + 0.3, [s[3] for s in stats], width=0.2, label="p99")
            ax.set_xticks(x); ax.set_xticklabels(metric_cols, rotation=20)
            ax.set_ylabel("ms"); ax.set_title(f"{name} — mean/median/p95/p99")
            ax.legend()
            _save(fig, "metrics_comparison")

        # latência por bloco de 100
        blocks = st.block_stats(np.asarray(
            [d[metric] for d in data if metric in d]), 100)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([b["block_index"] for b in blocks],
               [b["mean_ms"] for b in blocks], color="#10b981")
        ax.set_xlabel("bloco de 100 iterações"); ax.set_ylabel("média (ms)")
        ax.set_title(f"{name} — latência média por bloco")
        _save(fig, "per_block")


def _read_measurements_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            conv = {}
            for k, v in r.items():
                if v is None or v == "":
                    continue
                try:
                    if k.endswith("_ns") or k in ("iteration", "score",
                                                  "prediction", "result",
                                                  "num_predictions",
                                                  "predicted_argmax",
                                                  "timestamp_monotonic_ns",
                                                  "output_min", "output_max",
                                                  "output_mean", "output_std"):
                        conv[k] = float(v)
                    else:
                        conv[k] = v
                except ValueError:
                    conv[k] = v
            rows.append(conv)
    return rows


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        if np.isfinite(v):
            return f"{v:.4g}"
        return str(v)
    return str(v)
