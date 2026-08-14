#!/usr/bin/env python3
"""Build the reproducible native-timestamp five-run analysis notebook."""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).with_name("baseline_5runs_analysis_corrected.ipynb")
cells = []

def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))

def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))

md(r"""
# Análise completa das cinco execuções baseline — timestamps nativos

Este notebook caracteriza o comportamento do pipeline Edge AI sob a cadência temporal
original do dataset (`native_timestamps=True`, `fps=None`, `mode=single`, `engine=thread`).
Ele é uma análise de execução, não uma nova rodada experimental: os arquivos originais
são somente lidos e os resultados derivados são exportados para `outputs_baseline/`.

## Perguntas científicas

1. Qual é a distribuição da carga temporal, computacional, térmica e energética?
2. A frequência média de entrada esconde rajadas de imagens adequadas?
3. Essas rajadas ultrapassam temporariamente a capacidade de serviço isolada ou integrada do preditor?
4. Há evidência direta ou apenas indireta de trabalho residual após a captura?
5. As cinco repetições da mesma trace são reprodutíveis?

### Cuidados de interpretação

* Timestamps relativos `t=...ms` do log são usados para cadência e rajadas.
* Timestamps absolutos do JSON são usados para conclusão, predições e energia quando disponíveis.
* A ordem do log é apenas evidência ordinal; ela não é convertida artificialmente em tempo.
* `rho_isolated` e `rho_integrated_service` são comparações de carga, não utilização real nem tamanho de fila.
* O notebook separa observação diretamente medida, interpretação e hipótese.
""")

md("## 1. Configuração e bibliotecas\n\nEsta seção define caminhos, parâmetros de rajada, janelas móveis e sementes. A descoberta das cinco runs é automática.")
code(r'''
import json, math, os, platform, re, socket, sys, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display, Markdown

warnings.filterwarnings("ignore")
np.random.seed(42)

HERE = Path.cwd()
if not (HERE / "power_runs").exists() and (HERE.parent / "power_runs").exists():
    REPO_ROOT = HERE.parent
else:
    REPO_ROOT = HERE
POWER_ROOT = REPO_ROOT / "power_runs"
NATIVE_GLOB = "battery_mas-single_native_*"
OUTPUT_ROOT = REPO_ROOT / "outputs_baseline"
FIG_DIR_DEFAULT = OUTPUT_ROOT / "figures"
WARMUP_REFERENCE_MS = 136.4
PREDICTOR_CAPACITY_S = 1.0 / (WARMUP_REFERENCE_MS / 1000.0)
BURST_DEFINITIONS = [
    {"name": "1.5x_median", "kind": "median_multiplier", "value": 1.5},
    {"name": "2.0x_median", "kind": "median_multiplier", "value": 2.0},
    {"name": "250ms_absolute", "kind": "absolute_ms", "value": 250.0},
]
WINDOWS_MS = [250, 500, 1000, 2000]
BUSY_GAP_THRESHOLDS_S = [0.25, 0.50]
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 42

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": .3,
    "font.size": 10, "pdf.fonttype": 42,
})
def save_figure(fig, name):
    FIG_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR_DEFAULT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR_DEFAULT / f"{name}.pdf", bbox_inches="tight")
    try: fig.savefig(FIG_DIR_DEFAULT / f"{name}.svg", bbox_inches="tight")
    except Exception: pass
print("REPO_ROOT:", REPO_ROOT.resolve())
print("OUTPUT_ROOT:", OUTPUT_ROOT.resolve())
''')

md("## 2. Descoberta e validação dos arquivos\n\nA análise não depende de um nome fixo de subpasta. Cada diretório `battery_mas-single_native_*` é uma execução; sua subpasta de pipeline contém `metrics.json`, logs e telemetria. Arquivos opcionais ausentes produzem avisos, não uma falha global.")
code(r'''
def find_metrics_dir(run_dir: Path):
    candidates = [p.parent for p in run_dir.rglob("metrics.json")]
    return sorted(candidates)[0] if candidates else None

def discover_native_runs(root=POWER_ROOT):
    rows = []
    def add_run(p, power_parent=None):
        md = find_metrics_dir(p)
        if not md: return
        log_path = p / "pipeline.log" if (p / "pipeline.log").exists() else md / "debug.log"
        power_path = p / "power.csv" if (p / "power.csv").exists() else (power_parent or p) / "power.csv"
        tc66_path = p / "tc66.log" if (p / "tc66.log").exists() else (power_parent or p) / "tc66.log"
        rows.append({
            "run_dir": p, "run_name": p.name, "metrics_dir": md,
            "metrics": md / "metrics.json",
            "pipeline_log": log_path,
            "cpu": md / "cpu.csv", "mem": md / "mem.csv", "temp": md / "temp.csv",
            "power": power_path, "tc66": tc66_path,
        })
    for p in sorted(root.glob(NATIVE_GLOB)):
        if not p.is_dir(): continue
        if (p / "metrics.json").exists():
            add_run(p)
        else:
            for child in sorted(p.iterdir()):
                if child.is_dir() and re.search(r"_r\d+$", child.name): add_run(child, p)
    rows = sorted(rows, key=lambda x: x["run_name"])
    if len(rows) != 5:
        print(f"[AVISO] Foram descobertas {len(rows)} runs nativas; esperadas 5.")
    return rows

def inventory_table(runs):
    records = []
    for r in runs:
        rec = {"run": r["run_name"]}
        for key in ["metrics", "pipeline_log", "cpu", "mem", "temp", "power", "tc66"]:
            p = r[key]
            rec[key] = bool(p and p.exists())
            rec[key + "_path"] = str(p.relative_to(REPO_ROOT)) if p and p.exists() else ""
        records.append(rec)
    return pd.DataFrame(records)

runs = discover_native_runs()
file_inventory = inventory_table(runs)
display(file_inventory)
''')

md("## 3. Funções de parsing\n\nAs funções seguintes mantêm identificadores de animais como strings e distinguem timestamps absolutos do JSON, tempos relativos de captura e ordem textual do log.")
code(r'''
ISO_RE = re.compile(r"\[T(?P<tag>[01])\].*?(?P<ts>\d{4}-\d\d-\d\dT[^ ]+)")
CAP_RE = re.compile(r"\[CAPTURE\].*?animal=(?P<animal>\S+).*?idx=(?P<idx>\d+).*?t=(?P<t>[-+]?\d+(?:\.\d+)?)ms.*?label=(?P<label>\S+)")
SEL_RE = re.compile(r"\[SELECT\]\s+frame_id=(?P<frame>\S+)\s+animal=(?P<animal>\S+)\s+label=(?P<label>\S+).*?->\s*(?P<result>SUITABLE|DISCARDED).*?\(p=(?P<prob>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)")
SUMMARY_RE = re.compile(r"\[SELECT-SUMMARY\]\s+animal=(?P<animal>\S+)\s+total=(?P<total>\d+)\s+discarded=(?P<discarded>\d+)\s+forwarded=(?P<forwarded>\d+)")
FINAL_RE = re.compile(r"\[FINAL\]\s+Animal\s+(?P<animal>\S+):\s+n_suitable=(?P<n_suitable>\d+)")

def as_dt(value):
    return pd.to_datetime(value, errors="coerce")

def residual_delay_s(last_capture, last_prediction_end):
    if pd.isna(last_capture) or pd.isna(last_prediction_end): return np.nan
    return max(0.0, (last_prediction_end - last_capture).total_seconds())

def parse_log(path):
    captures, selections, events, t0, t1 = [], [], [], None, None
    if not path or not path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), t0, t1
    for order, line in enumerate(path.read_text(errors="replace").splitlines()):
        m = ISO_RE.search(line)
        if m:
            dt = as_dt(m.group("ts"))
            if m.group("tag") == "0" and pd.notna(dt): t0 = dt.tz_localize(None) if getattr(dt, "tzinfo", None) else dt
            if m.group("tag") == "1" and pd.notna(dt): t1 = dt.tz_localize(None) if getattr(dt, "tzinfo", None) else dt
        m = CAP_RE.search(line)
        if m:
            captures.append({"log_order": order, "animal_id": str(m.group("animal")),
                             "frame_index": int(m.group("idx")), "capture_t_ms": float(m.group("t")),
                             "label": m.group("label")})
        m = SEL_RE.search(line)
        if m:
            selections.append({"log_order": order, "frame_id": m.group("frame"),
                               "animal_id": str(m.group("animal")), "selection_label": m.group("label"),
                               "selection": m.group("result"), "selection_probability": float(m.group("prob"))})
        sm = SUMMARY_RE.search(line)
        fm = FINAL_RE.search(line)
        event_type = next((kind for kind, marker in [("passage_complete", "[PASSAGE-COMPLETE]"), ("final", "[FINAL]"), ("start", "[START]"), ("select_summary", "[SELECT-SUMMARY]")] if marker in line), None)
        if event_type:
            event = {"log_order": order, "line": line, "event_type": event_type, "timestamp": as_dt(m.group("ts")) if (m := ISO_RE.search(line)) else pd.NaT}
            if sm:
                event.update({"animal_id": str(sm.group("animal")), "summary_total": int(sm.group("total")), "summary_discarded": int(sm.group("discarded")), "summary_forwarded": int(sm.group("forwarded"))})
            if fm:
                event.update({"animal_id": str(fm.group("animal")), "final_n_suitable": int(fm.group("n_suitable"))})
            start_match = re.search(r"\[START\]\s+Animal\s+(\S+)", line)
            if start_match: event["animal_id"] = str(start_match.group(1))
            events.append(event)
    return pd.DataFrame(captures), pd.DataFrame(selections), pd.DataFrame(events), t0, t1

def parse_metrics(path, run_name):
    raw = json.loads(path.read_text())
    animals, inferences = [], []
    for aid, a in raw.get("animals", {}).items():
        aid = str(aid)
        first, last, final = as_dt(a.get("first_image_capture_time")), as_dt(a.get("last_image_capture_time")), as_dt(a.get("weight_prediction_final"))
        row = {"run": run_name, "animal_id": aid, "first_capture": first, "last_capture": last,
               "result_final": final, "total_images": pd.to_numeric(a.get("total_of_images"), errors="coerce"),
               "suitable_images": pd.to_numeric(a.get("suitable_images"), errors="coerce")}
        row["passage_s"] = (last-first).total_seconds() if pd.notna(first) and pd.notna(last) else np.nan
        row["post_capture_s"] = (final-last).total_seconds() if pd.notna(final) and pd.notna(last) else np.nan
        row["passage_to_final_s"] = (final-first).total_seconds() if pd.notna(final) and pd.notna(first) else np.nan
        preds = a.get("imgs", {}) or {}
        last_pred_end = pd.NaT
        for frame_id, im in preds.items():
            s, e = as_dt(im.get("weight_prediction_start")), as_dt(im.get("weight_prediction_final"))
            if pd.notna(e) and (pd.isna(last_pred_end) or e > last_pred_end): last_pred_end = e
            inferences.append({"run": run_name, "animal_id": aid, "frame_id": str(frame_id), "prediction_start": s, "prediction_end": e,
                               "inference_s": (e-s).total_seconds() if pd.notna(s) and pd.notna(e) else np.nan})
        row["last_prediction_end"] = last_pred_end
        row["pred_residual_s"] = residual_delay_s(last, last_pred_end)
        pivot = max(last, last_pred_end) if pd.notna(last) and pd.notna(last_pred_end) else pd.NaT
        row["final_overhead_s"] = (final-pivot).total_seconds() if pd.notna(final) and pd.notna(pivot) else np.nan
        animals.append(row)
    return raw, pd.DataFrame(animals), pd.DataFrame(inferences)

def stats_dict(values):
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0: return {k: np.nan for k in ["n","mean","median","std","cv","min","max","p01","p05","p10","p25","p50","p75","p90","p95","p99","iqr","mad","skew","kurtosis","sem","ci95_low","ci95_high"]}
    mean, med, sd = float(x.mean()), float(np.median(x)), float(x.std(ddof=1)) if len(x)>1 else 0.0
    sem = sd / np.sqrt(len(x)) if len(x)>1 else np.nan
    crit = stats.t.ppf(.975, len(x)-1) if len(x)>1 else np.nan
    return {"n": len(x), "mean": mean, "median": med, "std": sd, "cv": sd/mean if mean else np.nan,
            "min": float(x.min()), "max": float(x.max()), "p01": np.percentile(x,1), "p05": np.percentile(x,5),
            "p10": np.percentile(x,10), "p25": np.percentile(x,25), "p50": np.percentile(x,50), "p75": np.percentile(x,75),
            "p90": np.percentile(x,90), "p95": np.percentile(x,95), "p99": np.percentile(x,99), "iqr": np.percentile(x,75)-np.percentile(x,25),
            "mad": float(np.median(np.abs(x-med))), "skew": float(stats.skew(x)) if len(x)>2 else np.nan,
            "kurtosis": float(stats.kurtosis(x)) if len(x)>3 else np.nan, "sem": sem,
            "ci95_low": mean-crit*sem if np.isfinite(sem) else np.nan, "ci95_high": mean+crit*sem if np.isfinite(sem) else np.nan}

def bootstrap_ci(values, statistic=np.median, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(x) < 2: return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    samples = rng.choice(x, size=(reps, len(x)), replace=True)
    vals = np.apply_along_axis(statistic, 1, samples)
    return tuple(np.percentile(vals, [2.5, 97.5]))
''')

md("## 4. Carregamento das cinco runs\n\nO carregamento cria tabelas em diferentes unidades experimentais: run, animal, captura, frame adequado e inferência. A análise por animal não trata as 920 observações repetidas como independentes entre runs.")
code(r'''
run_data, animal_parts, inference_parts, capture_parts, selection_parts = [], [], [], [], []
for r in runs:
    if not r["metrics"] or not r["metrics"].exists():
        continue
    raw, animals, inf = parse_metrics(r["metrics"], r["run_name"])
    caps, sels, events, t0_log, t1_log = parse_log(r["pipeline_log"])
    animals["log_t0"] = t0_log; animals["log_t1"] = t1_log
    if not caps.empty:
        caps["run"] = r["run_name"]
        capture_parts.append(caps)
    if not sels.empty:
        sels["run"] = r["run_name"]
        selection_parts.append(sels)
    inf["run"] = r["run_name"]
    inference_parts.append(inf)
    animal_parts.append(animals)
    run_data.append({"run": r["run_name"], "run_dir": r["run_dir"], "power_file": r["power"], "raw": raw, "metrics_dir": r["metrics_dir"],
                     "animals": animals, "inferences": inf, "captures": caps, "selections": sels, "events": events,
                     "t0_log": t0_log, "t1_log": t1_log})

df_animals = pd.concat(animal_parts, ignore_index=True) if animal_parts else pd.DataFrame()
df_inferences = pd.concat(inference_parts, ignore_index=True) if inference_parts else pd.DataFrame()
df_captures = pd.concat(capture_parts, ignore_index=True) if capture_parts else pd.DataFrame()
df_selections = pd.concat(selection_parts, ignore_index=True) if selection_parts else pd.DataFrame()
print(f"runs={len(run_data)} animals={len(df_animals):,} inferences={len(df_inferences):,} captures={len(df_captures):,} selections={len(df_selections):,}")
''')

md("## 5. Integridade e consistência\n\nA tabela reporta esperado, observado e status. Inconsistências não são corrigidas silenciosamente. A ausência de uma telemetria opcional reduz somente as análises dependentes dela.")
code(r'''
integrity_rows, inconsistency_rows = [], []
for d in run_data:
    a = d["animals"]
    run = d["run"]
    caps = d["captures"]; sels = d["selections"]; raw = d["raw"]
    vals = {
        "animals": (184, len(a)), "frames": (13741, int(a["total_images"].sum())),
        "suitable": (1670, int(a["suitable_images"].sum())), "predictions": (int(a["suitable_images"].sum()), len(d["inferences"])),
        "final_results": (len(a), int(a["result_final"].notna().sum())),
    }
    checks = {}
    for name, (expected, observed) in vals.items():
        ok = expected == observed
        checks[name] = observed
        if not ok: inconsistency_rows.append({"run": run, "metric": name, "expected": expected, "observed": observed, "decision": "preserve_observed_and_warn"})
    first_ok = bool(a["first_capture"].notna().all()) if len(a) else False
    last_ok = bool(a["last_capture"].notna().all()) if len(a) else False
    chrono_ok = bool((a["passage_s"].dropna() >= 0).all()) if len(a) else False
    checks.update({"resources": all((d["metrics_dir"] / x).exists() for x in ["cpu.csv","mem.csv","temp.csv"]),
                   "energy": (d["run_dir"] / "power.csv").exists(), "first_capture": first_ok,
                   "last_capture": last_ok, "chronology": chrono_ok})
    status = "OK" if all([v[0] == v[1] for v in vals.values()]) and all(checks.values()) else "WARN"
    integrity_rows.append({"run": run, **{k:v[1] for k,v in vals.items()}, **checks, "status": status})
df_integrity = pd.DataFrame(integrity_rows)
df_inconsistencies = pd.DataFrame(inconsistency_rows)
display(df_integrity)
display(df_inconsistencies if not df_inconsistencies.empty else pd.DataFrame({"status": ["Nenhuma inconsistência de contagem"]}))
''')

md("## 6. Caracterização geral do baseline\n\nA taxa principal por animal usa `(N-1)/(última captura - primeira captura)`, pois há N-1 intervalos. A versão `N/duração` é mantida apenas para comparação. FPS global, por animal e animais/minuto respondem perguntas diferentes.")
code(r'''
def safe_rate(n, duration):
    return (n / duration) if pd.notna(duration) and duration > 0 else np.nan

for d in run_data:
    a = d["animals"]
    a["fps_effective"] = a.apply(lambda x: safe_rate(x.total_images-1, x.passage_s), axis=1)
    a["fps_effective_alt"] = a.apply(lambda x: safe_rate(x.total_images, x.passage_s), axis=1)
    a["suitable_fraction"] = a["suitable_images"] / a["total_images"].replace(0, np.nan)
df_animals["fps_effective"] = df_animals.apply(lambda x: safe_rate(x.total_images-1, x.passage_s), axis=1)
df_animals["fps_effective_alt"] = df_animals.apply(lambda x: safe_rate(x.total_images, x.passage_s), axis=1)
df_animals["suitable_fraction"] = df_animals["suitable_images"] / df_animals["total_images"].replace(0, np.nan)

general_rows = []
for d in run_data:
    a = d["animals"]
    duration = (d["t1_log"] - d["t0_log"]).total_seconds() if pd.notna(d["t0_log"]) and pd.notna(d["t1_log"]) else np.nan
    first, last = a["first_capture"].min(), a["last_capture"].max()
    capture_span = (last-first).total_seconds() if pd.notna(first) and pd.notna(last) else np.nan
    general_rows.append({"run": d["run"], "duration_s": duration, "animals": len(a), "frames": a.total_images.sum(),
                         "suitable": a.suitable_images.sum(), "suitable_fraction": a.suitable_images.sum()/a.total_images.sum(),
                         "fps_global": safe_rate(a.total_images.sum()-len(a), capture_span),
                         "fps_global_alt": safe_rate(a.total_images.sum(), capture_span),
                         "animals_per_min": safe_rate(len(a), duration/60 if pd.notna(duration) else np.nan),
                         "passage_median_s": a.passage_s.median(), "fps_animal_median": a.fps_effective.median(),
                         "fps_animal_mean": a.fps_effective.mean(), "fps_animal_p05": a.fps_effective.quantile(.05),
                         "fps_animal_p25": a.fps_effective.quantile(.25), "fps_animal_p75": a.fps_effective.quantile(.75),
                         "fps_animal_p95": a.fps_effective.quantile(.95), "fps_animal_cv": a.fps_effective.std()/a.fps_effective.mean()})
df_general = pd.DataFrame(general_rows)
display(df_general.round(4))

fig, ax = plt.subplots(figsize=(8,4.5))
for d in run_data:
    x = d["animals"]["fps_effective"].dropna()
    ax.hist(x, bins=25, alpha=.35, label=d["run"][-2:])
ax.set(xlabel="FPS efetivo por animal (intervalos, frame/s)", ylabel="Animais", title="Distribuição do FPS efetivo por animal")
ax.legend(title="run", ncol=5)
save_figure(fig, "01_fps_efetivo_por_animal")
plt.show()
''')

md("## 7. Recursos computacionais\n\nCPU, RAM e temperatura são resumidas por run e também mantidas em séries temporais. Temperatura elevada, isoladamente, não confirma throttling; somente flags/frequência registradas podem sustentar essa conclusão.")
code(r'''
resource_frames, resource_summary = [], []
def load_resources(d):
    md = d["metrics_dir"]
    parts = []
    if (md / "cpu.csv").exists():
        cpu = pd.read_csv(md / "cpu.csv", parse_dates=["timestamp"]); cores = [c for c in cpu if c.startswith("cpu_core_")]
        if cores: cpu["cpu_mean"] = cpu[cores].mean(axis=1); cpu["cpu_peak"] = cpu[cores].max(axis=1)
        parts.append(cpu)
    if (md / "mem.csv").exists(): parts.append(pd.read_csv(md / "mem.csv", parse_dates=["timestamp"]))
    if (md / "temp.csv").exists(): parts.append(pd.read_csv(md / "temp.csv", parse_dates=["timestamp"]))
    if not parts: return pd.DataFrame()
    base = parts[0].sort_values("timestamp")
    for other in parts[1:]:
        base = pd.merge_asof(base, other.sort_values("timestamp"), on="timestamp", direction="nearest", tolerance=pd.Timedelta("2s"))
    base["run"] = d["run"]; base["t_rel_s"] = (base.timestamp - base.timestamp.min()).dt.total_seconds()
    return base

for d in run_data:
    r = load_resources(d)
    if r.empty: continue
    resource_frames.append(r)
    cores = [c for c in r if c.startswith("cpu_core_")]
    resource_summary.append({"run": d["run"], "cpu_mean": r.cpu_mean.mean() if "cpu_mean" in r else np.nan,
                             "cpu_median": r.cpu_mean.median() if "cpu_mean" in r else np.nan,
                             "cpu_p95": r.cpu_mean.quantile(.95) if "cpu_mean" in r else np.nan,
                             "cpu_max": r.cpu_mean.max() if "cpu_mean" in r else np.nan,
                             "cpu_above_50_pct": (r.cpu_mean>50).mean() if "cpu_mean" in r else np.nan,
                             "cpu_above_75_pct": (r.cpu_mean>75).mean() if "cpu_mean" in r else np.nan,
                             "cpu_peak_100_pct": (r.cpu_peak>=99.9).mean() if "cpu_peak" in r else np.nan,
                             "ram_mean_pct": r.percent.mean() if "percent" in r else np.nan,
                             "ram_p95_pct": r.percent.quantile(.95) if "percent" in r else np.nan,
                             "ram_max_pct": r.percent.max() if "percent" in r else np.nan,
                             "ram_growth_pct": (r.percent.iloc[-1]-r.percent.iloc[0]) if "percent" in r and len(r)>1 else np.nan,
                             "temp_initial_c": r.temperature.iloc[0] if "temperature" in r and len(r) else np.nan,
                             "temp_mean_c": r.temperature.mean() if "temperature" in r else np.nan,
                             "temp_median_c": r.temperature.median() if "temperature" in r else np.nan,
                             "temp_p95_c": r.temperature.quantile(.95) if "temperature" in r else np.nan,
                             "temp_max_c": r.temperature.max() if "temperature" in r else np.nan,
                             "temp_delta_c": r.temperature.iloc[-1]-r.temperature.iloc[0] if "temperature" in r and len(r)>1 else np.nan,
                             **{f"temp_above_{t}c_pct": (r.temperature>t).mean() if "temperature" in r else np.nan for t in [70,75,80,85]}})
    resource_frames[-1]["cpu_frequency_mhz"] = np.nan
df_resources = pd.concat(resource_frames, ignore_index=True) if resource_frames else pd.DataFrame()
df_resource_summary = pd.DataFrame(resource_summary)
display(df_resource_summary.round(3))

if not df_resources.empty:
    fig, axes = plt.subplots(3, 1, figsize=(10,8), sharex=True)
    for run, g in df_resources.groupby("run"):
        axes[0].plot(g.t_rel_s/60, g.cpu_mean, alpha=.7, label=run[-2:])
        if "percent" in g: axes[1].plot(g.t_rel_s/60, g.percent, alpha=.7)
        if "temperature" in g: axes[2].plot(g.t_rel_s/60, g.temperature, alpha=.7)
    axes[0].set_ylabel("CPU média (%)"); axes[1].set_ylabel("RAM (%)"); axes[2].set_ylabel("Temp. (°C)"); axes[2].set_xlabel("Tempo desde início da telemetria (min)")
    axes[0].legend(title="run", ncol=5); fig.suptitle("Recursos ao longo de uma execução completa"); plt.tight_layout(); save_figure(fig, "02_recursos_ao_longo_do_tempo"); plt.show()
''')

md("## 8. Energia e potência\n\nA energia é integrada pela regra trapezoidal usando os timestamps reais de `power.csv`; não se assume amostragem exatamente de 1 Hz. A cobertura e os intervalos temporais são reportados.")
code(r'''
def energy_from_csv(path, start=None, end=None):
    if not path.exists(): return {"power_mean_w": np.nan, "power_median_w": np.nan, "power_p95_w": np.nan, "power_max_w": np.nan, "energy_j": np.nan, "energy_wh": np.nan, "coverage_s": np.nan, "n_power": 0}
    p = pd.read_csv(path, skipinitialspace=True)
    p["timestamp"] = pd.to_datetime(p.get("Datetime"), errors="coerce")
    p["power_w"] = pd.to_numeric(p.get("Power[W]"), errors="coerce")
    p = p.dropna(subset=["timestamp","power_w"]).sort_values("timestamp").drop_duplicates("timestamp")
    if start is not None: p = p[p.timestamp >= start]
    if end is not None: p = p[p.timestamp <= end]
    if len(p) < 2: return {"power_mean_w": p.power_w.mean() if len(p) else np.nan, "power_median_w": p.power_w.median() if len(p) else np.nan, "power_p95_w": p.power_w.quantile(.95) if len(p) else np.nan, "power_max_w": p.power_w.max() if len(p) else np.nan, "energy_j": np.nan, "energy_wh": np.nan, "coverage_s": np.nan, "n_power": len(p)}
    ts = (p.timestamp - p.timestamp.iloc[0]).dt.total_seconds().to_numpy()
    energy = float(np.trapezoid(p.power_w.to_numpy(), ts) if hasattr(np, "trapezoid") else np.trapz(p.power_w.to_numpy(), ts))
    return {"power_mean_w": p.power_w.mean(), "power_median_w": p.power_w.median(), "power_p95_w": p.power_w.quantile(.95), "power_p99_w": p.power_w.quantile(.99), "power_max_w": p.power_w.max(), "energy_j": energy, "energy_wh": energy/3600, "coverage_s": ts[-1], "n_power": len(p), "median_dt_s": np.median(np.diff(ts)), "max_dt_s": np.max(np.diff(ts))}

energy_rows = []
for d in run_data:
    a = d["animals"]; start, end = d["t0_log"], d["t1_log"]
    e = energy_from_csv(d["power_file"], start, end)
    duration = (end-start).total_seconds() if pd.notna(start) and pd.notna(end) else np.nan
    e.update({"run": d["run"], "duration_s": duration, "coverage_fraction": e.get("coverage_s", np.nan)/duration if duration else np.nan,
              "energy_per_animal_j": e.get("energy_j",np.nan)/len(a), "energy_per_frame_j": e.get("energy_j",np.nan)/a.total_images.sum(),
              "energy_per_suitable_j": e.get("energy_j",np.nan)/a.suitable_images.sum()})
    energy_rows.append(e)
df_energy = pd.DataFrame(energy_rows)
display(df_energy.round(4))
''')

md("## 9. Latência e responsividade\n\nAs métricas abaixo preservam os nomes definidos no protocolo: latência de conclusão pós-captura, latência da passagem ao resultado, atraso residual do preditor e overhead final.")
code(r'''
latency_cols = ["post_capture_s", "passage_to_final_s", "pred_residual_s", "final_overhead_s"]
latency_summary = []
for c in latency_cols:
    s = stats_dict(df_animals[c]); s["median_ci95_low"], s["median_ci95_high"] = bootstrap_ci(df_animals[c]); s["metric"] = c; latency_summary.append(s)
df_latency_summary = pd.DataFrame(latency_summary)
display(df_latency_summary[["metric","n","mean","median","std","p90","p95","p99","min","max","cv","ci95_low","ci95_high"]].round(5))
inf_stats = stats_dict(df_inferences["inference_s"] if not df_inferences.empty else [])
print("Tempo de inferência integrado (s):", {k: round(v,6) if isinstance(v,float) else v for k,v in inf_stats.items()})
if not df_animals.empty:
    print("Animais com residual > 0.1/0.5/1/2 s:", {s: int((df_animals.pred_residual_s>s).sum()) for s in [.1,.5,1,2]})
fig, ax = plt.subplots(figsize=(8,4.5))
data = [df_animals.loc[df_animals.run==r["run"], "post_capture_s"].dropna() for r in run_data]
ax.boxplot(data, tick_labels=[r["run"][-2:] for r in run_data], showfliers=True)
ax.set(ylabel="Segundos", title="Latência de conclusão pós-captura por run"); save_figure(fig, "03_latencia_pos_captura"); plt.show()
fig, ax = plt.subplots(figsize=(8,4.5))
if not df_inferences.empty: ax.hist(df_inferences.inference_s.dropna()*1000, bins=40, color="#2f6690", alpha=.85)
ax.set(xlabel="Tempo de inferência integrada (ms)", ylabel="Predições", title="Distribuição do tempo individual de predição"); save_figure(fig, "04_tempo_inferencia_integrada"); plt.show()
''')

md("## 10. Cadência temporal das capturas e jitter\n\nOs intervalos são derivados exclusivamente de `t=...ms` no log. O jitter absoluto é `std(Δt)` e o relativo é seu coeficiente de variação.")
code(r'''
cadence_rows, interval_rows = [], []
for d in run_data:
    caps = d["captures"]
    if caps.empty: continue
    for aid, g in caps.groupby("animal_id"):
        t = np.sort(g.capture_t_ms.dropna().to_numpy()); dt = np.diff(t)
        dt = dt[dt >= 0]
        if len(dt):
            interval_rows.extend({"run": d["run"], "animal_id": aid, "interval_ms": x} for x in dt)
            cadence_rows.append({"run": d["run"], "animal_id": aid, "n_captures_log": len(t), "mean_dt_ms": dt.mean(), "median_dt_ms": np.median(dt), "std_dt_ms": dt.std(ddof=1) if len(dt)>1 else 0, "jitter_cv": dt.std(ddof=1)/dt.mean() if len(dt)>1 and dt.mean() else np.nan, "p05_dt_ms": np.percentile(dt,5), "p95_dt_ms": np.percentile(dt,95), "min_dt_ms": dt.min(), "max_dt_ms": dt.max(), "fps_local_mean": np.mean(1000/dt[dt>0]) if np.any(dt>0) else np.nan})
df_cadence = pd.DataFrame(cadence_rows); df_intervals = pd.DataFrame(interval_rows)
if not df_cadence.empty: display(df_cadence.groupby("run").agg({"mean_dt_ms":"mean","median_dt_ms":"median","std_dt_ms":"mean","jitter_cv":"mean","fps_local_mean":"median"}).round(4))
if not df_intervals.empty:
    fig, ax = plt.subplots(figsize=(8,4.5)); ax.hist(df_intervals.interval_ms, bins=50, color="#6a994e"); ax.set(xlabel="Intervalo entre capturas (ms)", ylabel="Ocorrências", title="Distribuição dos intervalos de captura"); save_figure(fig, "05_intervalos_captura_jitter"); plt.show()
''')

md("## 11. Pareamento ordinal, validação dos rótulos e rajadas\n\nA associação é feita primeiro entre todas as capturas e todas as decisões de seleção, por animal e na ordem do log. Só depois o dataframe pareado é filtrado para `SUITABLE`. Como não há timestamp absoluto por decisão de seleção, os tempos das decisões são os tempos relativos da captura ordinal correspondente. O campo `forwarded` de `[SELECT-SUMMARY]` é um contador cumulativo da run; a auditoria calcula também seu delta por animal antes de compará-lo com `suitable_paired`.")
code(r'''
def pair_capture_selection(captures, selections, run):
    """Pareia a sequência completa; nunca filtra SUITABLE antes do pareamento."""
    columns = ["run", "animal_id", "ordinal", "capture_idx", "capture_t_ms", "capture_label",
               "capture_log_order", "frame_id", "selection_label", "selection_result",
               "selection_probability", "selection_log_order", "association"]
    if captures is None: captures = pd.DataFrame()
    if selections is None: selections = pd.DataFrame()
    ids = sorted(set(captures.get("animal_id", pd.Series(dtype=str)).astype(str)) |
                 set(selections.get("animal_id", pd.Series(dtype=str)).astype(str)))
    rows = []
    for aid in ids:
        cg = captures[captures.animal_id.astype(str) == aid].sort_values("log_order").reset_index(drop=True)
        sg = selections[selections.animal_id.astype(str) == aid].sort_values("log_order").reset_index(drop=True)
        n = min(len(cg), len(sg))
        for i in range(n):
            rows.append({"run": run, "animal_id": aid, "ordinal": i,
                         "capture_idx": cg.loc[i, "frame_index"], "capture_t_ms": cg.loc[i, "capture_t_ms"],
                         "capture_label": cg.loc[i, "label"], "capture_log_order": cg.loc[i, "log_order"],
                         "frame_id": sg.loc[i, "frame_id"], "selection_label": sg.loc[i, "selection_label"],
                         "selection_result": sg.loc[i, "selection"], "selection_probability": sg.loc[i, "selection_probability"],
                         "selection_log_order": sg.loc[i, "log_order"], "association": "ordinal_log_order"})
    return pd.DataFrame(rows, columns=columns)

def event_value(events, aid, key):
    if events is None or events.empty or key not in events or "animal_id" not in events: return np.nan
    x = events[(events["animal_id"].astype(str) == str(aid)) & events[key].notna()]
    return x.iloc[-1][key] if not x.empty else np.nan

def summary_forwarded_deltas(events):
    """Interpreta forwarded do SELECT-SUMMARY como contador cumulativo da run."""
    if events is None or events.empty or "summary_forwarded" not in events: return {}
    x = events[events.event_type.eq("select_summary")].dropna(subset=["summary_forwarded"]).sort_values("log_order").copy()
    if x.empty: return {}
    x["summary_forwarded_delta"] = x.summary_forwarded.diff().fillna(x.summary_forwarded)
    if (x.summary_forwarded_delta < 0).any():
        print("[WARNING][SELECT-SUMMARY] Contador forwarded não é monotônico; deltas negativos foram preservados.")
    return {str(row.animal_id): row.summary_forwarded_delta for _, row in x.iterrows()}

paired_parts, audit_rows = [], []
pairing_warning_count = 0
pairing_warning_examples = []
for d in run_data:
    paired = pair_capture_selection(d["captures"], d["selections"], d["run"])
    paired_parts.append(paired)
    forwarded_deltas = summary_forwarded_deltas(d["events"])
    a = d["animals"]
    ids = sorted(set(a.animal_id.astype(str)) |
                 set(d["captures"].get("animal_id", pd.Series(dtype=str)).astype(str)) |
                 set(d["selections"].get("animal_id", pd.Series(dtype=str)).astype(str)))
    for aid in ids:
        cg = d["captures"][d["captures"].animal_id.astype(str) == aid].sort_values("log_order")
        sg = d["selections"][d["selections"].animal_id.astype(str) == aid].sort_values("log_order")
        pg = paired[paired.animal_id == aid].sort_values("ordinal")
        metrics_n = a.loc[a.animal_id.astype(str) == aid, "suitable_images"]
        metrics_n = float(metrics_n.iloc[0]) if len(metrics_n) else np.nan
        summary_forwarded_cumulative = event_value(d["events"], aid, "summary_forwarded")
        summary_forwarded_delta = forwarded_deltas.get(str(aid), np.nan)
        final_n = event_value(d["events"], aid, "final_n_suitable")
        if pd.isna(final_n): final_n = metrics_n
        checks = {
            "counts_match": len(cg) == len(sg),
            "paired_is_min": len(pg) == min(len(cg), len(sg)),
            "capture_idx_increasing": bool(pg.capture_idx.diff().dropna().gt(0).all()) if len(pg) > 1 else True,
            "capture_time_increasing": bool(pg.capture_t_ms.diff().dropna().ge(0).all()) if len(pg) > 1 else True,
            "no_duplicate_capture_idx": not pg.capture_idx.duplicated().any(),
            "no_duplicate_frame_id": not pg.frame_id.duplicated().any(),
            "summary_forwarded_delta_matches_suitable": pd.isna(summary_forwarded_delta) or int(summary_forwarded_delta) == int((pg.selection_result == "SUITABLE").sum()),
            "final_matches_suitable": pd.isna(final_n) or int((pg.selection_result == "SUITABLE").sum()) == int(final_n),
            "summary_total_matches_captures": pd.isna(event_value(d["events"], aid, "summary_total")) or int(event_value(d["events"], aid, "summary_total")) == len(cg),
        }
        status = "OK" if all(checks.values()) else "WARN"
        if status == "WARN":
            pairing_warning_count += 1
            if len(pairing_warning_examples) < 20: pairing_warning_examples.append({"run":d["run"],"animal":aid,"checks":checks,"summary_forwarded_cumulative":summary_forwarded_cumulative,"summary_forwarded_delta":summary_forwarded_delta,"suitable_paired":int((pg.selection_result == "SUITABLE").sum()),"final_n_suitable":final_n})
        audit_rows.append({"run": d["run"], "animal": aid, "captures": len(cg), "selections": len(sg), "paired": len(pg),
                           "suitable_paired": int((pg.selection_result == "SUITABLE").sum()) if len(pg) else 0,
                           "summary_forwarded": summary_forwarded_cumulative, "summary_forwarded_cumulative": summary_forwarded_cumulative, "summary_forwarded_delta": summary_forwarded_delta, "final_n_suitable": final_n, "status": status,
                           **checks})

df_paired = pd.concat([x for x in paired_parts if not x.empty], ignore_index=True) if any(not x.empty for x in paired_parts) else pd.DataFrame()
df_pairing_audit = pd.DataFrame(audit_rows)
df_suitable = df_paired[df_paired.selection_result == "SUITABLE"].copy() if not df_paired.empty else pd.DataFrame()
display(df_pairing_audit[["run", "animal", "captures", "selections", "paired", "suitable_paired", "summary_forwarded_cumulative", "summary_forwarded_delta", "final_n_suitable", "status"]])
if not df_pairing_audit.empty and (df_pairing_audit.status != "OK").any():
    print(f"[WARNING] {pairing_warning_count} linhas de auditoria têm divergências; exemplos: {pairing_warning_examples[:3]}")

def selector_confusion(paired):
    if paired.empty: return pd.DataFrame(), {}
    x = paired.copy()
    x["label_class"] = np.where(x.capture_label.astype(str).str.lower().eq("suited"), "suited", "not_suited")
    x["decision_class"] = np.where(x.selection_result.eq("SUITABLE"), "forwarded", "discarded")
    table = pd.crosstab(x.label_class, x.decision_class).reindex(index=["suited", "not_suited"], columns=["forwarded", "discarded"], fill_value=0)
    tp, fn = int(table.loc["suited", "forwarded"]), int(table.loc["suited", "discarded"])
    fp, tn = int(table.loc["not_suited", "forwarded"]), int(table.loc["not_suited", "discarded"])
    denom = tp + tn + fp + fn
    metrics = {"tp_suited_forwarded": tp, "fn_suited_discarded": fn, "fp_notsuited_forwarded": fp, "tn_notsuited_discarded": tn,
               "accuracy": (tp + tn) / denom if denom else np.nan, "precision_suitable": tp / (tp + fp) if tp + fp else np.nan,
               "recall_suitable": tp / (tp + fn) if tp + fn else np.nan, "f1_suitable": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan}
    return table.reset_index().rename(columns={"label_class": "capture_label_class"}), metrics

df_confusion, confusion_metrics = selector_confusion(df_paired)
if not df_confusion.empty:
    display(df_confusion); display(pd.DataFrame([confusion_metrics]))
    fig, ax = plt.subplots(figsize=(6, 4)); im = ax.imshow(df_confusion.set_index("capture_label_class").to_numpy(), cmap="Blues")
    ax.set_xticks(range(2), ["SUITABLE", "DISCARDED"]); ax.set_yticks(range(2), ["suited", "not suited"])
    ax.set(xlabel="Decisão do seletor", ylabel="Rótulo da captura", title="Matriz de confusão ordinal do seletor")
    for i in range(2):
        for j in range(2): ax.text(j, i, int(im.get_array()[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, shrink=.8); save_figure(fig, "10_matriz_confusao_seletor"); plt.show()

def burst_table(g, definition):
    x = np.sort(pd.to_numeric(g.capture_t_ms, errors="coerce").dropna().to_numpy(dtype=float))
    if len(x) == 0: return []
    median_interval = np.median(np.diff(x)) if len(x) > 1 else np.nan
    if definition["kind"] == "median_multiplier":
        threshold = definition["value"] * median_interval if np.isfinite(median_interval) and median_interval > 0 else np.inf
    else:
        threshold = definition["value"]
    groups = np.zeros(len(x), dtype=int)
    if len(x) > 1: groups[1:] = np.cumsum(np.diff(x) > threshold)
    out = []
    for bid, y in pd.DataFrame({"t": x, "b": groups}).groupby("b"):
        t = y.t.to_numpy(); duration = (t[-1] - t[0]) / 1000.0
        intervals = np.diff(t) / 1000.0
        out.append({"burst_id": int(bid), "threshold_name": definition["name"], "threshold_ms": threshold,
                    "median_capture_interval_ms": median_interval, "n_suitable": len(t), "start_ms": t[0], "end_ms": t[-1],
                    "duration_s": duration, "mean_interval_s": np.mean(intervals) if len(intervals) else np.nan,
                    "lambda_burst": (len(t) - 1) / duration if duration > 0 and len(t) > 1 else np.nan,
                    "lambda_burst_median_interval": 1 / np.median(intervals) if len(t) >= 3 and np.median(intervals) > 0 else np.nan})
    return out

burst_rows = []
for (run, aid), g in df_suitable.groupby(["run", "animal_id"]):
    for definition in BURST_DEFINITIONS:
        burst_rows.extend({"run": run, "animal_id": aid, **b} for b in burst_table(g, definition))
df_bursts = pd.DataFrame(burst_rows)
if not df_bursts.empty:
    burst_summary = df_bursts.groupby(["run", "animal_id", "threshold_name"]).agg(
        n_bursts=("burst_id", "nunique"), max_burst_n=("n_suitable", "max"), mean_burst_duration_s=("duration_s", "mean"),
        max_burst_duration_s=("duration_s", "max"), mean_lambda_burst=("lambda_burst", "mean"), max_lambda_burst=("lambda_burst", "max"),
        mean_interval_s=("mean_interval_s", "mean")).reset_index()
    cap_by_run = df_inferences.groupby("run").inference_s.mean().rdiv(1).rename("integrated_capacity_s") if not df_inferences.empty else pd.Series(dtype=float)
    burst_summary["integrated_capacity_s"] = burst_summary.run.map(cap_by_run)
    burst_summary["above_isolated_capacity"] = burst_summary.max_lambda_burst > PREDICTOR_CAPACITY_S
    burst_summary["above_integrated_capacity"] = burst_summary.max_lambda_burst > burst_summary.integrated_capacity_s
    df_burst_summary = burst_summary
    display(df_burst_summary.groupby("threshold_name").agg({"n_bursts": "mean", "mean_burst_duration_s": "mean", "mean_lambda_burst": "mean", "max_lambda_burst": "max", "above_isolated_capacity": "mean", "above_integrated_capacity": "mean"}).round(4))
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.hist(df_bursts.duration_s.dropna(), bins=30, color="#bc6c25"); ax.set(xlabel="Duração da rajada (s)", ylabel="Rajadas", title="Distribuição da duração das rajadas corrigidas"); save_figure(fig, "06_duracao_rajadas_adequadas"); plt.show()
else:
    df_burst_summary = pd.DataFrame()
''')

md("## 12. Janelas móveis, capacidade e testes de sanidade\n\nA taxa de janela é `max_t N_suitable(t,t+w)/w`, com busca por `searchsorted` em tempo linear amortizado. São mantidas duas referências de capacidade: o microbenchmark isolado e o tempo de serviço observado dentro do pipeline.\n\n**Interpretação:** `lambda_burst` é uma taxa média dentro de uma rajada específica, calculada como `(N-1)/(t_último-t_primeiro)`; ela descreve a cadência sustentada daquele agrupamento temporal. Já `peak_rate_1000ms` é o maior número de adequados que cabe em qualquer janela de 1 s, dividido por 1 s; ele captura concentração local, pode atravessar duas rajadas e não exige que todos os frames pertençam à mesma rajada. Portanto, as duas métricas respondem perguntas diferentes e não devem ser comparadas como se fossem equivalentes.")
code(r'''
def peak_window_rate(times_ms, window_ms):
    x = np.sort(pd.to_numeric(pd.Series(times_ms), errors="coerce").dropna().to_numpy(dtype=float))
    if len(x) == 0: return np.nan
    right = np.searchsorted(x, x + window_ms, side="right")
    return float((right - np.arange(len(x))).max()) / (window_ms / 1000.0)

def capacity_from_service_time(mean_service_s):
    return 1.0 / mean_service_s if pd.notna(mean_service_s) and mean_service_s > 0 else np.nan

def busy_blocks(inferences, gap_threshold_s):
    if inferences is None or inferences.empty: return pd.DataFrame()
    x = inferences.dropna(subset=["prediction_start", "prediction_end"]).sort_values("prediction_start").reset_index(drop=True)
    if x.empty: return pd.DataFrame()
    block = np.zeros(len(x), dtype=int)
    if len(x) > 1:
        gaps = (x.prediction_start.iloc[1:].reset_index(drop=True) - x.prediction_end.iloc[:-1].reset_index(drop=True)).dt.total_seconds()
        block[1:] = np.cumsum(gaps.to_numpy() > gap_threshold_s)
    rows = []
    for bid, g in x.groupby(block):
        span = (g.prediction_end.iloc[-1] - g.prediction_start.iloc[0]).total_seconds()
        rows.append({"busy_gap_threshold_s": gap_threshold_s, "busy_block_id": int(bid), "n_predictions": len(g),
                     "start": g.prediction_start.iloc[0], "end": g.prediction_end.iloc[-1], "span_s": span,
                     "busy_block_completion_rate": len(g) / span if span > 0 else np.nan})
    return pd.DataFrame(rows)

# Testes determinísticos: estes asserts protegem as definições corrigidas.
_tc = pd.DataFrame({"log_order": [1, 3, 5], "animal_id": ["a"] * 3, "frame_index": [1, 2, 3], "capture_t_ms": [0., 100., 200.], "label": ["not", "suited", "suited"]})
_ts = pd.DataFrame({"log_order": [2, 4, 6], "animal_id": ["a"] * 3, "frame_id": ["f1", "f2", "f3"], "selection_label": ["not", "suited", "suited"], "selection": ["DISCARDED", "SUITABLE", "SUITABLE"], "selection_probability": [.1, .9, .9]})
_tp = pair_capture_selection(_tc, _ts, "synthetic")
assert list(_tp.capture_idx) == [1, 2, 3]
assert list(_tp[_tp.selection_result == "SUITABLE"].capture_idx) == [2, 3]
assert peak_window_rate([0., 100., 200., 900., 1100.], 1000.) == 4.0
_bt = pd.DataFrame({"capture_t_ms": [0., 100., 200., 900.]})
assert burst_table(_bt, BURST_DEFINITIONS[0])[0]["n_suitable"] == 3
assert pd.isna(burst_table(pd.DataFrame({"capture_t_ms": [0.]}), BURST_DEFINITIONS[0])[0]["lambda_burst"])
assert pd.isna(residual_delay_s(pd.Timestamp("2020-01-01"), pd.NaT))
assert abs(capacity_from_service_time(.1364) - PREDICTOR_CAPACITY_S) < 1e-9
_empty_pair = pair_capture_selection(_tc, _ts.iloc[0:0], "synthetic")
assert len(_empty_pair) == 0
_one = busy_blocks(pd.DataFrame({"prediction_start": [pd.Timestamp("2020-01-01")], "prediction_end": [pd.Timestamp("2020-01-01 00:00:00.1")]}), .25)
assert len(_one) == 1 and _one.iloc[0].n_predictions == 1
print("Testes internos de pareamento, rajadas, janelas, atraso, capacidade, bloco ocupado e casos vazios: OK")

integrated_service_by_run = df_inferences.groupby("run").inference_s.mean().apply(capacity_from_service_time) if not df_inferences.empty else pd.Series(dtype=float)
window_rows=[]
for _, ar in df_pairing_audit.iterrows():
    run, aid = ar["run"], ar["animal"]
    g = df_suitable[(df_suitable.run == run) & (df_suitable.animal_id == aid)] if not df_suitable.empty else pd.DataFrame()
    times = g.capture_t_ms.to_numpy() if not g.empty else []
    row={"run":run,"animal_id":aid,"n_suitable":len(times),"integrated_service_capacity_s":integrated_service_by_run.get(run, np.nan)}
    for w in WINDOWS_MS:
        rate=peak_window_rate(times,w); row[f"peak_rate_{w}ms"]=rate
        row[f"rho_isolated_{w}ms"]=rate/PREDICTOR_CAPACITY_S if np.isfinite(rate) else np.nan
        cap = row["integrated_service_capacity_s"]
        row[f"rho_integrated_service_{w}ms"]=rate/cap if np.isfinite(rate) and pd.notna(cap) and cap > 0 else np.nan
    window_rows.append(row)
df_windows=pd.DataFrame(window_rows)
if not df_windows.empty:
    print("Capacidade isolada de referência:",round(PREDICTOR_CAPACITY_S,3),"inferências/s; tempo de serviço do pipeline por run:",integrated_service_by_run.round(3).to_dict())
    display(df_windows.groupby("run")[[f"peak_rate_{w}ms" for w in WINDOWS_MS]].agg(["median",lambda x:x.quantile(.9),lambda x:x.quantile(.95),"max"]).round(3))
    df_window_sensitivity = df_windows.groupby("run").agg(**{f"peak_rate_{w}ms_mean":(f"peak_rate_{w}ms","mean") for w in WINDOWS_MS}).reset_index()
    for cap_name, cap_col in [("isolated", "rho_isolated_1000ms"), ("integrated_service", "rho_integrated_service_1000ms")]:
        share = float((df_windows[cap_col] > 1).fillna(False).mean())
        print(f"Fração de animais acima da capacidade {cap_name} em janela de 1 s (sem excluir animais sem adequados): {share:.4f}")
    fig,ax=plt.subplots(figsize=(8,4.5)); ax.scatter(df_windows.peak_rate_1000ms,df_windows.rho_isolated_1000ms,alpha=.35); ax.axhline(1,color="red",ls="--",label="rho_isolated=1"); ax.set(xlabel="Pico local de adequados em janela de 1 s (frames/s)",ylabel="rho_isolated",title="Carga local versus capacidade isolada"); ax.legend(); save_figure(fig, "07_carga_local_versus_capacidade"); plt.show()
    fig,ax=plt.subplots(figsize=(8,4.5)); ax.scatter(df_windows.peak_rate_1000ms,df_windows.rho_integrated_service_1000ms,alpha=.35,color="#2f6690"); ax.axhline(1,color="red",ls="--",label="rho_integrated_service=1"); ax.set(xlabel="Pico local de adequados em janela de 1 s (frames/s)",ylabel="rho_integrated_service",title="Carga local versus capacidade integrada"); ax.legend(); save_figure(fig, "07b_carga_local_versus_capacidade_integrada"); plt.show()

busy_block_rows=[]
for d in run_data:
    for threshold in BUSY_GAP_THRESHOLDS_S:
        blocks = busy_blocks(d["inferences"], threshold)
        if not blocks.empty:
            blocks["run"] = d["run"]; busy_block_rows.extend(blocks.to_dict("records"))
df_busy_blocks = pd.DataFrame(busy_block_rows)
df_busy_summary = df_busy_blocks.groupby(["run", "busy_gap_threshold_s"]).agg(median_busy_block_rate=("busy_block_completion_rate", "median"), p95_busy_block_rate=("busy_block_completion_rate", lambda x: x.quantile(.95)), max_busy_block_rate=("busy_block_completion_rate", "max"), n_busy_blocks=("busy_block_id", "count")).reset_index() if not df_busy_blocks.empty else pd.DataFrame()
if not df_busy_summary.empty: display(df_busy_summary)
''')

md("## 13. Trabalho residual e sobreposição\n\nOs atrasos diretamente medidos vêm dos timestamps absolutos do JSON. A carga pendente reconstruída só seria calculada se houvesse timestamps absolutos compatíveis para capturas e conclusões; neste conjunto, a associação captura–seleção é apenas ordinal. A ordem do log não é convertida em sobreposição temporal.")
code(r'''
residual_rows=[]
for d in run_data:
    a=d["animals"].copy();
    residual_rows.append({"run":d["run"],"animals":len(a),"last_capture_measured":int(a.last_capture.notna().sum()),"last_prediction_end_measured":int(a.last_prediction_end.notna().sum()),"final_result_measured":int(a.result_final.notna().sum()),"with_pred_after_last_capture":int((a.pred_residual_s>0).sum()),"share_pred_after_last_capture":float((a.pred_residual_s>0).mean()),"predictions_after_last_capture":int((a.pred_residual_s>0).sum()),"with_residual_gt_100ms":int((a.pred_residual_s>.1).sum()),"max_pred_residual_s":a.pred_residual_s.max(),"mean_pred_residual_s":a.pred_residual_s.mean(),"final_overhead_mean_s":a.final_overhead_s.mean(),"pending_suitable_workload_status":"not_reconstructed_without_absolute_capture_timestamps"})
df_residual=pd.DataFrame(residual_rows); display(df_residual.round(5))
ordinal_rows=[]
for d in run_data:
    ev=d["events"]
    if ev.empty: continue
    finals=ev[ev.event_type.eq("final")].log_order.to_numpy(); completes=ev[ev.event_type.eq("passage_complete")].log_order.to_numpy(); starts=ev[ev.event_type.eq("start")].log_order.to_numpy()
    absolute_available = bool(ev[ev.event_type.isin(["final", "start"])].timestamp.notna().all()) if len(ev[ev.event_type.isin(["final", "start"])]) else False
    ordinal_rows.append({"run":d["run"],"final_events":len(finals),"passage_complete_events":len(completes),"start_events":len(starts),"final_after_passage_complete_log_order_evidence":int(sum(any(finals>x) for x in completes)),"absolute_timestamps_for_start_final":absolute_available,"quantitative_overlap_computed":False,"interpretation":"ordinal evidence only"})
df_ordinal=pd.DataFrame(ordinal_rows); display(df_ordinal)
''')

md("## 14. Taxas de conclusão e capacidade\n\nA taxa de conclusão ao longo da execução usa a janela entre a primeira e a última conclusão de inferência e não é chamada de throughput ocupado. A capacidade isolada vem do microbenchmark de 136,4 ms; a capacidade integrada usa o tempo médio de serviço observado no pipeline (não é throughput end-to-end).")
code(r'''
throughput_rows=[]
for d in run_data:
    a=d["animals"]; inf=d["inferences"]
    first,last=a.first_capture.min(),a.last_capture.max(); span=(last-first).total_seconds() if pd.notna(first) and pd.notna(last) else np.nan
    if not inf.empty and inf.prediction_start.notna().any() and inf.prediction_end.notna().any():
        ps,pe=inf.prediction_start.min(),inf.prediction_end.max(); completion_span=(pe-ps).total_seconds()
    else: completion_span=np.nan
    dur=(d["t1_log"]-d["t0_log"]).total_seconds() if pd.notna(d["t0_log"]) and pd.notna(d["t1_log"]) else np.nan
    mean_service = inf.inference_s.mean() if not inf.empty else np.nan
    integrated_capacity = capacity_from_service_time(mean_service)
    throughput_rows.append({"run":d["run"],"input_fps":safe_rate(a.total_images.sum()-len(a),span),"run_level_prediction_completion_rate":safe_rate(len(inf),completion_span),"isolated_capacity_s":PREDICTOR_CAPACITY_S,"isolated_service_time_s":WARMUP_REFERENCE_MS/1000.0,"integrated_service_time_s":mean_service,"integrated_service_capacity_s":integrated_capacity,"e2e_suitable_completion_rate":safe_rate(a.suitable_images.sum(),dur),"animals_per_s":safe_rate(len(a),dur)})
df_throughput=pd.DataFrame(throughput_rows); display(df_throughput.round(4))
df_capacity = df_throughput[["run", "isolated_service_time_s", "isolated_capacity_s", "integrated_service_time_s", "integrated_service_capacity_s"]].copy()
display(df_capacity.round(5))
''')

md("## 15. Repetibilidade entre as cinco runs\n\nAs cinco execuções repetem a mesma trace. O resumo entre runs usa cinco observações por métrica; as observações por animal são preservadas para distribuição e análise de medidas repetidas, sem testes de hipótese automáticos desnecessários.")
code(r'''
metric_sources={
    "duration_s":df_general,"fps_global":df_general,"suitable":df_general,"frames":df_general,
    "cpu_mean":df_resource_summary,"cpu_p95":df_resource_summary,"ram_mean_pct":df_resource_summary,
    "temp_mean_c":df_resource_summary,"temp_max_c":df_resource_summary,"energy_j":df_energy,
    "power_mean_w":df_energy,"post_capture_mean_s":df_animals.groupby("run").post_capture_s.mean().reset_index(name="post_capture_mean_s"),
    "inference_mean_s":df_inferences.groupby("run").inference_s.mean().reset_index(name="inference_mean_s") if not df_inferences.empty else pd.DataFrame(),
    "peak_rate_1s":df_windows.groupby("run").peak_rate_1000ms.median().reset_index(name="peak_rate_1s") if not df_windows.empty else pd.DataFrame(),
    "residual_share":df_residual[["run","share_pred_after_last_capture"]].rename(columns={"share_pred_after_last_capture":"residual_share"}),
}
repeat_rows=[]
for metric,table in metric_sources.items():
    if table.empty or metric not in table: continue
    s=stats_dict(table[metric]); repeat_rows.append({"metric":metric,"n_runs":s["n"],"mean":s["mean"],"std_between_runs":s["std"],"cv":s["cv"],"min":s["min"],"max":s["max"],"relative_range":(s["max"]-s["min"])/abs(s["mean"]) if s["mean"] else np.nan,"ci95_low":s["ci95_low"],"ci95_high":s["ci95_high"]})
df_repeatability=pd.DataFrame(repeat_rows); display(df_repeatability.round(5))
''')

md("## 16. Comparação baseline nativo versus 10 FPS fixo\n\nA comparação usa a mesma unidade experimental (run), as mesmas fórmulas e as mesmas unidades. A latência é `post_capture_completion_latency_s` nos dois casos. A tabela não afirma equivalência por similaridade de FPS médio.")
code(r'''
fixed_root=POWER_ROOT / "battery_mas-single_20260708_104924"
fixed_available=fixed_root.exists()
if fixed_available:
    fixed_dirs=sorted([p for p in fixed_root.iterdir() if p.is_dir() and re.search(r"_10fps_r\d+$", p.name)])
    fixed_rows=[]
    for fd in fixed_dirs:
        md=find_metrics_dir(fd)
        if not md: continue
        m=re.search(r"_(\d+)fps",fd.name); fps=int(m.group(1)) if m else np.nan
        raw,aa,ii=parse_metrics(md/"metrics.json",fd.name)
        log=fd/"pipeline.log" if (fd/"pipeline.log").exists() else (md/"debug.log")
        cc,ss,ee,t0,t1=parse_log(log)
        duration=(t1-t0).total_seconds() if pd.notna(t0) and pd.notna(t1) else np.nan
        d_fixed={"run":fd.name,"run_dir":fd,"metrics_dir":md,"power_file":fd/"power.csv","animals":aa,"inferences":ii,"captures":cc,"selections":ss,"events":ee,"t0_log":t0,"t1_log":t1}
        pair_fixed=pair_capture_selection(cc,ss,fd.name)
        sf=pair_fixed[pair_fixed.selection_result.eq("SUITABLE")] if not pair_fixed.empty else pd.DataFrame()
        first,last=aa.first_capture.min(),aa.last_capture.max(); capture_span=(last-first).total_seconds() if pd.notna(first) and pd.notna(last) else np.nan
        rr=load_resources(d_fixed); e=energy_from_csv(fd/"power.csv",t0,t1)
        fixed_peak_by_animal = [peak_window_rate(g.capture_t_ms,1000.) for _, g in sf.groupby("animal_id")] if not sf.empty else []
        peak_local=float(np.nanmean(fixed_peak_by_animal)) if fixed_peak_by_animal else np.nan
        fixed_rows.append({"condition":"10fps_fixed","fps_fixed":fps,"run":fd.name,"duration_s":duration,"effective_fps":safe_rate(aa.total_images.sum()-len(aa),capture_span),"total_frames":aa.total_images.sum(),"suitable_frames":aa.suitable_images.sum(),"cpu_mean":rr.cpu_mean.mean() if not rr.empty and "cpu_mean" in rr else np.nan,"cpu_p95":rr.cpu_mean.quantile(.95) if not rr.empty and "cpu_mean" in rr else np.nan,"ram_p95":rr.percent.quantile(.95) if not rr.empty and "percent" in rr else np.nan,"temperature_mean_c":rr.temperature.mean() if not rr.empty and "temperature" in rr else np.nan,"power_mean_w":e.get("power_mean_w",np.nan),"energy_j":e.get("energy_j",np.nan),"post_capture_completion_latency_s":aa.post_capture_s.mean(),"inference_service_time_s":ii.inference_s.mean() if not ii.empty else np.nan,"peak_local_1s_fps":peak_local,"residual_delay_s":aa.pred_residual_s.mean(),"residual_share":(aa.pred_residual_s>0).mean()})
    df_fixed=pd.DataFrame(fixed_rows)
    if not df_fixed.empty:
        baseline_comparison_rows=[]
        for _,r in df_throughput.iterrows():
            run=r["run"]; a=df_animals[df_animals.run.eq(run)]; rs=df_resource_summary[df_resource_summary.run.eq(run)]; en=df_energy[df_energy.run.eq(run)]; ww=df_windows[df_windows.run.eq(run)]
            native_span=(a.last_capture.max()-a.first_capture.min()).total_seconds() if a.first_capture.notna().any() and a.last_capture.notna().any() else np.nan
            baseline_comparison_rows.append({"condition":"baseline_native","run":run,"effective_fps":df_general.loc[df_general.run.eq(run),"fps_global"].iloc[0] if any(df_general.run.eq(run)) else np.nan,"total_frames":a.total_images.sum(),"suitable_frames":a.suitable_images.sum(),"cpu_mean":rs.cpu_mean.iloc[0] if len(rs) else np.nan,"cpu_p95":rs.cpu_p95.iloc[0] if len(rs) else np.nan,"ram_p95":rs.ram_p95_pct.iloc[0] if len(rs) else np.nan,"temperature_mean_c":rs.temp_mean_c.iloc[0] if len(rs) else np.nan,"power_mean_w":en.power_mean_w.iloc[0] if len(en) else np.nan,"energy_j":en.energy_j.iloc[0] if len(en) else np.nan,"post_capture_completion_latency_s":a.post_capture_s.mean(),"inference_service_time_s":df_inferences[df_inferences.run.eq(run)].inference_s.mean() if not df_inferences.empty else np.nan,"peak_local_1s_fps":ww.peak_rate_1000ms.mean() if len(ww) else np.nan,"residual_delay_s":a.pred_residual_s.mean(),"residual_share":(a.pred_residual_s>0).mean()})
        df_comparison_runs=pd.concat([pd.DataFrame(baseline_comparison_rows),df_fixed],ignore_index=True)
        comp_metrics=["effective_fps","total_frames","suitable_frames","cpu_mean","cpu_p95","ram_p95","temperature_mean_c","power_mean_w","energy_j","post_capture_completion_latency_s","inference_service_time_s","peak_local_1s_fps","residual_delay_s"]
        comp_rows=[]
        bmean=df_comparison_runs[df_comparison_runs.condition.eq("baseline_native")].mean(numeric_only=True); fmean=df_comparison_runs[df_comparison_runs.condition.eq("10fps_fixed")].mean(numeric_only=True)
        for metric in comp_metrics:
            bv, fv=bmean.get(metric,np.nan), fmean.get(metric,np.nan); diff=bv-fv
            comp_rows.append({"metric":metric,"baseline_native":bv,"fixed_10fps":fv,"absolute_difference_baseline_minus_fixed":diff,"relative_difference_baseline_minus_fixed":diff/abs(fv) if pd.notna(fv) and fv != 0 else np.nan})
        df_comparison=pd.DataFrame(comp_rows)
        display(df_comparison.round(5))
    else: df_comparison_runs=pd.DataFrame(); df_comparison=pd.DataFrame()
else:
    df_fixed=pd.DataFrame(); df_comparison_runs=pd.DataFrame(); df_comparison=pd.DataFrame()
    print("Dados fixos não encontrados em",fixed_root,"; seção opcional não executada.")
''')

md("## 16.1. Relação entre rajadas, residual e timelines\n\nAs timelines combinam tempos absolutos do JSON para predições com tempos relativos de captura apenas após alinhamento ao primeiro frame do mesmo animal. O alinhamento é visual e intra-animal; não cria timestamps para seleção/enhancement.")
code(r'''
if not df_windows.empty:
    join_cols=["run","animal_id","pred_residual_s","post_capture_s"]
    corr_df=df_windows.merge(df_animals[join_cols],on=["run","animal_id"],how="left")
    df_corr_animal=corr_df.groupby("animal_id",as_index=False).agg(peak_rate_1000ms=("peak_rate_1000ms","median"),pred_residual_s=("pred_residual_s","median"),post_capture_s=("post_capture_s","median"),n_runs=("run","nunique"))
    fig,ax=plt.subplots(figsize=(8,4.5)); ax.scatter(df_corr_animal.peak_rate_1000ms,df_corr_animal.pred_residual_s,alpha=.55); ax.set(xlabel="Mediana por animal do pico local de adequados em 1 s",ylabel="Mediana por animal do atraso residual (s)",title="Carga local versus atraso residual (agregado por animal)"); save_figure(fig,"08_carga_local_versus_residual"); plt.show()
    if df_corr_animal.peak_rate_1000ms.nunique()>1 and df_corr_animal.pred_residual_s.notna().sum()>2:
        rho,p=stats.spearmanr(df_corr_animal.peak_rate_1000ms,df_corr_animal.pred_residual_s,nan_policy="omit"); print(f"Spearman exploratório agregado por animal: rho={rho:.3f}, p={p:.3g}; não implica causalidade.")
else:
    df_corr_animal=pd.DataFrame()

if run_data:
    candidates=df_animals.dropna(subset=["post_capture_s"]).copy()
    selected=[]
    choices = [("mediana_latencia", (candidates.post_capture_s-candidates.post_capture_s.median()).abs().idxmin()),
               ("maior_latencia", candidates.post_capture_s.idxmax()), ("maior_residual", candidates.pred_residual_s.idxmax())]
    if not df_windows.empty:
        idx_peak = df_windows.peak_rate_1000ms.idxmax()
        choices.append(("maior_pico_local", df_animals[(df_animals.run == df_windows.loc[idx_peak,"run"]) & (df_animals.animal_id == df_windows.loc[idx_peak,"animal_id"])].index[0]))
    if not df_burst_summary.empty:
        multi = df_burst_summary[df_burst_summary.n_bursts > 1]
        if not multi.empty:
            rr = multi.sort_values("n_bursts", ascending=False).iloc[0]
            idx_multi = df_animals[(df_animals.run == rr.run) & (df_animals.animal_id == rr.animal_id)].index[0]
            choices.append(("multiplas_rajadas", idx_multi))
    few = candidates.suitable_images.idxmin()
    choices.append(("poucos_adequados", few))
    for label, idx in choices:
        if pd.notna(idx): selected.append((label,candidates.loc[idx,"run"],candidates.loc[idx,"animal_id"]))
    selected=list(dict.fromkeys(selected))
    fig,axes=plt.subplots(len(selected),1,figsize=(11,3.2*len(selected)),squeeze=False)
    for ax,(label,run,aid) in zip(axes[:,0],selected):
        d=next(x for x in run_data if x["run"]==run); a=d["animals"].set_index("animal_id").loc[str(aid)]; caps=d["captures"]; st=caps[caps.animal_id==str(aid)].capture_t_ms.to_numpy()
        if len(st): ax.vlines(st/1000,0,1,color="#6a994e",alpha=.25,label="capturas")
        if not df_suitable.empty:
            su=df_suitable[(df_suitable.run==run)&(df_suitable.animal_id==str(aid))].capture_t_ms.to_numpy();
            if len(su): ax.scatter(su/1000,np.full(len(su),1.05),color="#bc6c25",s=16,label="adequados (associação ordinal)")
        inf=d["inferences"][d["inferences"].animal_id==str(aid)]
        t0=a.first_capture
        for _,row in inf.dropna(subset=["prediction_start","prediction_end"]).iterrows(): ax.plot([(row.prediction_start-t0).total_seconds(),(row.prediction_end-t0).total_seconds()],[1.35,1.35],color="#2f6690",lw=3)
        ax.axvline((a.last_capture-t0).total_seconds(),color="black",ls="--",label="última captura"); ax.axvline((a.result_final-t0).total_seconds(),color="red",ls=":",label="resultado final")
        ax.set_title(f"{label}: {run} / animal {aid}"); ax.set_yticks([]); ax.set_xlabel("Tempo relativo ao primeiro frame (s)")
    axes[0,0].legend(ncol=4,fontsize=8,loc="upper right"); fig.tight_layout(); save_figure(fig,"09_timelines_animais_representativos"); plt.show()
''')

md("## 17. Síntese científica automática\n\nA síntese abaixo é deliberadamente separada em observações, interpretações e hipóteses. Ela não redige a seção final do artigo e não transforma correlação em causalidade.")
code(r'''
native_mean_fps=df_general.fps_global.mean() if not df_general.empty else np.nan
native_duration=df_general.duration_s.mean() if not df_general.empty else np.nan
native_suitable=df_general.suitable.mean() if not df_general.empty else np.nan
resid_share=df_residual.share_pred_after_last_capture.mean() if not df_residual.empty else np.nan
peak1=df_windows.peak_rate_1000ms.median() if not df_windows.empty else np.nan
summary_md=f"""# Síntese do baseline nativo

## Observações diretamente medidas

- Runs válidas descobertas: **{len(run_data)}**; animais por run: **{df_general.animals.iloc[0] if len(df_general) else 'indisponível'}**.
- Frames por run: média **{df_general.frames.mean():.0f}**; adequados: média **{native_suitable:.0f}**.
- Duração média: **{native_duration/60:.2f} min**; FPS global por intervalos: **{native_mean_fps:.3f} frame/s**.
- Latência pós-captura média por animal: **{df_animals.post_capture_s.mean():.4f} s**; P95: **{df_animals.post_capture_s.quantile(.95):.4f} s**.
- Tempo integrado de inferência: média **{df_inferences.inference_s.mean()*1000:.3f} ms** quando disponível.
- Fração com predição após a última captura: **{resid_share:.1%}** quando disponível.

## Interpretações permitidas

- A média de entrada resume a trace, mas não substitui a distribuição de intervalos nem os picos locais de frames adequados.
- A mediana do pico em janela de 1 s é **{peak1:.3f} adequados/s** quando disponível; ela é comparada separadamente à capacidade isolada (**{PREDICTOR_CAPACITY_S:.3f} inferências/s**) e à capacidade integrada observada por run, apenas como indicação de sobrecarga transitória potencial.
- A repetibilidade deve ser julgada pela dispersão entre as cinco runs, não por tratar os mesmos 184 animais como 920 réplicas independentes.

## Hipóteses que exigem dados adicionais

- Uma rajada acima da capacidade isolada pode contribuir para atraso, mas não prova causalidade nem revela o tamanho da fila.
- Sem timestamps absolutos para todos os estágios, não é possível reconstruir exatamente a fila ou o backlog do preditor.
- Sem flags de throttling/frequência, temperatura alta não confirma throttling.

## Limitações

- A associação captura–seleção é ordinal no log, mas o pareamento agora usa todas as capturas e todas as decisões antes do filtro `SUITABLE`.
- A matriz de confusão do seletor é uma checagem de consistência dos rótulos, não a avaliação principal do classificador.
- A comparação com FPS fixo não deve interpolar o baseline nem ignorar jitter.
- Energia por frame adequado inclui o consumo-base do Raspberry Pi e não é um custo marginal puro.
"""
display(Markdown(summary_md))
''')

md("## 18. Exportação\n\nOs dados originais não são alterados. São criadas tabelas processadas, figuras PNG/PDF/SVG quando possível, um JSON consolidado, inconsistências e um relatório Markdown. O relatório de correções acompanha o notebook corrigido.")
code(r'''
from IPython.display import Markdown
TABLE_DIR=OUTPUT_ROOT/"tables"; FIG_DIR=OUTPUT_ROOT/"figures"; PROC_DIR=OUTPUT_ROOT/"processed"; REPORT_DIR=OUTPUT_ROOT/"report"
for p in [TABLE_DIR,FIG_DIR,PROC_DIR,REPORT_DIR]: p.mkdir(parents=True,exist_ok=True)

def export_df(df,name):
    if df is None or df.empty: return
    df.to_csv(TABLE_DIR/(name+".csv"),index=False)
    if len(df) > 5000:
        return
    try: (TABLE_DIR/(name+".md")).write_text(df.to_markdown(index=False),encoding="utf-8")
    except Exception: (TABLE_DIR/(name+".md")).write_text(df.to_string(index=False),encoding="utf-8")
    try: df.to_latex(TABLE_DIR/(name+".tex"),index=False)
    except Exception: pass

exports={"integrity":df_integrity,"inconsistencies":df_inconsistencies,"pairing_audit":df_pairing_audit,"paired_capture_selection":df_paired,"general_by_run":df_general,"resource_summary":df_resource_summary,"energy":df_energy,"latency_summary":df_latency_summary,"cadence_by_animal":df_cadence,"capture_intervals":df_intervals,"suitable_frames":df_suitable,"bursts":df_bursts,"burst_summary":df_burst_summary,"moving_window_rates":df_windows,"window_sensitivity":df_window_sensitivity if "df_window_sensitivity" in globals() else pd.DataFrame(),"capacity":df_capacity,"residual":df_residual,"ordinal_evidence":df_ordinal,"throughput":df_throughput,"busy_blocks":df_busy_blocks,"busy_block_summary":df_busy_summary,"selector_confusion":df_confusion,"selector_confusion_metrics":pd.DataFrame([confusion_metrics]),"repeatability":df_repeatability,"correlation_by_animal":df_corr_animal,"baseline_vs_10fps":df_comparison,"baseline_vs_10fps_runs":df_comparison_runs}
for name,df in exports.items(): export_df(df,name)
for name,df in {"animals":df_animals,"inferences":df_inferences,"captures":df_captures,"selections":df_selections,"resources":df_resources}.items(): export_df(df, "processed_"+name)

def json_safe(obj):
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.floating,)): return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj,(pd.Timestamp,datetime)): return obj.isoformat()
    if isinstance(obj,Path): return str(obj)
    if isinstance(obj,dict): return {str(k):json_safe(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [json_safe(v) for v in obj]
    return obj

consolidated={"configuration":{"native_timestamps":True,"fps":None,"mode":"single","engine":"thread","burst_definitions":BURST_DEFINITIONS,"windows_ms":WINDOWS_MS,"busy_gap_thresholds_s":BUSY_GAP_THRESHOLDS_S,"bootstrap_reps":BOOTSTRAP_REPS,"bootstrap_seed":BOOTSTRAP_SEED,"isolated_service_time_ms":WARMUP_REFERENCE_MS,"isolated_capacity_s":PREDICTOR_CAPACITY_S},"integrity":df_integrity.to_dict("records"),"pairing_audit":df_pairing_audit.to_dict("records"),"general":df_general.to_dict("records"),"resources":df_resource_summary.to_dict("records"),"energy":df_energy.to_dict("records"),"latency":df_latency_summary.to_dict("records"),"residual":df_residual.to_dict("records"),"throughput":df_throughput.to_dict("records"),"capacity":df_capacity.to_dict("records"),"repeatability":df_repeatability.to_dict("records"),"selector_confusion":confusion_metrics}
(REPORT_DIR/"baseline_summary.json").write_text(json.dumps(json_safe(consolidated),ensure_ascii=False,indent=2),encoding="utf-8")
(REPORT_DIR/"inconsistencies.csv").write_text(df_inconsistencies.to_csv(index=False),encoding="utf-8")
(REPORT_DIR/"baseline_summary.md").write_text(summary_md,encoding="utf-8")

# Tabela curta, com uma linha por indicador, para uso direto no artigo.
article_native = df_general.set_index("run")[["fps_global", "frames", "suitable"]].rename(columns={"fps_global":"effective_fps", "frames":"total_frames", "suitable":"suitable_frames"})
article_native["post_capture_latency_s"] = df_animals.groupby("run").post_capture_s.mean()
article_native["cpu_mean_pct"] = df_resource_summary.set_index("run").cpu_mean
article_native["cpu_p95_pct"] = df_resource_summary.set_index("run").cpu_p95
article_native["temperature_mean_c"] = df_resource_summary.set_index("run").temp_mean_c
article_native["power_mean_w"] = df_energy.set_index("run").power_mean_w
article_native["energy_j"] = df_energy.set_index("run").energy_j
article_native["inference_service_time_ms"] = df_inferences.groupby("run").inference_s.mean().mul(1000)
article_native["peak_local_1s_fps"] = df_windows.groupby("run").peak_rate_1000ms.mean()
article_native["residual_delay_s"] = df_animals.groupby("run").pred_residual_s.mean()
article_native["share_above_isolated_capacity_1s"] = df_windows.assign(_above=df_windows.rho_isolated_1000ms.gt(1).fillna(False)).groupby("run")._above.mean()
article_native["share_above_integrated_capacity_1s"] = df_windows.assign(_above=df_windows.rho_integrated_service_1000ms.gt(1).fillna(False)).groupby("run")._above.mean()
article_fixed = df_comparison_runs[df_comparison_runs.condition.eq("10fps_fixed")].set_index("run") if "df_comparison_runs" in globals() and not df_comparison_runs.empty else pd.DataFrame()
if isinstance(article_fixed,pd.DataFrame) and not article_fixed.empty:
    article_fixed["post_capture_latency_s"] = article_fixed["post_capture_completion_latency_s"]
    article_fixed["inference_service_time_ms"] = article_fixed["inference_service_time_s"] * 1000
    article_fixed["cpu_mean_pct"] = article_fixed["cpu_mean"]
    article_fixed["cpu_p95_pct"] = article_fixed["cpu_p95"]
article_specs = [("effective_fps", "FPS efetivo", "frames/s"), ("total_frames", "Frames totais", "frames"), ("suitable_frames", "Frames adequados", "frames"), ("post_capture_latency_s", "Latência pós-captura", "s"), ("inference_service_time_ms", "Tempo de serviço de inferência", "ms"), ("cpu_mean_pct", "CPU média", "%"), ("cpu_p95_pct", "CPU P95", "%"), ("temperature_mean_c", "Temperatura média", "°C"), ("power_mean_w", "Potência média", "W"), ("energy_j", "Energia", "J"), ("peak_local_1s_fps", "Pico local em janela de 1 s", "adequados/s"), ("share_above_isolated_capacity_1s", "Animais acima da capacidade isolada em 1 s", "proporção"), ("share_above_integrated_capacity_1s", "Animais acima da capacidade integrada em 1 s", "proporção"), ("residual_delay_s", "Atraso residual", "s")]
article_rows=[]
for key,label,unit in article_specs:
    bv=article_native[key].dropna() if key in article_native else pd.Series(dtype=float)
    fv=article_fixed[key].dropna() if isinstance(article_fixed,pd.DataFrame) and key in article_fixed else pd.Series(dtype=float)
    article_rows.append({"indicator":label,"baseline_native_mean":bv.mean() if len(bv) else np.nan,"baseline_native_sd":bv.std(ddof=1) if len(bv)>1 else np.nan,"fixed_10fps_mean":fv.mean() if len(fv) else np.nan,"fixed_10fps_sd":fv.std(ddof=1) if len(fv)>1 else np.nan,"unit":unit,"definition":"média e DP entre runs; pico local e proporções agregados por animal dentro de cada run"})
df_article_indicators=pd.DataFrame(article_rows)
display(df_article_indicators.round(5))
export_df(df_article_indicators,"article_indicators")

corrections_md = """# Correções da análise baseline de cinco runs

## Problemas encontrados

- O notebook anterior filtrava `SUITABLE` antes de associar decisões às capturas, deslocando os tempos adequados para o início da passagem.
- A taxa anteriormente chamada de throughput ocupado usava a janela global de inferências e incluía períodos sem trabalho.
- A capacidade isolada e o tempo de serviço observado no pipeline não estavam apresentados como referências distintas.
- Não havia auditoria por animal, matriz de confusão ordinal, testes sintéticos completos ou comparação equivalente com 10 FPS.

## Alterações realizadas

- Pareamento completo captura–seleção por animal e `log_order`, seguido do filtro `SUITABLE`.
- Auditoria de capturas, seleções, pareamento, monotonicidade, duplicações, summaries e `[FINAL]`, com warnings explícitos.
- O campo `forwarded` do `[SELECT-SUMMARY]` é preservado como contador cumulativo; o delta por animal é calculado antes da comparação com `SUITABLE`, evitando warnings falsos.
- A interpretação cumulativa de `forwarded` foi corrigida usando deltas sucessivos por run; os deltas são comparados ao número de adequados pareados.
- A correlação entre carga e atraso residual foi refeita após agregação pela mediana de cada animal entre as cinco runs.
- A diferença entre taxa média de rajada e pico em janela de 1 s foi explicitada no notebook.
- A taxa média global de adequados foi retirada da tabela baseline versus 10 FPS; o pico local por animal foi mantido.
- Foi exportada a tabela curta `article_indicators` para o artigo.
- Rajadas com limiar configurável: 1,5× mediana, 2,0× mediana e 250 ms; taxa `(N-1)/Δt`, taxa por mediana dos intervalos e tratamento `NaN` para rajada unitária.
- Janelas móveis de 250, 500, 1000 e 2000 ms com `searchsorted` e teste sintético `[0, 0.1, 0.2, 0.9, 1.1]` s.
- Razões `rho_isolated` e `rho_integrated_service`; a taxa global foi renomeada para `run_level_prediction_completion_rate`.
- Throughput em blocos ocupados reportado apenas como blocos observáveis e com sensibilidade a 0,25/0,50 s.
- Trabalho residual separado em medições diretas e evidência ordinal; nenhuma sobreposição temporal é inferida apenas da ordem textual.
- Comparação baseline/10 FPS com definições, unidades e agregação por run equivalentes.
- Resumo científico separado em observação, interpretação e hipótese.

## Validações e limitações

- Foram adicionados asserts para pareamento, filtro posterior, rajadas, janelas, atraso residual, capacidade, bloco unitário, animal sem adequado e divergência de contagens.
- Os mesmos animais aparecem nas cinco runs; análises por animal são descritivas/exploratórias e não tratam 920 linhas como réplicas independentes.
- A associação de tempo entre captura e seleção permanece ordinal porque os logs não fornecem timestamp absoluto para cada decisão.
- Arquivos brutos das runs não foram alterados.

## Outputs

As tabelas e figuras derivadas foram regeneradas em `outputs_baseline/`; o notebook corrigido é `baseline_5runs_analysis_corrected.ipynb`.
"""
(REPO_ROOT/"data-analysis"/"baseline_5runs_analysis_corrections.md").write_text(corrections_md,encoding="utf-8")
metadata={"created_utc":datetime.utcnow().isoformat()+"Z","python":sys.version,"platform":platform.platform(),"hostname":socket.gethostname(),"repo_root":str(REPO_ROOT),"runs_discovered":len(run_data),"files":file_inventory.to_dict("records"),"method":"native timestamps for latency/inference; complete ordinal capture-selection pairing before SUITABLE filtering; no temporal overlap inferred from log order"}
(REPORT_DIR/"metadata.json").write_text(json.dumps(json_safe(metadata),ensure_ascii=False,indent=2),encoding="utf-8")
print("Exportado em:",OUTPUT_ROOT.resolve())
''')

md("## 19. Checklist metodológico\n\n- [x] Cinco runs baseline descobertas e validadas quando presentes.\n- [x] FPS efetivo por intervalos `(N-1)/Δt` separado da versão alternativa.\n- [x] Cadência média separada de rajadas e janelas móveis.\n- [x] Capacidade isolada separada de throughput integrado.\n- [x] Trabalho residual direto separado de evidência ordinal.\n- [x] Temperatura não é tratada como prova de throttling.\n- [x] As cinco runs são a unidade para repetibilidade.\n- [x] Arquivos originais não são modificados.\n\nA execução deve ser feita de cima para baixo. Para reprodutibilidade, registre o commit do código, o diretório de dados usado e o ambiente Python.")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}
OUT.write_text(nbf.writes(nb), encoding="utf-8")
print(f"wrote {OUT} ({len(cells)} cells)")
