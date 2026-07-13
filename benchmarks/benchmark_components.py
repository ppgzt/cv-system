#!/usr/bin/env python3
"""Benchmark isolado dos componentes do pipeline Edge AI (sheep weighing).

Mede, de forma sequencial e isolada, o tempo de serviço de cada estágio real do
pipeline (seletor / enhancement / preditor / agregação) no Raspberry Pi 5 —
reusando as classes e interpretadores TFLite reais com a MESMA configuração do
pipeline (num_threads=2, delegate XNNPACK default, mesmas preprocess_fn /
transforms). Não repete a latência fim a fim (já medida nos experimentos FPS).

As variáveis de ambiente do TensorFlow abaixo são setadas ANTES de qualquer
import do TF, replicando exatamente o mas-main.py do pipeline.

Uso:
    python benchmarks/benchmark_components.py --component selector
    python benchmarks/benchmark_components.py --component enhancer
    python benchmarks/benchmark_components.py --component predictor
    python benchmarks/benchmark_components.py --component aggregation
    python benchmarks/benchmark_components.py --component all
    python benchmarks/benchmark_components.py --component all --dry-run
    python benchmarks/benchmark_components.py --verify-pipeline-config

Defaults: --warmup 50 --iterations 1000 --seed 42.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# --------------------------------------------------------------------------- #
# 1. sys.path + variáveis de ambiente do TF ANTES de importar tensorflow.
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Replicam mas-main.py linha a linha (inter/intra op = 1; XNNPACK usa num_threads).
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_INTRA_OP_PARALLELISM_THREADS", "1")
os.environ.setdefault("TF_INTER_OP_PARALLELISM_THREADS", "1")

from component_benchmarks import (  # noqa: E402  (import após setar env TF)
    SelectorBenchmark,
    EnhancerBenchmark,
    PredictorBenchmark,
    AggregationBenchmark,
)
from component_benchmarks import environment  # noqa: E402
from component_benchmarks import monitoring   # noqa: E402
from component_benchmarks import imageset     # noqa: E402
from component_benchmarks import reporting    # noqa: E402

# Caminhos padrão idênticos aos do thread_pipeline.py (linhas 531-532).
DEFAULT_SELECTOR_MODEL = "infra/models/frame_selector.tflite"
DEFAULT_PREDICTOR_MODEL = "infra/models/sheep_weight_predictor.tflite"
DEFAULT_DATA_ROOT = "data/exp1"
DEFAULT_OUTPUT_DIR = "benchmarks/runs"
COMPONENT_SEED_OFFSETS = {
    "selector": 101,
    "enhancer": 202,
    "predictor": 303,
    "aggregation": 404,
}


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Microbenchmark isolado dos componentes do pipeline.")
    p.add_argument("--component", default="all",
                   choices=["selector", "enhancer", "predictor",
                            "aggregation", "all"],
                   help="componente a medir (default: all, sequencial)")
    p.add_argument("--warmup", type=int, default=50,
                   help="operações de warm-up por estágio (default 50)")
    p.add_argument("--iterations", type=int, default=1000,
                   help="medições válidas por estágio (default 1000)")
    p.add_argument("--seed", type=int, default=42, help="semente fixa (default 42)")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="diretório base de saída")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--selector-model", default=DEFAULT_SELECTOR_MODEL)
    p.add_argument("--predictor-model", default=DEFAULT_PREDICTOR_MODEL)
    p.add_argument("--num-threads", type=int, default=2,
                   help="XNNPACK intra-op threads (default 2, igual ao pipeline)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="threshold do seletor (default 0.5)")
    p.add_argument("--pool-size", type=int, default=300,
                   help="tamanho do pool de imagens por componente")
    p.add_argument("--predictor-pool-cap", type=int, default=250,
                   help="cap do pool do preditor (tensores enhanced pesam memória)")
    p.add_argument("--limit-tags", type=int, default=None,
                   help="limitar N animais (None = rebanho completo)")
    p.add_argument("--monitor-interval", type=float, default=1.0,
                   help="intervalo do monitor de sistema em segundos (default 1)")
    p.add_argument("--no-monitor", action="store_true",
                   help="desligar a thread de monitoramento de sistema")
    p.add_argument("--no-plots", action="store_true",
                   help="pular geração de gráficos (mesmo se matplotlib presente)")
    p.add_argument("--decompose-enhancer", action="store_true",
                   help="medição secundária por sub-operação do enhancement")
    p.add_argument("--bootstrap-resamples", type=int, default=10000,
                   help="reamostragens do IC bootstrap (default 10000)")
    p.add_argument("--dry-run", action="store_true",
                   help="carrega modelos, seleciona imagens, roda poucas ops, "
                        "valida shapes/dtypes e imprime config — sem 1000 medições")
    p.add_argument("--verify-pipeline-config", action="store_true",
                   help="compara config do benchmark com a do pipeline e sai")
    return p


# --------------------------------------------------------------------------- #
def select_components(name: str) -> list[str]:
    if name == "all":
        return ["selector", "enhancer", "predictor", "aggregation"]
    return [name]


def build_pools(args) -> dict:
    """Constrói os pools de imagens por componente (carga única, fora do timer)."""
    pools = {}
    print(f"[SETUP] inventariando dataset em {args.data_root} ...")
    t0 = time.perf_counter()
    sel_pool, sel_stats = imageset.build_selector_pool(
        args.data_root, args.seed, per_class=max(1, args.pool_size // 2),
        limit_tags=args.limit_tags)
    enh_pool, enh_stats = imageset.build_suited_pool(
        args.data_root, args.seed + 1, quota=args.pool_size,
        limit_tags=args.limit_tags)
    pred_quota = min(args.pool_size, args.predictor_pool_cap)
    pred_pool, pred_stats = imageset.build_suited_pool(
        args.data_root, args.seed + 2, quota=pred_quota,
        limit_tags=args.limit_tags)
    dt = time.perf_counter() - t0
    print(f"[SETUP] pools prontos em {dt:.1f}s "
          f"(selector={len(sel_pool)}, enhancer={len(enh_pool)}, "
          f"predictor={len(pred_pool)})")
    pools["selector"] = (sel_pool, sel_stats)
    pools["enhancer"] = (enh_pool, enh_stats)
    pools["predictor"] = (pred_pool, pred_stats)
    return pools


def make_benchmark(name, args, pool, monitor):
    order, reuse = imageset.cyclic_order(len(pool), args.iterations,
                                         args.seed + COMPONENT_SEED_OFFSETS[name])
    if name == "selector":
        b = SelectorBenchmark(args.selector_model, pool, order, args.warmup,
                              args.iterations, args.seed, monitor=monitor,
                              threshold=args.threshold,
                              num_threads=args.num_threads)
    elif name == "enhancer":
        b = EnhancerBenchmark(pool, order, args.warmup, args.iterations,
                              args.seed, monitor=monitor,
                              decompose=args.decompose_enhancer)
    elif name == "predictor":
        b = PredictorBenchmark(args.predictor_model, pool, order, args.warmup,
                               args.iterations, args.seed, monitor=monitor,
                               num_threads=args.num_threads)
    elif name == "aggregation":
        b = AggregationBenchmark(args.warmup, args.iterations, args.seed,
                                 monitor=monitor)
    else:
        raise ValueError(name)
    b._reuse_counts = reuse
    return b


def model_info_for(name, bench, args) -> dict:
    info = {"num_threads": getattr(bench, "num_threads", None)}
    if name == "selector":
        info.update(path=args.selector_model,
                    sha256=reporting.sha256_of_file(args.selector_model),
                    size_mb=reporting.file_size_mb(args.selector_model),
                    threshold=bench.threshold,
                    input_shape=str(bench.input_shape),
                    input_dtype=str(bench.input_dtype),
                    preprocessing={"resize": "224x224 bilinear (no pad)",
                                   "clip_max": 4000.0, "norm_scale": 2000.0,
                                   "offset": -1.0, "grayscale_to_rgb": True})
    elif name == "predictor":
        info.update(path=args.predictor_model,
                    sha256=reporting.sha256_of_file(args.predictor_model),
                    size_mb=reporting.file_size_mb(args.predictor_model),
                    input_shape=str(bench.input_shape),
                    input_dtype=str(bench.input_dtype),
                    output_shape=str(bench.output_shape),
                    output_dtype=str(bench.output_dtype))
    elif name == "enhancer":
        info.update(num_threads=None,
                    transforms=["NoiseRemovalSetMaxValue(1950)",
                                "AdjustScaleWithFixedMaxValue(1950)",
                                "Replicate1DtoNDimChannel(3)",
                                "ResizeImageWithPadding((300,300))"],
                    input_shape=str(getattr(bench, "input_shape", None)),
                    input_dtype=str(getattr(bench, "input_dtype", None)))
    else:  # aggregation
        info = {"operation": "float(np.mean(weights))",
                "sizes": list(bench.sizes)}
    if name in ("selector", "predictor"):
        info["delegates"] = (
            "TFLite CPU default delegate (XNNPACK when available; "
            "not explicitly configured by the pipeline)")
        info["delegate_verification"] = (
            "best_effort: TFLite does not expose a stable public delegate "
            "introspection API")
    else:
        info["delegates"] = None
        info["delegate_verification"] = "not_applicable"
    return info


# --------------------------------------------------------------------------- #
def run_dry_run(args, pools) -> int:
    print("\n================ DRY-RUN ================")
    comps = select_components(args.component)
    for name in comps:
        if name == "aggregation":
            # aggregation não tem pool/ctx; exercita np.mean nos tamanhos reais.
            import numpy as np
            b = AggregationBenchmark(args.warmup, args.iterations, args.seed)
            b.setup()
            for i, s in enumerate(b.sizes):
                t0 = time.perf_counter_ns()
                r = float(np.mean(b.weight_sets[s]))
                dt = (time.perf_counter_ns() - t0) / 1e6
                ok = bool(np.isfinite(r))
                print(f"[aggregation] size={s}: ok={ok} result={r:.4f} {dt:.4g}ms")
            print(f"[aggregation] sizes={b.sizes} op=float(np.mean(weights))")
            continue

        pool = pools.get(name, ([], {}))[0]
        if not pool:
            print(f"[{name}] pool vazio — pulando")
            continue
        b = make_benchmark(name, args, pool, monitor=None)
        b.setup()
        # roda 3 operações (warm-up mínimo) e valida
        for i in range(3):
            idx = b.order[i % len(b.order)] if b.order else 0
            ctx = b._make_ctx(idx, i + 1)
            ctx["timestamp_monotonic_ns"] = time.monotonic_ns()
            ctx["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  time.gmtime())
            res = b._run_one(ctx)
            ok, reason = b._validate(res, ctx)
            print(f"[{name}] op{i+1}: ok={ok} {('— '+reason) if not ok else ''}")
        print(f"[{name}] input_shape={getattr(b,'input_shape',None)} "
              f"input_dtype={getattr(b,'input_dtype',None)} "
              f"output_shape={getattr(b,'output_shape',None)} "
              f"output_dtype={getattr(b,'output_dtype',None)}")
        info = model_info_for(name, b, args)
        print(f"[{name}] model_info: {info}")
        print(f"[{name}] corretude: {b._verify_correctness()}")
    print("============== FIM DO DRY-RUN ==============")
    print("(nenhuma medição válida foi gravada)")
    return 0


# --------------------------------------------------------------------------- #
def verify_pipeline_config(args) -> int:
    print("================ VERIFY PIPELINE CONFIG ================")
    checks = []

    def ok(label, detail=""):
        checks.append(("OK", label, detail))

    def warn(label, detail=""):
        checks.append(("WARN", label, detail))

    def fail(label, detail=""):
        checks.append(("FAIL", label, detail))

    # 1. Caminhos/configurações idênticos ao thread_pipeline.py
    if args.selector_model == DEFAULT_SELECTOR_MODEL:
        ok("selector model path", f"{args.selector_model} (== pipeline)")
    else:
        fail("selector model path difere do pipeline",
             f"{args.selector_model} != {DEFAULT_SELECTOR_MODEL}")
    if args.predictor_model == DEFAULT_PREDICTOR_MODEL:
        ok("predictor model path", f"{args.predictor_model} (== pipeline)")
    else:
        fail("predictor model path difere do pipeline",
             f"{args.predictor_model} != {DEFAULT_PREDICTOR_MODEL}")
    if args.data_root == DEFAULT_DATA_ROOT:
        ok("dataset path", f"{args.data_root} (== pipeline)")
    else:
        fail("dataset path difere do pipeline",
             f"{args.data_root} != {DEFAULT_DATA_ROOT}")

    # 2. Arquivos existem + hash
    for label, path in [("selector", args.selector_model),
                        ("predictor", args.predictor_model)]:
        if os.path.exists(path):
            ok(f"{label} model file exists",
               f"sha256={reporting.sha256_of_file(path)[:16]}... "
               f"size={reporting.file_size_mb(path)}MB")
        else:
            fail(f"{label} model file missing", path)

    # 3. num_threads / threshold == adapter defaults
    if args.num_threads == 2:
        ok("num_threads=2", "== FrameSelectionAdapter/InferenceAdapter default")
    else:
        fail("num_threads != 2",
             f"{args.num_threads} (pipeline usa 2) — diferença do pipeline")
    if args.threshold == 0.5:
        ok("threshold=0.5", "== FrameSelectionAdapter default")
    else:
        fail("threshold != 0.5", f"{args.threshold}")

    # 4. Constantes de preprocessing (lidas das classes reais)
    try:
        from domain.modules.frame_selection import FrameSelection
        from domain.modules.predict_weight import PredictWeight

        ok("selector _IMG_SIZE", str(FrameSelection._IMG_SIZE))
        ok("selector _CLIP_MAX", str(FrameSelection._CLIP_MAX))
        ok("selector _NORM_SCALE", str(FrameSelection._NORM_SCALE))
        if os.path.exists(args.selector_model):
            selector = FrameSelection(args.selector_model,
                                       threshold=args.threshold,
                                       num_threads=args.num_threads,
                                       suitable_window=None)
            selector_detail = selector._interpreter.get_input_details()[0]
            actual_shape = tuple(int(x) for x in selector_detail["shape"])
            actual_dtype = str(selector_detail["dtype"])
            if actual_shape == (1, 224, 224, 3):
                ok("selector input shape", str(actual_shape))
            else:
                fail("selector input shape", str(actual_shape))
            if actual_dtype == "<class 'numpy.float32'>":
                ok("selector input dtype", actual_dtype)
            else:
                fail("selector input dtype", actual_dtype)
            delegates = getattr(selector._interpreter, "_delegates", None)
            if not delegates:
                warn("selector delegate",
                     "nenhum delegate explícito exposto; XNNPACK é "
                     "selecionado automaticamente pelo TFLite quando disponível")
            else:
                ok("selector delegate", f"introspectado ({len(delegates)})")

        if os.path.exists(args.predictor_model):
            predictor = PredictWeight(args.predictor_model,
                                      num_threads=args.num_threads)
            predictor_detail = predictor._interpreter.get_input_details()[0]
            pred_shape = tuple(int(x) for x in predictor_detail["shape"])
            pred_dtype = str(predictor_detail["dtype"])
            if pred_shape == (1, 300, 300, 3):
                ok("predictor input shape", str(pred_shape))
            else:
                fail("predictor input shape", str(pred_shape))
            if pred_dtype == "<class 'numpy.float32'>":
                ok("predictor input dtype", pred_dtype)
            else:
                fail("predictor input dtype", pred_dtype)
            out_detail = predictor._interpreter.get_output_details()[0]
            ok("predictor output shape", str(tuple(int(x) for x in out_detail["shape"])))
            ok("predictor output dtype", str(out_detail["dtype"]))
            delegates = getattr(predictor._interpreter, "_delegates", None)
            if not delegates:
                warn("predictor delegate",
                     "nenhum delegate explícito exposto; XNNPACK é "
                     "selecionado automaticamente pelo TFLite quando disponível")
            else:
                ok("predictor delegate", f"introspectado ({len(delegates)})")
    except Exception as e:  # noqa: BLE001
        fail("ler constantes FrameSelection", repr(e))
    try:
        from domain.modules.data_enhance import DataEnhance
        e = DataEnhance()
        actual = [(type(t).__name__, getattr(t, "max_value", None),
                   getattr(t, "dim", None), getattr(t, "shape", None))
                  for t in e.transfs]
        expected = [("NoiseRemovalSetMaxValue", 1950, None, None),
                    ("AdjustScaleWithFixedMaxValue", 1950, None, None),
                    ("Replicate1DtoNDimChannel", None, 3, None),
                    ("ResizeImageWithPadding", None, None, (300, 300))]
        if actual == expected:
            ok("enhancer transforms", str(actual))
        else:
            fail("enhancer transforms", f"got={actual} want={expected}")
    except Exception as exc:  # noqa: BLE001
        fail("instanciar DataEnhance", repr(exc))

    # 5. Env vars de TF (precisam bater com mas-main.py)
    for k, want in [("TF_INTRA_OP_PARALLELISM_THREADS", "1"),
                    ("TF_INTER_OP_PARALLELISM_THREADS", "1")]:
        got = os.environ.get(k)
        if got == want:
            ok(f"env {k}={want}", "== mas-main.py")
        else:
            warn(f"env {k}", f"got={got} want={want}")

    # 6. Delegates — a API pública do TFLite não garante introspecção estável.
    warn("delegates",
         "XNNPACK é o default do tf.lite; pipeline também não seta delegates "
         "explicitamente — assume-se idêntico. Não verificável automaticamente.")

    print("\nstatus | check | detalhe")
    print("-------+-------+-------")
    for status, label, detail in checks:
        print(f"{status:6} | {label} | {detail}")
    n_fail = sum(1 for s, _, _ in checks if s == "FAIL")
    print(f"\n{len(checks)} verificações: "
          f"{sum(1 for s,_,_ in checks if s=='OK')} OK, "
          f"{sum(1 for s,_,_ in checks if s=='WARN')} WARN, {n_fail} FAIL")
    return 1 if n_fail else 0


# --------------------------------------------------------------------------- #
def main() -> int:
    args = build_parser().parse_args()

    if args.verify_pipeline_config:
        return verify_pipeline_config(args)

    env_before = environment.snapshot_environment()
    print(f"[ENV] host={env_before.get('hostname')} "
          f"device={env_before.get('device_model')} "
          f"temp0={env_before.get('temperature_celsius')}°C")

    pools = build_pools(args)

    if args.dry_run:
        run_dry_run(args, pools)
        return 0

    comps = select_components(args.component)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    report_dir = os.path.join(args.output_dir,
                              f"benchmark_components_{stamp}")
    os.makedirs(report_dir, exist_ok=True)
    print(f"[OUT] diretório de relatório: {report_dir}")

    components_out: dict = {}
    all_monitor_rows: list[dict] = []
    bench_for_info: dict = {}

    for name in comps:
        if name == "aggregation":
            pool = []
        else:
            pool = pools[name][0]
            if not pool:
                print(f"[{name}] pool vazio — pulando (registre em failures)")
                continue
        monitor = None if args.no_monitor else monitoring.SystemMonitor(
            component_fn=(lambda n=name: n),
            interval=args.monitor_interval,
            pid=os.getpid())
        b = make_benchmark(name, args, pool, monitor)
        b.setup()
        bench_for_info[name] = b
        _CURRENT_COMPONENT[0] = name

        print(f"\n[{name}] warmup={args.warmup} iterações={args.iterations} "
              f"pool={len(pool)} seed={args.seed}")
        t0 = time.perf_counter()
        run_result = b.run()
        dt = time.perf_counter() - t0
        print(f"[{name}] concluído em {dt:.1f}s -> {run_result}")
        if monitor is not None:
            all_monitor_rows.extend(monitor.rows)

        components_out[name] = {
            "benchmark": b,
            "run_result": run_result,
            "reuse_counts": getattr(b, "_reuse_counts", []),
        }

    env_after = environment.snapshot_environment()
    print(f"[ENV] temp1={env_after.get('temperature_celsius')}°C")

    # model_infos + pool_stats
    model_infos = {n: model_info_for(n, bench_for_info[n], args)
                   for n in components_out if n in bench_for_info}
    pool_stats = {n: pools[n][1] for n in components_out
                  if n in pools and n != "aggregation"}
    pool_stats["aggregation"] = {"note": "pesos sintéticos (seed fixa)"}

    bootstrap_cfg = {"method": "bootstrap_percentile",
                     "n_resamples": args.bootstrap_resamples,
                     "seed": args.seed, "ci_level": 0.95,
                     "statistics": ["median"] + [
                         f"p{q}" for q in (1, 5, 10, 25, 75, 90, 95, 99)],
                     "mean_ci": "t-Student 95%"}

    config = {"component": args.component, "warmup": args.warmup,
              "iterations": args.iterations, "seed": args.seed,
              "num_threads": args.num_threads, "threshold": args.threshold,
              "pool_size": args.pool_size, "monitor_interval": args.monitor_interval,
              "decompose_enhancer": args.decompose_enhancer,
              "data_root": args.data_root,
              "selector_model": args.selector_model,
              "predictor_model": args.predictor_model,
              "start_utc": env_before.get("timestamp_utc"),
              "end_utc": env_after.get("timestamp_utc")}

    reporter = reporting.Reporter(
        report_dir=report_dir, config=config, env_before=env_before,
        env_after=env_after, model_infos=model_infos, pool_stats=pool_stats,
        bootstrap_cfg=bootstrap_cfg, plots=not args.no_plots)

    # escreve CSVs de medições/warmup/failures
    all_failures = []
    for name, info in components_out.items():
        b = info["benchmark"]
        reporter.write_measurements(name, b.measurements)
        reporter.write_warmup(name, b.warmup_rows)
        all_failures.extend(b.failures)
    reporter.write_monitor(all_monitor_rows)
    reporter.write_failures(all_failures)
    reporter.compute_summaries(components_out, all_monitor_rows)
    reporter.write_all()

    print("\n==========================================================")
    print(f"BENCHMARK CONCLUÍDO. Relatório: {report_dir}")
    for name, info in components_out.items():
        rr = info["run_result"]
        print(f"  {name:11s}: válidas={rr['valid']}/{rr['requested']} "
              f"falhas={rr['failures']} warmup={rr['warmup']}")
    if all_failures:
        print(f"  ( {len(all_failures)} falhas registradas em failures.csv )")
    print("==========================================================")
    # marcador机器-legível p/ o run_benchmark_components.sh capturar o caminho:
    print(f"__REPORT_DIR__={os.path.abspath(report_dir)}")
    return 0


# componente corrente, lido pelo monitor (setado no loop principal)
_CURRENT_COMPONENT = ["idle"]


if __name__ == "__main__":
    raise SystemExit(main())
