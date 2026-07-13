"""Estatísticas para microbenchmarks (§13, §14, §15 da especificação).

Todas as durações entram em nanossegundos (int64). As funções de resumo
reportam campos *_ns e *_ms (ms = ns / 1e6). Não removemos outliers
silenciosamente: a versão sem outliers (IQR) é opcional e claramente marcada
como secundária.
"""

from __future__ import annotations

import math
import numpy as np

try:
    from scipy import stats as sp_stats  # skew, kurtosis, t, linregress, pearsonr
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

_PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def _to_ns(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    return a[np.isfinite(a)]


# --------------------------------------------------------------------------- #
# Resumo de uma métrica (nanossegundos)
# --------------------------------------------------------------------------- #
def describe_ns(samples_ns) -> dict:
    """Resumo estatístico completo de uma métrica temporal (entrada em ns)."""
    a = _to_ns(samples_ns)
    n = int(a.size)
    if n == 0:
        return _empty_describe()

    mean = float(a.mean())
    median = float(np.median(a))
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    var = float(a.var(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 0 else 0.0

    pct = {f"p{q}": float(np.percentile(a, q)) for q in _PERCENTILES}
    q1, q3 = pct["p25"], pct["p75"]
    iqr = q3 - q1
    mad = float(np.median(np.abs(a - median)))

    # CI 95% da média via t de Student
    if _HAVE_SCIPY and n > 1:
        t_crit = float(sp_stats.t.ppf(0.975, df=n - 1))
        ci_mean = [mean - t_crit * sem, mean + t_crit * sem]
    else:  # fallback normal aproximado
        z = 1.959963984540054
        ci_mean = [mean - z * sem, mean + z * sem]

    skew = float(sp_stats.skew(a, bias=False)) if (_HAVE_SCIPY and n > 2) else NA()
    kurt = float(sp_stats.kurtosis(a, fisher=True, bias=False)) if (_HAVE_SCIPY and n > 3) else NA()

    # Outliers IQR (1.5 x IQR) — só contagem, não removemos
    low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers_iqr = int(np.sum((a < low_fence) | (a > high_fence)))

    cv = (std / mean) if mean != 0 else NA()
    mean_median_pct_diff = ((mean - median) / median * 100.0) if median != 0 else NA()

    d = {
        "n_valid": n,
        "mean_ns": mean,
        "mean_ms": mean / 1e6,
        "median_ns": median,
        "median_ms": median / 1e6,
        "std_ns": std,
        "std_ms": std / 1e6,
        "variance_ns2": var,
        "min_ns": float(a.min()),
        "min_ms": float(a.min()) / 1e6,
        "max_ns": float(a.max()),
        "max_ms": float(a.max()) / 1e6,
        "range_ns": float(a.max() - a.min()),
        "range_ms": (a.max() - a.min()) / 1e6,
        "cv": cv,
        "sem_ns": sem,
        "ci_mean_95_ns": [ci_mean[0], ci_mean[1]],
        "ci_mean_95_ms": [ci_mean[0] / 1e6, ci_mean[1] / 1e6],
        "iqr_ns": iqr,
        "mad_ns": mad,
        "skewness": skew,
        "kurtosis_excess": kurt,
        "n_outliers_iqr": n_outliers_iqr,
        "count_above_p95": int(np.sum(a > pct["p95"])),
        "count_above_p99": int(np.sum(a > pct["p99"])),
        "mean_median_pct_diff": mean_median_pct_diff,
    }
    # percentis em ns e ms
    for q, v in pct.items():
        d[f"{q}_ns"] = v
        d[f"{q}_ms"] = v / 1e6
    return d


def NA():
    return None


def _empty_describe() -> dict:
    d = {
        "n_valid": 0, "mean_ns": None, "mean_ms": None, "median_ns": None,
        "median_ms": None, "std_ns": None, "std_ms": None, "variance_ns2": None,
        "min_ns": None, "min_ms": None, "max_ns": None, "max_ms": None,
        "range_ns": None, "range_ms": None, "cv": None, "sem_ns": None,
        "ci_mean_95_ns": [None, None], "ci_mean_95_ms": [None, None],
        "iqr_ns": None, "mad_ns": None, "skewness": None,
        "kurtosis_excess": None, "n_outliers_iqr": 0, "count_above_p95": 0,
        "count_above_p99": 0, "mean_median_pct_diff": None,
    }
    for q in _PERCENTILES:
        d[f"p{q}_ns"] = None
        d[f"p{q}_ms"] = None
    return d


# --------------------------------------------------------------------------- #
# Bootstrap (mediana e percentis) — seed fixa, >= 10k reamostragens
# --------------------------------------------------------------------------- #
def bootstrap_ci(samples_ns, statistic="median", n_boot: int = 10000,
                 seed: int = 42, ci: float = 0.95) -> dict:
    """IC bootstrap de 95% para a mediana (default) ou outro percentil.

    n_boot reamostragens com reposição, RNG com seed fixa (reprodutível).
    """
    return bootstrap_cis(samples_ns, [statistic], n_boot=n_boot,
                         seed=seed, ci=ci).get(statistic, {
                             "statistic": statistic,
                             "method": "bootstrap_percentile",
                             "n_resamples": n_boot,
                             "seed": seed,
                             "ci_level": ci,
                             "ci_ns": [None, None],
                             "ci_ms": [None, None],
                             "point_estimate_ns": None,
                         })


def bootstrap_cis(samples_ns, statistics=None, n_boot: int = 10000,
                  seed: int = 42, ci: float = 0.95) -> dict:
    """Calcula ICs bootstrap em uma única rodada de reamostragem.

    ``statistics`` aceita ``median`` e rótulos como ``p95``. Compartilhar a
    mesma matriz de reamostragem torna os ICs reprodutíveis e evita repetir
    desnecessariamente a operação mais pesada para cada percentil.
    """
    a = _to_ns(samples_ns)
    n = int(a.size)
    if statistics is None:
        statistics = ["median"]
    statistics = list(dict.fromkeys(statistics))

    def empty(statistic):
        return {"statistic": statistic, "method": "bootstrap_percentile",
                "n_resamples": n_boot, "seed": seed, "ci_level": ci,
                "ci_ns": [None, None], "ci_ms": [None, None],
                "point_estimate_ns": None}

    out = {statistic: empty(statistic) for statistic in statistics}
    if n < 2 or n_boot < 1:
        return {statistic: empty(statistic) for statistic in statistics}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = a[idx]  # (n_boot, n)
    alpha = (1.0 - ci) / 2.0
    for statistic in statistics:
        if statistic == "median":
            stat = np.median(boots, axis=1)
            point = float(np.median(a))
        elif statistic == "mean":
            stat = boots.mean(axis=1)
            point = float(a.mean())
        else:
            q = int(str(statistic).replace("p", ""))
            stat = np.percentile(boots, q, axis=1)
            point = float(np.percentile(a, q))
        lo, hi = np.percentile(stat, [100 * alpha, 100 * (1 - alpha)])
        out[statistic]["ci_ns"] = [float(lo), float(hi)]
        out[statistic]["ci_ms"] = [float(lo / 1e6), float(hi / 1e6)]
        out[statistic]["point_estimate_ns"] = point
    return out


# --------------------------------------------------------------------------- #
# Throughput e razões
# --------------------------------------------------------------------------- #
def throughput_ops_per_sec(mean_ns, median_ns) -> dict:
    return {
        "ops_per_sec_by_mean": (1e9 / mean_ns) if mean_ns else None,
        "ops_per_sec_by_median": (1e9 / median_ns) if median_ns else None,
    }


def iqr_outlier_mask(samples_ns) -> np.ndarray:
    a = _to_ns(samples_ns)
    if a.size == 0:
        return np.array([], dtype=bool)
    q1, q3 = np.percentile(a, [25, 75])
    fence = 1.5 * (q3 - q1)
    return (a >= q1 - fence) & (a <= q3 + fence)


# --------------------------------------------------------------------------- #
# Análise temporal (§15)
# --------------------------------------------------------------------------- #
def temporal_slope_ns_per_iter(samples_ns) -> dict:
    """Inclinação da latência ao longo das iterações (regressão linear)."""
    a = _to_ns(samples_ns)
    n = a.size
    if n < 2:
        return {"slope_ns_per_iter": None, "rvalue": None, "pvalue": None,
                "stderr": None}
    x = np.arange(n, dtype=np.float64)
    if _HAVE_SCIPY:
        r = sp_stats.linregress(x, a)
        return {"slope_ns_per_iter": float(r.slope),
                "rvalue": float(r.rvalue), "pvalue": float(r.pvalue),
                "stderr": float(r.stderr)}
    # fallback OLS
    slope = float(np.polyfit(x, a, 1)[0])
    return {"slope_ns_per_iter": slope, "rvalue": None, "pvalue": None,
            "stderr": None}


def split_halves(samples_ns) -> dict:
    a = _to_ns(samples_ns)
    n = a.size
    if n < 4:
        return {}
    h = n // 2
    first_half, second_half = a[:h], a[h:]
    return {
        "first_half_mean_ns": float(first_half.mean()),
        "second_half_mean_ns": float(second_half.mean()),
        "first_half_mean_ms": first_half.mean() / 1e6,
        "second_half_mean_ms": second_half.mean() / 1e6,
        "second_vs_first_pct": ((second_half.mean() - first_half.mean())
                                / first_half.mean() * 100.0),
    }


def first_last_n(samples_ns, n: int = 100) -> dict:
    a = _to_ns(samples_ns)
    if a.size < 2 * n:
        n = a.size // 2
    if n == 0:
        return {}
    first, last = a[:n], a[-n:]
    return {
        "first_n_mean_ms": first.mean() / 1e6,
        "last_n_mean_ms": last.mean() / 1e6,
        "last_vs_first_pct": ((last.mean() - first.mean()) / first.mean() * 100.0),
    }


def block_stats(samples_ns, block: int = 100) -> list[dict]:
    """Estatísticas por bloco de `block` iterações (ex.: 1-100, 101-200...)."""
    a = _to_ns(samples_ns)
    blocks = []
    for i in range(0, a.size, block):
        chunk = a[i:i + block]
        blocks.append({
            "block_index": i // block,
            "iter_start": i + 1,
            "iter_end": i + chunk.size,
            "n": int(chunk.size),
            "mean_ms": float(chunk.mean() / 1e6),
            "median_ms": float(np.median(chunk) / 1e6),
            "p95_ms": float(np.percentile(chunk, 95) / 1e6),
            "min_ms": float(chunk.min() / 1e6),
            "max_ms": float(chunk.max() / 1e6),
        })
    return blocks


# --------------------------------------------------------------------------- #
# Correlações latência x ambiente (requer séries alinhadas por tempo monotônico)
# --------------------------------------------------------------------------- #
def _align_by_time(env_times_ns, env_vals, lat_times_ns, lat_vals) -> tuple:
    """Para cada latência (por iteração), devolve o valor de ambiente (1 Hz) do
    timestamp monotônico mais próximo. Retorna (lat_vals, env_vals_alinhados).
    """
    env_t = np.asarray(env_times_ns, dtype=np.float64)
    env_v = np.asarray(env_vals, dtype=np.float64)
    lat_t = np.asarray(lat_times_ns, dtype=np.float64)
    lat_v = np.asarray(lat_vals, dtype=np.float64)
    if env_t.size == 0 or lat_t.size == 0:
        return np.array([]), np.array([])
    idx = np.clip(np.searchsorted(env_t, lat_t), 0, env_t.size - 1)
    return lat_v, env_v[idx]


def correlation(latency_times_ns, latency_vals_ns,
                env_times_ns, env_vals) -> dict:
    """Correlação de Pearson entre latência (por iteração) e um sinal de ambiente
    (1 Hz), alinhando pelo timestamp monotônico mais próximo."""
    lat_v, env_a = _align_by_time(env_times_ns, env_vals,
                                  latency_times_ns, latency_vals_ns)
    if lat_v.size < 3 or env_a.size < 3:
        return {"pearson_r": None, "pvalue": None, "n": int(lat_v.size)}
    mask = np.isfinite(lat_v) & np.isfinite(env_a)
    if int(mask.sum()) < 3:
        return {"pearson_r": None, "pvalue": None,
                "n": int(mask.sum())}
    if _HAVE_SCIPY:
        r = sp_stats.pearsonr(lat_v[mask], env_a[mask])
        return {"pearson_r": float(r[0]), "pvalue": float(r[1]),
                "n": int(mask.sum())}
    # fallback numpy
    lv, ev = lat_v[mask], env_a[mask]
    if lv.std() == 0 or ev.std() == 0:
        return {"pearson_r": None, "pvalue": None, "n": int(mask.sum())}
    r = float(np.corrcoef(lv, ev)[0, 1])
    return {"pearson_r": r, "pvalue": None, "n": int(mask.sum())}


def env_series_blocks(latency_times_ns, env_times_ns, env_vals,
                      n_iter: int, block: int = 100) -> list[dict]:
    """Média do sinal de ambiente por bloco de iterações (temp/freq/CPU média
    por bloco), alinhando pelo tempo."""
    lat_t = np.asarray(latency_times_ns, dtype=np.float64)
    env_t = np.asarray(env_times_ns, dtype=np.float64)
    env_v = np.asarray(env_vals, dtype=np.float64)
    blocks = []
    if env_t.size == 0:
        return blocks
    aligned = env_v[np.clip(np.searchsorted(env_t, lat_t), 0, env_t.size - 1)]
    for i in range(0, n_iter, block):
        chunk = aligned[i:i + block]
        if chunk.size == 0:
            continue
        blocks.append({"block_index": i // block,
                       "iter_start": i + 1,
                       "iter_end": i + chunk.size,
                       "mean": float(np.nanmean(chunk)) if chunk.size else None})
    return blocks
