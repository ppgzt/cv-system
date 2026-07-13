"""Detecção do ambiente de execução (hardware/software/energia).

Tudo aqui é só-leitura e NUNCA altera governor/frequência/afinidade — apenas
detecta e documenta, conforme §8 da especificação do benchmark.

Comandos Raspberry Pi (vcgencmd) e arquivos /sys são lidos com tratamento de
erro; quando indisponível (ex.: macOS no desenvolvimento), registra
"not_available" em vez de falhar.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from datetime import datetime

# Variáveis de ambiente relacionadas a threads que devem ser registradas.
_THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "TF_INTRA_OP_PARALLELISM_THREADS",
    "TF_INTER_OP_PARALLELISM_THREADS",
    "TF_ENABLE_ONEDNN_OPTS",
    "KERAS_BACKEND",
    "XNNPACK_FLAGS",
]

NA = "not_available"


# --------------------------------------------------------------------------- #
# Utilidades de leitura defensiva
# --------------------------------------------------------------------------- #
def _read_file(path: str) -> str | None:
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return None


def _run_cmd(args: list[str], timeout: float = 2.0) -> str | None:
    """Roda um comando externo; retorna stdout stripped ou None."""
    try:
        out = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
            text=True,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def _vcgencmd(subcmd: list[str]) -> str | None:
    return _run_cmd(["vcgencmd", *subcmd])


# --------------------------------------------------------------------------- #
# /sys e device-tree (Linux/ARM, sobretudo Raspberry Pi)
# --------------------------------------------------------------------------- #
def read_pi_model() -> str:
    return _read_file("/proc/device-tree/model") or NA


def read_cpuinfo() -> dict:
    info = {"model_name": NA, "hardware": NA, "revision": NA, "serial": NA}
    txt = _read_file("/proc/cpuinfo")
    if not txt:
        # macOS / outros: usa platform.processor()
        try:
            info["model_name"] = platform.processor() or NA
        except Exception:
            pass
        return info
    for line in txt.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower()
        val = val.strip()
        if key == "model name" and info["model_name"] == NA:
            info["model_name"] = val
        elif key == "hardware":
            info["hardware"] = val
        elif key == "revision":
            info["revision"] = val
        elif key == "serial":
            info["serial"] = val
    return info


def read_cpu_freq() -> dict:
    """Frequência atual/mín/máx (Hz) e governor, lendo cpufreq."""
    out = {"scaling_cur_freq_hz": NA, "scaling_min_freq_hz": NA,
           "scaling_max_freq_hz": NA, "governor": NA}
    base = "/sys/devices/system/cpu/cpu0/cpufreq"
    cur = _read_file(f"{base}/scaling_cur_freq")
    fmin = _read_file(f"{base}/scaling_min_freq")
    fmax = _read_file(f"{base}/scaling_max_freq")
    gov = _read_file(f"{base}/scaling_governor")
    if cur is not None:
        try:
            out["scaling_cur_freq_hz"] = int(cur) * 1000  # kHz -> Hz
        except ValueError:
            out["scaling_cur_freq_hz"] = cur
    if fmin is not None:
        try:
            out["scaling_min_freq_hz"] = int(fmin) * 1000
        except ValueError:
            out["scaling_min_freq_hz"] = fmin
    if fmax is not None:
        try:
            out["scaling_max_freq_hz"] = int(fmax) * 1000
        except ValueError:
            out["scaling_max_freq_hz"] = fmax
    if gov is not None:
        out["governor"] = gov
    return out


def read_temp_celsius() -> float | None:
    """Temperatura em °C: vcgencmd measure_temp OU /sys/class/thermal."""
    out = _vcgencmd(["measure_temp"])
    if out and "=" in out:
        # "temp=52.3'C"
        try:
            return float(out.split("=")[1].split("'")[0].strip())
        except Exception:
            pass
    raw = _read_file("/sys/class/thermal/thermal_zone0/temp")
    if raw is not None:
        try:
            return int(raw) / 1000.0
        except ValueError:
            return None
    return None


def measure_clock_arm_hz() -> int | None:
    """vcgencmd measure_clock arm -> frequência em Hz."""
    out = _vcgencmd(["measure_clock", "arm"])
    if out and "=" in out:
        try:
            return int(out.split("=")[1].strip())
        except Exception:
            return None
    return None


def read_throttled() -> dict:
    """Interpreta vcgencmd get_throttled (bitmask de throttle/undervoltage).

    Bits atuais (ocorrendo agora): 0 undervoltage, 1 arm freq capped,
    2 throttled, 3 soft temp limit.
    Bits históricos (ocorreram desde o boot): 16-19 espelham 0-3.
    """
    res = {
        "raw": NA, "undervoltage_now": NA, "freq_capped_now": NA,
        "throttled_now": NA, "soft_temp_now": NA,
        "undervoltage_occurred": NA, "freq_capped_occurred": NA,
        "throttled_occurred": NA, "soft_temp_occurred": NA,
    }
    out = _vcgencmd(["get_throttled"])
    if not out or "=" not in out:
        return res
    try:
        val = int(out.split("=")[1].strip(), 0)
    except Exception:
        res["raw"] = out
        return res
    res["raw"] = hex(val)
    res["undervoltage_now"] = bool(val & (1 << 0))
    res["freq_capped_now"] = bool(val & (1 << 1))
    res["throttled_now"] = bool(val & (1 << 2))
    res["soft_temp_now"] = bool(val & (1 << 3))
    res["undervoltage_occurred"] = bool(val & (1 << 16))
    res["freq_capped_occurred"] = bool(val & (1 << 17))
    res["throttled_occurred"] = bool(val & (1 << 18))
    res["soft_temp_occurred"] = bool(val & (1 << 19))
    return res


# --------------------------------------------------------------------------- #
# Versões de bibliotecas (best-effort)
# --------------------------------------------------------------------------- #
def _version(import_path: str, attr: str = "__version__") -> str:
    try:
        mod = __import__(import_path, fromlist=["x"])
        return str(getattr(mod, attr, NA))
    except Exception:
        return NA


def read_library_versions() -> dict:
    # tensorflow (e tflite-runtime, se presente) são especialmente relevantes.
    versions = {
        "tensorflow": _version("tensorflow"),
        "tflite_runtime": _version("tflite_runtime"),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "psutil": _version("psutil"),
        "scikit-image (skimage)": _version("skimage"),
        "pillow (PIL)": _version("PIL"),
        "opencv (cv2)": _version("cv2"),
        "matplotlib": _version("matplotlib"),
    }
    # Versão do interpretador LiteRT (tf.lite) se disponível.
    try:
        import tensorflow as tf  # noqa: WPS433
        interp = tf.lite.Interpreter  # type: ignore[attr-defined]
        versions["tf.lite.Interpreter"] = "available (class)"
        versions["tf.lite_module_path"] = getattr(interp, "__module__", NA)
    except Exception as e:  # noqa: BLE001
        versions["tf.lite.Interpreter"] = f"error: {e}"
    return versions


def read_thread_env_vars() -> dict:
    return {k: os.environ.get(k, NA) for k in _THREAD_ENV_VARS}


# --------------------------------------------------------------------------- #
# Snapshot completo (chamado no início/fim do benchmark)
# --------------------------------------------------------------------------- #
def snapshot_environment() -> dict:
    """Coleta tudo o que der sobre o ambiente, SEM falhar."""
    import psutil  # local: psutil é dep do pipeline

    pid = os.getpid()
    proc = psutil.Process(pid)
    cpuinfo = read_cpuinfo()

    # Load average (Unix); Windows lança -> NA
    try:
        load_avg = list(os.getloadavg())
    except Exception:
        load_avg = NA

    # Afinidade de CPU do processo
    try:
        affinity = list(proc.cpu_affinity())
    except Exception:
        affinity = NA

    vm = psutil.virtual_memory()

    snap = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "kernel_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor() or NA,
        "device_model": read_pi_model(),
        "cpu": {
            "model_name": cpuinfo["model_name"],
            "hardware": cpuinfo["hardware"],
            "revision": cpuinfo["revision"],
            "logical_cores": psutil.cpu_count(logical=True) or NA,
            "physical_cores": psutil.cpu_count(logical=False) or NA,
            "frequency_mhz": (psutil.cpu_freq()._asdict() if psutil.cpu_freq() else NA),
            "cpufreq_sys": read_cpu_freq(),
            "clock_arm_hz": measure_clock_arm_hz(),
        },
        "memory": {
            "total_bytes": vm.total,
            "available_bytes": vm.available,
            "available_gb": round(vm.available / 1e9, 3),
            "total_gb": round(vm.total / 1e9, 3),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys_executable_safe(),
        },
        "libraries": read_library_versions(),
        "thread_env_vars": read_thread_env_vars(),
        "load_average": load_avg,
        "process_cpu_affinity": affinity,
        "temperature_celsius": read_temp_celsius(),
        "throttled": read_throttled(),
    }
    return snap


def sys_executable_safe() -> str:
    import sys
    return sys.executable
