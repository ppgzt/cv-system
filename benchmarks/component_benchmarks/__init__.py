"""Benchmarks isolados dos componentes do pipeline Edge AI (sheep weighing).

Cada módulo mede um estágio real, reusando as classes/interpretadores do
pipeline (FrameSelection, DataEnhance, PredictWeight) com a mesma configuração
(num_threads=2, XNNPACK, mesmas preprocess_fn / transforms). Ver
benchmark_components.py (CLI) e BENCHMARK_COMPONENTS.md (reprodutibilidade).
"""

from .base import BenchmarkBase
from .selector import SelectorBenchmark, SELECTOR_DEFAULTS
from .enhancer import EnhancerBenchmark
from .predictor import PredictorBenchmark, PREDICTOR_DEFAULTS
from .aggregation import AggregationBenchmark, DEFAULT_SIZES

__all__ = [
    "BenchmarkBase",
    "SelectorBenchmark",
    "SELECTOR_DEFAULTS",
    "EnhancerBenchmark",
    "PredictorBenchmark",
    "PREDICTOR_DEFAULTS",
    "AggregationBenchmark",
    "DEFAULT_SIZES",
]
