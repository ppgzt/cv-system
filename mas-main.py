"""Entrypoint experimental oficial do PIBIC.

O comando suportado para o runtime atual é ``python mas-main.py --engine pade
--mode <condição>``. Argumentos posicionais antigos permanecem apenas para
reprodução histórica e não devem ser usados em deploys novos.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# MUST precede qualquer import de TensorFlow.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_INTRA_OP_PARALLELISM_THREADS"] = "1"
os.environ["TF_INTER_OP_PARALLELISM_THREADS"] = "1"

from mas.experiment_config import (
    DEFAULT_IDLE_PATIENCE,
    DEFAULT_LOW_FPS,
    DEFAULT_MEDIUM_FPS,
    DEFAULT_PDI_THRESHOLD,
    DEFAULT_PIXEL_THRESHOLD_MM,
    DEFAULT_ROI_FRACTIONS,
    DEFAULT_SELECTOR_THRESHOLD,
    EXPERIMENT_MODES,
    RESOURCE_CRITICAL_TEMPERATURE_C,
    RESOURCE_STALE_AFTER_SECONDS,
    RESOURCE_WARNING_PREDICTION_BACKLOG,
    RESOURCE_WARNING_TEMPERATURE_C,
    get_experiment_mode,
)
from mas_pipeline import MASStrategy


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Runtime experimental PIBIC (oficial: --engine pade --mode ...).",
    )
    parser.add_argument("legacy_mode", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("legacy_fps", nargs="?", type=float, help=argparse.SUPPRESS)
    parser.add_argument("legacy_num_animals", nargs="?", type=int, help=argparse.SUPPRESS)
    parser.add_argument("legacy_max_passage_seconds", nargs="?", type=float, help=argparse.SUPPRESS)

    parser.add_argument("--engine", choices=["thread", "pade"], default="pade")
    parser.add_argument("--mode", choices=tuple(EXPERIMENT_MODES), dest="experiment_mode")
    parser.add_argument("--aggregation-mode", choices=["single", "batch"], default="single")
    parser.add_argument("--fps", type=float, default=None, help="FPS do modo fixed-fps (obrigatório nesse modo)")
    parser.add_argument("--low-fps", type=float, default=DEFAULT_LOW_FPS)
    parser.add_argument("--medium-fps", type=float, default=DEFAULT_MEDIUM_FPS)
    parser.add_argument("--num-animals", type=int, default=None)
    parser.add_argument("--max-passage-seconds", type=float, default=None)
    parser.add_argument("--data-root", default="data/exp1")
    parser.add_argument("--output-dir", default="infra/reports")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--repetition", default=None)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--frame-selection-model", default="infra/models/frame_selector.tflite")
    parser.add_argument("--selector-threshold", type=float, default=DEFAULT_SELECTOR_THRESHOLD)
    parser.add_argument("--visual-pdi-threshold", type=float, default=DEFAULT_PDI_THRESHOLD)
    parser.add_argument("--visual-pixel-threshold-mm", type=float, default=DEFAULT_PIXEL_THRESHOLD_MM)
    parser.add_argument("--visual-idle-patience", type=int, default=DEFAULT_IDLE_PATIENCE)
    parser.add_argument("--visual-roi", type=float, nargs=4, metavar=("Y0", "Y1", "X0", "X1"), default=DEFAULT_ROI_FRACTIONS)

    parser.add_argument("--resource-warning-temperature", type=float, default=RESOURCE_WARNING_TEMPERATURE_C)
    parser.add_argument("--resource-critical-temperature", type=float, default=RESOURCE_CRITICAL_TEMPERATURE_C)
    parser.add_argument("--resource-warning-backlog", type=int, default=RESOURCE_WARNING_PREDICTION_BACKLOG)
    parser.add_argument("--resource-stale-after-seconds", type=float, default=RESOURCE_STALE_AFTER_SECONDS)

    # Compatibilidade explícita de chamadas antes do congelamento do CLI.
    parser.add_argument("--native-timestamps", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--visual-event", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--visual-gated", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--no-visual-gated", dest="visual_gated", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--selection-hold-n", type=int, default=2, help=argparse.SUPPRESS)
    return parser


def _validate_common(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not Path(args.frame_selection_model).is_file():
        parser.error(f"Frame Selection model not found: {args.frame_selection_model}")
    if not 0.0 <= args.selector_threshold <= 1.0:
        parser.error("--selector-threshold deve estar entre 0 e 1")
    if args.low_fps <= 0 or args.medium_fps <= 0:
        parser.error("--low-fps e --medium-fps devem ser maiores que zero")
    if args.visual_pdi_threshold < 0 or args.visual_pixel_threshold_mm <= 0:
        parser.error("thresholds visuais inválidos")
    if args.visual_idle_patience <= 0:
        parser.error("--visual-idle-patience deve ser maior que zero")
    y0, y1, x0, x1 = args.visual_roi
    if not (0 <= y0 < y1 <= 1 and 0 <= x0 < x1 <= 1):
        parser.error("--visual-roi exige 0 <= y0 < y1 <= 1 e 0 <= x0 < x1 <= 1")
    if args.resource_warning_temperature > args.resource_critical_temperature:
        parser.error("warning temperature não pode exceder critical temperature")
    if args.resource_warning_backlog < 0 or args.resource_stale_after_seconds <= 0:
        parser.error("resource backlog/stale timeout inválido")


def resolve_experiment_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    """Produz uma configuração sem ambiguidade para o launcher PADE."""
    _validate_common(args, parser)
    if args.experiment_mode:
        if args.legacy_mode is not None:
            parser.error("use --mode ou argumentos posicionais legados, não ambos")
        if args.engine != "pade":
            parser.error("as cinco modalidades oficiais exigem --engine pade")
        spec = get_experiment_mode(args.experiment_mode)
        if spec.requires_fixed_fps:
            if args.fps is None or args.fps <= 0:
                parser.error("--mode fixed-fps exige --fps <valor maior que zero>")
        elif args.fps is not None:
            parser.error("--fps é permitido apenas no modo fixed-fps")
        return {
            "experiment_mode": spec.name,
            "aggregation_mode": args.aggregation_mode,
            "fps": args.fps if spec.requires_fixed_fps else None,
            "native_timestamps": spec.native_timestamps,
            "visual_event_enabled": spec.visual_adaptive,
            "visual_gated": spec.visual_gated,
            "resource_control_enabled": spec.resource_cap,
            "num_animals": args.num_animals,
            "max_passage_seconds": args.max_passage_seconds,
        }

    # Compatibilidade temporária com: mas-main.py mas-single [fps] [...].
    if args.legacy_mode is None:
        parser.error("informe uma das cinco condições com --mode")
    if args.fps is not None:
        parser.error("--fps nomeado pertence ao CLI oficial; use --mode fixed-fps --fps N")
    if "batch" in args.legacy_mode:
        aggregation_mode = "batch"
    elif "single" in args.legacy_mode:
        aggregation_mode = "single"
    else:
        parser.error("modo legado deve ser mas-single ou mas-batch")
    if not args.native_timestamps and not args.visual_event and args.legacy_fps is None:
        parser.error("fps é obrigatório no modo legado normal")
    if args.legacy_fps is not None and args.legacy_fps <= 0:
        parser.error("fps deve ser maior que zero")
    return {
        "experiment_mode": "legacy",
        "aggregation_mode": aggregation_mode,
        "fps": args.legacy_fps,
        "native_timestamps": args.native_timestamps,
        "visual_event_enabled": args.visual_event,
        "visual_gated": args.visual_gated,
        "resource_control_enabled": False,
        "num_animals": args.num_animals if args.num_animals is not None else args.legacy_num_animals,
        "max_passage_seconds": args.max_passage_seconds if args.max_passage_seconds is not None else args.legacy_max_passage_seconds,
    }


def main() -> None:
    # Permite chamar o entrypoint por caminho absoluto sem depender do cwd
    # (por exemplo, a partir de systemd ou dos scripts no Raspberry).
    os.chdir(PROJECT_ROOT)
    parser = build_parser()
    args = parser.parse_args()
    config = resolve_experiment_arguments(args, parser)

    run_id = args.run_id or f"{config['experiment_mode']}_{args.engine}_{datetime.now().isoformat()}"
    if args.repetition:
        run_id = f"{run_id}_rep-{args.repetition}"
    reports_dir = Path(args.output_dir)
    reports_dir.joinpath(run_id).mkdir(parents=True, exist_ok=True)

    log_file = None
    if args.debug:
        from mas.utils.debug_log import enable_debug_log
        log_file = enable_debug_log(str(reports_dir / run_id / "debug.log"))

    try:
        if args.engine == "pade":
            from mas.agents.resource_manager_agent import ResourceThresholds
            strategy = MASStrategy(
                pid=run_id,
                mode=config["aggregation_mode"],
                fps=config["fps"],
                low_fps=args.low_fps if config["visual_event_enabled"] else None,
                medium_fps=args.medium_fps if config["visual_event_enabled"] else None,
                visual_gated=config["visual_gated"],
                selection_hold_n=args.selection_hold_n,
                num_animals=config["num_animals"],
                max_passage_seconds=config["max_passage_seconds"],
                data_root=args.data_root,
                native_timestamps=config["native_timestamps"],
                visual_event_enabled=config["visual_event_enabled"],
                visual_pdi_threshold=args.visual_pdi_threshold,
                visual_pixel_threshold_mm=args.visual_pixel_threshold_mm,
                visual_idle_patience=args.visual_idle_patience,
                visual_roi_fractions=tuple(args.visual_roi),
                selector_threshold=args.selector_threshold,
                resource_thresholds=ResourceThresholds(
                    warning_temperature_c=args.resource_warning_temperature,
                    critical_temperature_c=args.resource_critical_temperature,
                    warning_prediction_backlog=args.resource_warning_backlog,
                    stale_after_seconds=args.resource_stale_after_seconds,
                ),
                resource_control_enabled=config["resource_control_enabled"],
                frame_selection_model=args.frame_selection_model,
                reports_dir=str(reports_dir),
                verbose=args.debug,
            )
        else:
            if args.experiment_mode:
                parser.error("as cinco modalidades oficiais exigem --engine pade")
            from thread_pipeline import ThreadPipeline
            strategy = ThreadPipeline(
                pid=run_id,
                mode=config["aggregation_mode"],
                fps=config["fps"],
                num_animals=config["num_animals"],
                max_passage_seconds=config["max_passage_seconds"],
                native_timestamps=config["native_timestamps"],
                frame_selection_model=args.frame_selection_model,
                verbose=args.debug,
            )
        strategy.run()
    finally:
        if log_file is not None:
            log_file.flush()
            log_file.close()


if __name__ == "__main__":
    main()
