"""Entry point do pipeline MAS data-driven por tag.

Uso:
    python mas-main.py <mode> [fps] [num_animals] [max_passage_seconds]
                        [--debug] [--engine {thread,pade}]
                        [--native-timestamps]

    mode                 : 'mas-single' | 'mas-batch'
    fps                  : taxa de captura simulada (frames por segundo); não é
                           necessário quando --native-timestamps está ativo
    num_animals (opc)    : limita quantos animais (tags) processar; default: todos
    max_passage_seconds  : cap do span de cada animal; default: sem cap
    --debug              : modo verbose — log por frame (label real x classificacao
                           do seletor x peso) e salva TODO o log em
                           infra/reports/<pid>/debug.log
    --engine             : 'thread' (padrão, pipeline de threads) | 'pade' (PADE/FIPA)
    --native-timestamps  : captura cada frame uma vez, seguindo os
                           relative_time_ms do simulation_index.json

Exemplos:
    python mas-main.py mas-single 5 3              # 3 animais a 5 fps (threads)
    python mas-main.py mas-batch 10                # todos os animais a 10 fps
    python mas-main.py mas-single --native-timestamps  # todos os frames originais
    python mas-main.py mas-single --native-timestamps --num-animals 3
    python mas-main.py mas-single --native-timestamps --engine pade
    python mas-main.py mas-single 5 3 --debug      # 3 animais, log detalhado
    python mas-main.py mas-single 5 3 --engine pade # mesmo pipeline via PADE/FIPA
    python mas-main.py mas-single 5 192 30 --debug # todos, cap 30s/animal, log detalhado

O monitoramento de CPU/RAM/Temp escreve cpu.csv/mem.csv/temp.csv em
infra/reports/<pid>/ em ambos os engines (mesma psutil, mesma cadência).
"""

import sys
import os
import argparse
from datetime import datetime

# TensorFlow configuration — MUST be before any TF import (igual ao main.py)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_INTRA_OP_PARALLELISM_THREADS"] = "1"
os.environ["TF_INTER_OP_PARALLELISM_THREADS"] = "1"

from mas_pipeline import MASStrategy


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline MAS data-driven por tag (sheep-weighing CV).",
    )
    p.add_argument("mode", help="'mas-single' | 'mas-batch'")
    # Mantém compatibilidade com as chamadas posicionais existentes, mas
    # permite omitir fps no modo --native-timestamps.
    p.add_argument("fps", type=float, nargs="?", default=None,
                   help="taxa de captura simulada (fps)")
    p.add_argument("num_animals", type=int, nargs="?", default=None,
                   help="quantos animais (tags) processar (default: todos)")
    p.add_argument("max_passage_seconds", type=float, nargs="?", default=None,
                   help="cap do span de cada animal em segundos (default: sem cap)")
    # Formas nomeadas são úteis quando fps é omitido no modo nativo.
    p.add_argument("--num-animals", dest="num_animals_opt", type=int,
                   default=None,
                   help="limita a quantidade de animais (útil sem fps)")
    p.add_argument("--max-passage-seconds", dest="max_passage_seconds_opt",
                   type=float, default=None,
                   help="cap do span por animal (útil sem fps)")
    p.add_argument("--debug", action="store_true",
                   help="modo verbose + salva todo o log em debug.log")
    p.add_argument("--engine", choices=["thread", "pade"], default="thread",
                   help="engine de orquestração: 'thread' (padrão) ou 'pade'")
    p.add_argument("--native-timestamps", action="store_true",
                   help="segue os timestamps originais do dataset")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.num_animals is not None and args.num_animals_opt is not None:
        parser.error("num_animals não pode ser informado nas formas posicional e nomeada")
    if (args.max_passage_seconds is not None
            and args.max_passage_seconds_opt is not None):
        parser.error(
            "max_passage_seconds não pode ser informado nas formas posicional e nomeada"
        )

    num_animals = (args.num_animals_opt
                   if args.num_animals_opt is not None else args.num_animals)
    max_passage_seconds = (
        args.max_passage_seconds_opt
        if args.max_passage_seconds_opt is not None
        else args.max_passage_seconds
    )

    if not args.native_timestamps:
        if args.fps is None:
            parser.error("fps é obrigatório no modo normal")
        if args.fps <= 0:
            parser.error("fps deve ser maior que zero")

    if "batch" in args.mode:
        mode = "batch"
    elif "single" in args.mode:
        mode = "single"
    else:
        print(f"[ERROR] mode deve ser mas-single ou mas-batch (recebido: {args.mode})")
        sys.exit(1)

    # pid inclui o engine p/ não colidir relatórios entre thread/pade
    pid = f"{args.mode}_{args.engine}_{datetime.now().isoformat()}"
    reports_dir = f"infra/reports/{pid}"
    os.makedirs(reports_dir, exist_ok=True)

    # --debug: duplica stdout/stderr para debug.log e liga logs verbose
    log_file = None
    if args.debug:
        from mas.utils.debug_log import enable_debug_log
        log_file = enable_debug_log(f"{reports_dir}/debug.log")
        print(f"[DEBUG] log verbose sendo gravado em {reports_dir}/debug.log")
        print(f"[DEBUG] mode={mode} fps={args.fps} num_animals={num_animals} "
              f"max_passage_seconds={max_passage_seconds} engine={args.engine} "
              f"native_timestamps={args.native_timestamps}")

    try:
        if args.engine == "pade":
            strategy = MASStrategy(
                pid=pid,
                mode=mode,
                fps=args.fps,
                num_animals=num_animals,
                max_passage_seconds=max_passage_seconds,
                native_timestamps=args.native_timestamps,
                verbose=args.debug,
            )
        else:
            from thread_pipeline import ThreadPipeline
            strategy = ThreadPipeline(
                pid=pid,
                mode=mode,
                fps=args.fps,
                num_animals=num_animals,
                max_passage_seconds=max_passage_seconds,
                native_timestamps=args.native_timestamps,
                verbose=args.debug,
            )
        strategy.run()
    finally:
        if log_file is not None:
            log_file.flush()
            log_file.close()


if __name__ == "__main__":
    main()
