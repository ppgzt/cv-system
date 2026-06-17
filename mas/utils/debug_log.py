"""Captura de log completo para o modo --debug.

Há dois produtores de saída no pipeline:

1. `print()` / tracebacks do reator -> vão para `sys.stdout`/`sys.stderr`.
2. `display_message` (pade) -> usa `click.echo(click.style(...))`, ou seja,
   escreve via `click.echo`.

Por isso NÃO basta trocar `sys.stdout` por um Tee: o `click` pode resolver (e
cachear) o stream de stdout independente da nossa substituição, e as mensagens
dos agentes simplesmente não chegam ao arquivo. A solução robusta é:

- instalar um Tee em stdout/stderr (captura `print()` e tracebacks); E
- patchear `click.echo` para escrever a linha (sem códigos ANSI) no arquivo e
  no console real diretamente, SEM repassar ao `click.echo` original (evita
  dupla gravação, já que o Tee do stdout também escreve no arquivo).

Assim o `infra/reports/<pid>/debug.log` contém TODO o log: mensagens dos
agentes, prints e eventuais tracebacks do reactor.
"""

import re
import sys
from pathlib import Path

_ANSI = re.compile(r'\x1b\[[0-9;]*m')


class _Tee:
    """Stream que escreve em múltiplos streams subjacentes."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
        # força flush no arquivo para não perder linhas em crash
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    # delegações comuns (isatty, fileno, etc.) para não quebrar libs
    def isatty(self):
        for s in self.streams:
            if getattr(s, "isatty", None) and s.isatty():
                return True
        return False

    def __getattr__(self, name):
        # fallback: delega ao primeiro stream (stdout original)
        return getattr(self.streams[0], name)


# Estado do patch (module-level para ser idempotente).
_log_file = None
_console_out = None   # terminal real (stdout original, antes do Tee)
_orig_echo = None


def _echo_tee(message='', *args, **kwargs):
    """Wrapper de click.echo: grava no arquivo (sem ANSI) e no console real.

    Não chama o click.echo original para evitar dupla gravação no arquivo (o
    Tee do stdout já trata print/traceback -> arquivo; e nossas mensagens dos
    agentes passam por aqui, não pelo sys.stdout).
    """
    global _log_file, _console_out
    text = '' if message is None else str(message)
    clean = _ANSI.sub('', text)
    if _log_file is not None:
        try:
            _log_file.write(clean + '\n')
            _log_file.flush()
        except Exception:
            pass
    if _console_out is not None:
        try:
            _console_out.write(text + '\n')
            _console_out.flush()
        except Exception:
            pass


def enable_debug_log(log_path) -> object:
    """Instala o Tee em stdout/stderr e patcheia click.echo.

    Retorna o handle do arquivo aberto (o chamador pode fechá-lo ao fim).
    """
    global _log_file, _console_out, _orig_echo

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(log_path, "a", encoding="utf-8")

    # Console real = stdout original (antes de qualquer substituição).
    _console_out = sys.stdout

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_stdout, _log_file)
    sys.stderr = _Tee(orig_stderr, _log_file)

    # Patcheia click.echo UMA vez. display_message -> click.echo -> _echo_tee.
    import click
    if _orig_echo is None:
        _orig_echo = click.echo
        click.echo = _echo_tee

    return _log_file
