#!/usr/bin/env bash
# ==============================================================================
# run_benchmark_components.sh — roda o microbenchmark de componentes no
# Raspberry Pi (via SSH) e puxa o relatório de volta para o Mac.
# PIBIC - CV System
# ------------------------------------------------------------------------------
# Roda no MAC (orquestrador). O benchmark roda no RASPBERRY (mesmo pyenv do
# pipeline). SEM medição de potência — este benchmark mede apenas os tempos
# internos dos componentes (seletor / enhancement / preditor / agregação).
#
# Protocolo:
#   1. preflight: valida SSH + pyenv + repo + dataset no Pi
#   2. ssh Pi -> python benchmarks/benchmark_components.py ...   (BLOQUEIA)
#   3. captura o __REPORT_DIR__ impresso pelo benchmark
#   4. scp -r puxa a pasta de relatório do Pi para o Mac
#
# Pré-requisitos no PI: SSH por chave (sem senha) e o pyenv cv_vend_mas
# (mesmas deps do pipeline). Veja BENCHMARK_COMPONENTS.md.
#
# Uso:
#   ./run_benchmark_components.sh                              # all, defaults
#   COMPONENT=selector ./run_benchmark_components.sh
#   COMPONENT=predictor WARMUP=100 ITERATIONS=2000 ./run_benchmark_components.sh
#   COMPONENT=all EXTRA_ARGS="--decompose-enhancer" ./run_benchmark_components.sh
#   COMPONENT=selector EXTRA_ARGS="--dry-run" ./run_benchmark_components.sh
# ==============================================================================
set -uo pipefail

# ============================== CONFIG ========================================
PI_HOST="${PI_HOST:-ewertonsjp@192.168.0.141}"   # usuario@host do Pi
PI_DIR="${PI_DIR:-~/projects/cv-system/}"        # repo no Pi (AJUSTE se necessário)
PY_PI="${PY_PI:-/home/ewertonsjp/.pyenv/versions/cv_vend_mas/bin/python}"

# Parâmetros do benchmark (defaults = especificação: warmup 50, 1000, seed 42)
COMPONENT="${COMPONENT:-all}"        # selector | enhancer | predictor | aggregation | all
WARMUP="${WARMUP:-50}"
ITERATIONS="${ITERATIONS:-1000}"
SEED="${SEED:-42}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Saída no Mac
TS="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="${WORK_DIR:-./benchmark_runs}"
OUT_DIR="${WORK_DIR}/${COMPONENT}_${TS}"
SSH_LOG="${OUT_DIR}/benchmark.log"
# ==============================================================================

ts() { date "+%Y-%m-%dT%H:%M:%S%z"; }
mkdir -p "$OUT_DIR"

echo "=========================================================="
echo "  BENCHMARK DE COMPONENTES — ${COMPONENT}"
echo "  Pipeline   : ${PI_HOST}:${PI_DIR}  (Raspberry, via SSH)"
echo "  Python(Pi) : ${PY_PI}"
echo "  Params     : warmup=${WARMUP} iters=${ITERATIONS} seed=${SEED}"
echo "  Extra      : ${EXTRA_ARGS:-(nenhum)}"
echo "  Saída(Mac) : ${OUT_DIR}"
echo "=========================================================="

# --- 0. Preflight: SSH + pyenv + repo + dataset --------------------------------
echo "[0/4] Preflight (SSH, pyenv, repo, dataset)..."
read -r PI_HOME _ < <(ssh -o BatchMode=yes "$PI_HOST" 'echo "$HOME $PWD"') || true
PF_CMD="test -x ${PY_PI} || exit 11; cd ${PI_DIR} 2>/dev/null || exit 12; \
test -f benchmarks/benchmark_components.py || exit 13; \
test -d data/exp1 || exit 14; echo PREFLIGHT_OK"
PF_OUT=$(ssh -o BatchMode=yes "$PI_HOST" "$PF_CMD" 2>&1) || PF_ST=$?
PF_ST=${PF_ST:-0}
if [ "$PF_ST" -ne 0 ] || [ "${PF_OUT##*$'\n'}" != "PREFLIGHT_OK" ]; then
    echo "[ERROR] Preflight falhou no Pi (exit=${PF_ST}). Saída:"
    echo "$PF_OUT" | sed 's/^/    /'
    case "$PF_ST" in
        11) echo "    -> pyenv python não encontrado: ${PY_PI}" ;;
        12) echo "    -> repo não encontrado no Pi: ${PI_DIR}" ;;
        13) echo "    -> benchmarks/benchmark_components.py ausente (同步 o repo no Pi)" ;;
        14) echo "    -> data/exp1 ausente no Pi" ;;
    esac
    exit 1
fi
echo "      Preflight OK."

# --- 1. Roda o benchmark no Pi (SSH bloqueia até terminar) --------------------
echo "[1/4] Disparando benchmark no Raspberry..."
CMD="cd ${PI_DIR} && ${PY_PI} benchmarks/benchmark_components.py"
CMD+=" --component '${COMPONENT}' --warmup ${WARMUP} --iterations ${ITERATIONS} --seed ${SEED}"
[ -n "$EXTRA_ARGS" ] && CMD+=" ${EXTRA_ARGS}"
echo "      ssh ${PI_HOST} \"${CMD}\""
echo "[T0] início: $(ts)" | tee -a "$SSH_LOG"
ssh -o BatchMode=yes "$PI_HOST" "$CMD" 2>&1 | tee -a "$SSH_LOG"
SSH_STATUS=${PIPESTATUS[0]}
echo "[T1] fim:    $(ts) (exit=${SSH_STATUS})" | tee -a "$SSH_LOG"

# --- 2. Captura o __REPORT_DIR__ impresso pelo benchmark -----------------------
# TensorFlow pode escrever um byte NUL no stderr; nesse caso o grep padrão
# classifica o log como binário e devolve "Binary file ... matches" em vez da
# linha encontrada. -a força tratamento textual e mantém o caminho do relatório.
REMOTE_DIR=$(LC_ALL=C grep -a -m1 '^__REPORT_DIR__=' "$SSH_LOG" \
    | cut -d= -f2- | tr -d '\r\000')
if [ -z "$REMOTE_DIR" ]; then
    # fallback: pega a pasta mais recente no Pi
    REMOTE_DIR=$(ssh -o BatchMode=yes "$PI_HOST" \
        "cd ${PI_DIR} && ls -td \"\$PWD\"/benchmarks/runs/benchmark_components_* 2>/dev/null | head -n1")
fi
if [ -z "$REMOTE_DIR" ]; then
    echo "[ERROR] não localizei a pasta de relatório no Pi (benchmark abortou?)."
    [ "$SSH_STATUS" -ne 0 ] && echo "          ssh exit=${SSH_STATUS} — veja ${SSH_LOG}"
    exit 1
fi
echo "      relatório no Pi: ${REMOTE_DIR}"

# --- 3. Puxa a pasta de relatório para o Mac ----------------------------------
echo "[2/4] Puxando relatório do Raspberry..."
if scp -q -o BatchMode=yes -r "${PI_HOST}:${REMOTE_DIR}" "$OUT_DIR/"; then
    LOCAL_REPORT="${OUT_DIR}/$(basename "$REMOTE_DIR")"
    echo "      puxado: ${REMOTE_DIR} -> ${LOCAL_REPORT}/"
else
    echo "      [WARN] scp falhou (relatórios continuam no Pi em ${REMOTE_DIR})"
    LOCAL_REPORT=""
fi

# --- 4. Resumo ----------------------------------------------------------------
echo "[3/4] CONCLUÍDO."
echo "----------------------------------------------------------"
echo "  log ssh    : ${SSH_LOG}  (ssh exit=${SSH_STATUS})"
if [ -n "$LOCAL_REPORT" ]; then
    echo "  relatório  : ${LOCAL_REPORT}/"
    echo "  metadata   : ${LOCAL_REPORT}/metadata.json"
    echo "  summary    : ${LOCAL_REPORT}/summary.csv"
    echo "  report.md  : ${LOCAL_REPORT}/report.md"
fi
[ "$SSH_STATUS" -ne 0 ] && echo "  [WARN] benchmark saiu com erro — veja ${SSH_LOG}"
echo "=========================================================="

if [ "$SSH_STATUS" -ne 0 ] || [ -z "$LOCAL_REPORT" ]; then
    exit 1
fi
