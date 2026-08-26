#!/usr/bin/env bash
# Roda no Mac e mede, no Raspberry, os quatro agentes PADE isoladamente.
# A métrica primária é service_time_ms; não é teste de throughput/concorrência.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
source "${PROJECT_ROOT}/scripts/mac_remote_power.sh"

WARMUP="${WARMUP:-50}"
ITERATIONS="${ITERATIONS:-1000}"
SEED="${SEED:-42}"
POOL_SIZE="${POOL_SIZE:-300}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/benchmark_runs}"
LOCAL_DIR="${WORK_DIR}/agent_microbenchmark_instrumented_${TIMESTAMP}"
LOG_FILE="${LOCAL_DIR}/remote_run.log"
METADATA_FILE="${LOCAL_DIR}/metadata.txt"

fail() { printf '[agent-microbenchmark] ERROR: %s\n' "$*" >&2; exit 1; }
timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

[[ "${WARMUP}" =~ ^[0-9]+$ ]] || fail 'WARMUP must be a non-negative integer'
[[ "${ITERATIONS}" == "1000" ]] || fail 'final protocol requires ITERATIONS=1000'
[[ "${WARMUP}" == "50" ]] || fail 'final protocol requires WARMUP=50'
[[ "${SEED}" =~ ^[0-9]+$ && "${POOL_SIZE}" =~ ^[1-9][0-9]*$ ]] || fail 'SEED/POOL_SIZE invalid'

mkdir -p "${LOCAL_DIR}"
require_remote_config

printf '%s\n' '=========================================================='
printf '%s\n' '  MICROBENCHMARK INSTRUMENTADO DOS AGENTES PADE'
printf '%s\n' "  Pi         : ${PI_TARGET}:${PI_PROJECT_ROOT}"
printf '%s\n' "  Python(Pi) : ${PI_PYTHON}"
printf '%s\n' "  Protocolo  : warmup=${WARMUP}; measurements=${ITERATIONS}; seed=${SEED}"
printf '%s\n' "  Saída(Mac) : ${LOCAL_DIR}"
printf '%s\n' '=========================================================='

PREFLIGHT_CMD="test -x '${PI_PYTHON}' || exit 11; cd '${PI_PROJECT_ROOT}' || exit 12; '${PI_PYTHON}' -c 'import mas; import pade' || exit 13; test -f benchmarks/benchmark_pade_agents_instrumented.py || exit 14; test -f benchmarks/analyze_pade_agents_instrumented.py || exit 15; test -d data/exp1 || exit 16; test -f infra/models/frame_selector.tflite || exit 17; test -f infra/models/sheep_weight_predictor.tflite || exit 18; echo PREFLIGHT_OK"
printf '%s\n' '[1/4] Preflight no Raspberry...'
PREFLIGHT_OUT="$(remote_exec "${PREFLIGHT_CMD}" 2>&1)" || PREFLIGHT_STATUS=$?
PREFLIGHT_STATUS="${PREFLIGHT_STATUS:-0}"
[[ "${PREFLIGHT_STATUS}" -eq 0 && "${PREFLIGHT_OUT##*$'\n'}" == 'PREFLIGHT_OK' ]] || { printf '%s\n' "${PREFLIGHT_OUT}" >&2; fail "preflight failed (exit=${PREFLIGHT_STATUS})"; }

LOCAL_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || printf unknown)"
REMOTE_COMMIT="$(remote_git_commit)"
{
    printf 'started_at=%s\n' "$(timestamp)"
    printf 'pi_target=%s\npi_project_root=%s\npi_python=%s\n' "${PI_TARGET}" "${PI_PROJECT_ROOT}" "${PI_PYTHON}"
    printf 'local_git_commit=%s\nremote_git_commit=%s\n' "${LOCAL_COMMIT}" "${REMOTE_COMMIT}"
    printf 'warmup=%s\niterations=%s\nseed=%s\npool_size=%s\n' "${WARMUP}" "${ITERATIONS}" "${SEED}" "${POOL_SIZE}"
} > "${METADATA_FILE}"

printf '%s\n' '[2/4] Executando os quatro componentes sequencialmente no Raspberry...'
REMOTE_CMD="cd '${PI_PROJECT_ROOT}' && '${PI_PYTHON}' benchmarks/benchmark_pade_agents_instrumented.py --warmup '${WARMUP}' --iterations '${ITERATIONS}' --seed '${SEED}' --pool-size '${POOL_SIZE}'"
printf '[T0] %s\n' "$(timestamp)" | tee -a "${LOG_FILE}"
set +e
remote_exec "${REMOTE_CMD}" 2>&1 | tee -a "${LOG_FILE}"
STATUS="${PIPESTATUS[0]}"
set -e
printf '[T1] %s exit=%s\n' "$(timestamp)" "${STATUS}" | tee -a "${LOG_FILE}"
[[ "${STATUS}" -eq 0 ]] || fail "benchmark remote failed (exit=${STATUS})"

REMOTE_DIR="$(LC_ALL=C grep -a -m1 '^__BENCHMARK_DIR__=' "${LOG_FILE}" | cut -d= -f2- | tr -d '\r\000')"
[[ -n "${REMOTE_DIR}" ]] || fail 'benchmark did not emit __BENCHMARK_DIR__'

printf '%s\n' '[3/4] Gerando análise no Raspberry...'
remote_exec "cd '${PI_PROJECT_ROOT}' && '${PI_PYTHON}' benchmarks/analyze_pade_agents_instrumented.py --input-dir '${REMOTE_DIR}'" 2>&1 | tee -a "${LOG_FILE}"

printf '%s\n' '[4/4] Copiando CSVs e relatório para o Mac...'
mkdir -p "${LOCAL_DIR}/raspberry_outputs"
"${SCP_BIN}" -q -o BatchMode=yes -r "${PI_TARGET}:${REMOTE_DIR}" "${LOCAL_DIR}/raspberry_outputs/"
LOCAL_REPORT_DIR="${LOCAL_DIR}/raspberry_outputs/$(basename "${REMOTE_DIR}")"
printf '%s\n' 'MICROBENCHMARK PASS'
printf '%s\n' "  CSVs     : ${LOCAL_REPORT_DIR}/*_measurements.csv"
printf '%s\n' "  relatório: ${LOCAL_REPORT_DIR}/analysis_report.md"
printf '%s\n' "  log      : ${LOG_FILE}"
