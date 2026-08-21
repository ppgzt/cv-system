#!/usr/bin/env bash
# Mac controller for the real Pi/TC66C smoke path.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
source "${SCRIPT_DIR}/mac_remote_power.sh"

SMOKE_NUM_ANIMALS="${SMOKE_NUM_ANIMALS:-3}"
RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/smoke_raspberry_from_mac_${RUN_TIMESTAMP}}"
RUN_ID="${RUN_ID:-smoke_raspberry_from_mac_${RUN_TIMESTAMP}}"
REMOTE_OUTPUT_BASE="${PI_REMOTE_OUTPUT_BASE:-<PI_PROJECT_ROOT>/results/mac_controlled_runs/${RUN_ID}}"
MODE_DIR="${RESULT_ROOT}"
LOG_FILE="${MODE_DIR}/remote_run.log"
METADATA="${MODE_DIR}/metadata.txt"
ACTIVE_MODE_DIR="${MODE_DIR}"
ACTIVE_TC66_LOG="${MODE_DIR}/tc66.log"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%s\n' "[dry-run] Mac TC66C -> SSH Pi: resource-aware-visual-gated, num_animals=${SMOKE_NUM_ANIMALS}"
    printf '%s\n' "[dry-run] remote: python mas-main.py --engine pade --mode resource-aware-visual-gated --num-animals ${SMOKE_NUM_ANIMALS} --run-id ${RUN_ID} --output-dir ${REMOTE_OUTPUT_BASE}"
    exit 0
fi

trap stop_power_logger_on_exit EXIT INT TERM
mkdir -p "${MODE_DIR}"
require_remote_config
REMOTE_OUTPUT_BASE="${PI_REMOTE_OUTPUT_BASE:-${PI_PROJECT_ROOT}/results/mac_controlled_runs/${RUN_ID}}"
REMOTE_RUN_DIR="${REMOTE_OUTPUT_BASE}/${RUN_ID}"
validate_mac_tc66
remote_exec true
validate_remote_runtime || runner_fail "remote runtime/models/vcgencmd preflight failed" || exit 1
TEMP_BEFORE="$(remote_temperature_c)" || exit 1
THROTTLED_BEFORE="$(remote_throttled)" || exit 1

{
    printf 'mode=resource-aware-visual-gated\nrun_id=%s\n' "${RUN_ID}"
    printf 'recorded_before=%s\n' "$(mac_timestamp)"
    printf 'local_git_commit=%s\nremote_git_commit=%s\npi_hostname=%s\n' "$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || printf unknown)" "$(remote_git_commit)" "$(remote_hostname)"
    printf 'temperature_before_c=%s\nthrottled_before=%s\nremote_run_dir=%s\n' "${TEMP_BEFORE}" "${THROTTLED_BEFORE}" "${REMOTE_RUN_DIR}"
    printf 'power_log_path=%s\n' "${MODE_DIR}/power.csv"
} > "${METADATA}"

printf '%s\n' "[power] logger start=$(mac_timestamp)" | tee -a "${LOG_FILE}"
start_power_logger "${MODE_DIR}" "${ACTIVE_TC66_LOG}"
STARTED_SECONDS="$(date +%s)"
T0="$(mac_timestamp)"; printf '[T0] %s\n' "${T0}" | tee -a "${LOG_FILE}"; printf 't0=%s\n' "${T0}" >> "${METADATA}"
REMOTE_CMD="cd '${PI_PROJECT_ROOT}' && '${PI_PYTHON}' mas-main.py --engine pade --mode resource-aware-visual-gated --num-animals '${SMOKE_NUM_ANIMALS}' --run-id '${RUN_ID}' --output-dir '${REMOTE_OUTPUT_BASE}'"
if run_remote_pipeline "${REMOTE_CMD}" "${LOG_FILE}"; then STATUS=0; else STATUS=$?; fi
T1="$(mac_timestamp)"; printf '[T1] %s exit=%s\n' "${T1}" "${STATUS}" | tee -a "${LOG_FILE}"; printf 't1=%s\nexit_code=%s\n' "${T1}" "${STATUS}" >> "${METADATA}"
printf '%s\n' "[power] logger stop=$(mac_timestamp)" | tee -a "${LOG_FILE}"
stop_power_logger "${MODE_DIR}" "${ACTIVE_TC66_LOG}"
TEMP_AFTER="$(remote_temperature_c)" || exit 1
THROTTLED_AFTER="$(remote_throttled)" || exit 1
printf 'temperature_after_c=%s\nthrottled_after=%s\nduration_seconds=%s\n' "${TEMP_AFTER}" "${THROTTLED_AFTER}" "$(( $(date +%s) - STARTED_SECONDS ))" >> "${METADATA}"
printf 'recorded_after=%s\n' "$(mac_timestamp)" >> "${METADATA}"
if remote_log_has_internal_error "${LOG_FILE}"; then
    printf '%s\n' "[smoke] internal runtime error detected in ${LOG_FILE}" | tee -a "${LOG_FILE}" >&2
    printf '%s\n' "SMOKE FAIL" | tee -a "${LOG_FILE}" >&2
    exit 70
fi
if [[ "${STATUS}" -ne 0 ]]; then
    printf '%s\n' "SMOKE FAIL" | tee -a "${LOG_FILE}" >&2
    exit "${STATUS}"
fi
copy_remote_output "${REMOTE_RUN_DIR}" "${MODE_DIR}"
printf '%s\n' "SMOKE PASS results=${MODE_DIR}"
