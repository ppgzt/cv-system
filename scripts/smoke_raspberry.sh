#!/usr/bin/env bash
# Pi-local diagnostic helper only. It does not measure energy, SSH, SCP, or
# coordinate an experiment. The real experiment smoke is controlled from the
# Mac by scripts/smoke_raspberry_from_mac.sh.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
SMOKE_NUM_ANIMALS="${SMOKE_NUM_ANIMALS:-3}"
VERIFY_MODEL_SHA256="${VERIFY_MODEL_SHA256:-1}"
DRY_RUN="${DRY_RUN:-0}"

SELECTOR_MODEL="${PROJECT_ROOT}/infra/models/frame_selector.tflite"
WEIGHT_MODEL="${PROJECT_ROOT}/infra/models/sheep_weight_predictor.tflite"
SELECTOR_SHA256="f0886d0f01a1b48ccb836da7ea139caa58f0e0e445ee27ef2ec2a07abd9adca7"
WEIGHT_SHA256="15b9d310c8deffc4629a107b62e889d13c8fb55186759c595ee7b0c192e50d4a"

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%s\n' "[dry-run] project=${PROJECT_ROOT}"
    printf '%s\n' "[dry-run] ${PYTHON_BIN} mas-main.py --engine pade --mode resource-aware-visual-gated --num-animals ${SMOKE_NUM_ANIMALS}"
    exit 0
fi

cd "${PROJECT_ROOT}"

fail() {
    printf '%s\n' "[smoke] ERROR: $*" >&2
    exit 1
}

[[ -x "${PYTHON_BIN}" ]] || fail "Python/venv not executable: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)' \
    || fail "expected Python 3.13 in ${PYTHON_BIN}"
"${PYTHON_BIN}" -c 'import tensorflow, twisted, pade, psutil, scipy, dotenv' \
    || fail "required Python dependencies are missing from ${PYTHON_BIN}"

[[ -f "${SELECTOR_MODEL}" ]] || fail "missing active selector: ${SELECTOR_MODEL}"
[[ -f "${WEIGHT_MODEL}" ]] || fail "missing active regressor: ${WEIGHT_MODEL}"

if [[ "${VERIFY_MODEL_SHA256}" == "1" ]]; then
    command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is required when VERIFY_MODEL_SHA256=1"
    [[ "$(sha256sum "${SELECTOR_MODEL}" | awk '{print $1}')" == "${SELECTOR_SHA256}" ]] || fail "selector SHA256 mismatch"
    [[ "$(sha256sum "${WEIGHT_MODEL}" | awk '{print $1}')" == "${WEIGHT_SHA256}" ]] || fail "regressor SHA256 mismatch"
fi

command -v vcgencmd >/dev/null 2>&1 || fail "vcgencmd is required on Raspberry Pi"
TEMP_RAW="$(vcgencmd measure_temp 2>&1)" || fail "vcgencmd measure_temp failed: ${TEMP_RAW}"
THROTTLED_RAW="$(vcgencmd get_throttled 2>&1)" || fail "vcgencmd get_throttled failed: ${THROTTLED_RAW}"

RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_ROOT="${PROJECT_ROOT}/results/smoke_raspberry_${RUN_TIMESTAMP}"
RUN_ID="smoke_raspberry_${RUN_TIMESTAMP}"
mkdir -p "${RESULT_ROOT}"
LOG_FILE="${RESULT_ROOT}/run.log"

{
    printf '%s\n' "[smoke] project=${PROJECT_ROOT}"
    printf '%s\n' "[smoke] python=$(${PYTHON_BIN} --version)"
    printf '%s\n' "[smoke] vcgencmd measure_temp: ${TEMP_RAW}"
    printf '%s\n' "[smoke] vcgencmd get_throttled: ${THROTTLED_RAW}"
    printf '%s\n' "[smoke] mode=resource-aware-visual-gated num_animals=${SMOKE_NUM_ANIMALS}"
    printf '%s\n' "[smoke] note: CLI has no per-tag selector; --num-animals uses the dataset's official sorted cohort order."
} | tee "${LOG_FILE}"

set +e
"${PYTHON_BIN}" mas-main.py \
    --engine pade \
    --mode resource-aware-visual-gated \
    --num-animals "${SMOKE_NUM_ANIMALS}" \
    --run-id "${RUN_ID}" \
    --output-dir "${RESULT_ROOT}" \
    2>&1 | tee -a "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}
set -e

if [[ "${STATUS}" -ne 0 ]]; then
    printf '%s\n' "[smoke] FAILED exit_code=${STATUS}; log=${LOG_FILE}" | tee -a "${LOG_FILE}" >&2
    exit "${STATUS}"
fi

printf '%s\n' "SMOKE PASS" | tee -a "${LOG_FILE}"
