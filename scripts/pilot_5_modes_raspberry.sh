#!/usr/bin/env bash
# One ordered pilot run per official PIBIC mode, with mandatory Pi cooldown.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
FIXED_FPS="${FIXED_FPS:-5}"
COOLDOWN_MIN_SECONDS="${COOLDOWN_MIN_SECONDS:-180}"
COOL_TEMP_C="${COOL_TEMP_C:-50}"
RECHECK_SECONDS="${RECHECK_SECONDS:-60}"
COOLDOWN_EXPECTED_MAX_SECONDS="${COOLDOWN_EXPECTED_MAX_SECONDS:-600}"
VERIFY_MODEL_SHA256="${VERIFY_MODEL_SHA256:-1}"
DRY_RUN="${DRY_RUN:-0}"

SELECTOR_MODEL="${PROJECT_ROOT}/infra/models/frame_selector.tflite"
WEIGHT_MODEL="${PROJECT_ROOT}/infra/models/sheep_weight_predictor.tflite"
SELECTOR_SHA256="f0886d0f01a1b48ccb836da7ea139caa58f0e0e445ee27ef2ec2a07abd9adca7"
WEIGHT_SHA256="15b9d310c8deffc4629a107b62e889d13c8fb55186759c595ee7b0c192e50d4a"

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%s\n' "[dry-run] project=${PROJECT_ROOT}; FIXED_FPS=${FIXED_FPS}"
    for mode in original-timing fixed-fps visual-adaptive visual-gated resource-aware-visual-gated; do
        printf '%s\n' "[dry-run] ${PYTHON_BIN} mas-main.py --engine pade --mode ${mode}"
    done
    exit 0
fi

cd "${PROJECT_ROOT}"

fail() {
    printf '%s\n' "[pilot] ERROR: $*" >&2
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

read_temperature_c() {
    local raw
    raw="$(vcgencmd measure_temp 2>&1)" || {
        printf '%s\n' "[temperature] command failed: ${raw}" >&2
        return 1
    }
    if [[ "${raw}" =~ ^temp=([0-9]+([.][0-9]+)?) ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    printf '%s\n' "[temperature] unparseable raw value: ${raw}" >&2
    return 1
}

is_cool() {
    awk -v temp="$1" -v limit="${COOL_TEMP_C}" 'BEGIN { exit !(temp < limit) }'
}

wait_for_cooldown() {
    local started elapsed temp warned=0
    started="${SECONDS}"
    log_pilot "[cooldown] waiting mandatory ${COOLDOWN_MIN_SECONDS} s"
    sleep "${COOLDOWN_MIN_SECONDS}"
    while true; do
        temp="$(read_temperature_c)" || return 1
        elapsed=$((SECONDS - started))
        log_pilot "[cooldown] temp=${temp} C after ${elapsed} s"
        if is_cool "${temp}"; then
            log_pilot "[cooldown] ready after ${elapsed} s"
            return 0
        fi
        if [[ "${elapsed}" -ge "${COOLDOWN_EXPECTED_MAX_SECONDS}" && "${warned}" -eq 0 ]]; then
            log_pilot "[cooldown] WARNING: still >= ${COOL_TEMP_C} C after expected ${COOLDOWN_EXPECTED_MAX_SECONDS} s; continuing until actually cool"
            warned=1
        fi
        log_pilot "[cooldown] temp >= ${COOL_TEMP_C} C; rechecking in ${RECHECK_SECONDS} s"
        sleep "${RECHECK_SECONDS}"
    done
}

RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PILOT_ROOT="${PROJECT_ROOT}/results/pilot_5_modes_${RUN_TIMESTAMP}"
mkdir -p "${PILOT_ROOT}"
PILOT_LOG="${PILOT_ROOT}/pilot.log"

log_pilot() {
    printf '%s\n' "$*" | tee -a "${PILOT_LOG}"
}

record_before() {
    local mode="$1" log="$2" temp throttled commit host
    temp="$(read_temperature_c)" || return 1
    throttled="$(vcgencmd get_throttled 2>&1)" || return 1
    commit="$(git rev-parse HEAD 2>/dev/null || printf unknown)"
    host="$(hostname)"
    {
        printf '%s\n' "[before] timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '%s\n' "[before] mode=${mode}"
        printf '%s\n' "[before] temperature=${temp} C"
        printf '%s\n' "[before] throttled=${throttled}"
        printf '%s\n' "[before] git_commit=${commit}"
        printf '%s\n' "[before] hostname=${host}"
    } | tee -a "${log}"
}

run_mode() {
    local ordinal="$1" mode="$2" slug="$3" mode_dir log run_id started status ended temp throttled
    mode_dir="${PILOT_ROOT}/${ordinal}_${slug}"
    mkdir -p "${mode_dir}"
    log="${mode_dir}/run.log"
    run_id="pilot_${RUN_TIMESTAMP}_${ordinal}_${slug}"
    record_before "${mode}" "${log}" || fail "cannot record pre-run temperature/throttling for ${mode}"
    started="${SECONDS}"
    local -a command=("${PYTHON_BIN}" mas-main.py --engine pade --mode "${mode}" --run-id "${run_id}" --output-dir "${mode_dir}")
    if [[ "${mode}" == "fixed-fps" ]]; then
        command+=(--fps "${FIXED_FPS}")
    fi
    printf '[run] command=' | tee -a "${log}"
    printf '%q ' "${command[@]}" | tee -a "${log}"
    printf '\n' | tee -a "${log}"
    set +e
    "${command[@]}" 2>&1 | tee -a "${log}"
    status=${PIPESTATUS[0]}
    set -e
    ended=$((SECONDS - started))
    temp="$(read_temperature_c)" || fail "cannot read post-run temperature for ${mode}"
    throttled="$(vcgencmd get_throttled 2>&1)" || fail "cannot read post-run throttling for ${mode}"
    {
        printf '%s\n' "[after] exit_code=${status}"
        printf '%s\n' "[after] duration_seconds=${ended}"
        printf '%s\n' "[after] temperature=${temp} C"
        printf '%s\n' "[after] throttled=${throttled}"
    } | tee -a "${log}"
    [[ "${status}" -eq 0 ]] || return "${status}"
}

run_mode 01 original-timing original_timing || exit $?
wait_for_cooldown || fail "cooldown temperature check failed"
run_mode 02 fixed-fps fixed_fps || exit $?
wait_for_cooldown || fail "cooldown temperature check failed"
run_mode 03 visual-adaptive visual_adaptive || exit $?
wait_for_cooldown || fail "cooldown temperature check failed"
run_mode 04 visual-gated visual_gated || exit $?
wait_for_cooldown || fail "cooldown temperature check failed"
run_mode 05 resource-aware-visual-gated resource_aware_visual_gated || exit $?

printf '%s\n' "[pilot] PASS results=${PILOT_ROOT}"
