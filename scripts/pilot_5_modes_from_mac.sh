#!/usr/bin/env bash
# Mac controller for one full-cohort run of every official PIBIC mode.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
source "${SCRIPT_DIR}/mac_remote_power.sh"

FIXED_FPS="${FIXED_FPS:-5}"
COOLDOWN_MIN_SECONDS="${COOLDOWN_MIN_SECONDS:-180}"
COOL_TEMP_C="${COOL_TEMP_C:-50}"
RECHECK_SECONDS="${RECHECK_SECONDS:-60}"
COOLDOWN_EXPECTED_MAX_SECONDS="${COOLDOWN_EXPECTED_MAX_SECONDS:-600}"
RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PILOT_ROOT="${PILOT_ROOT:-${PROJECT_ROOT}/results/pilot_5_modes_${RUN_TIMESTAMP}}"
PILOT_LOG="${PILOT_ROOT}/pilot.log"
ACTIVE_MODE_DIR=""
ACTIVE_TC66_LOG=""

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    for mode in original-timing fixed-fps visual-adaptive visual-gated resource-aware-visual-gated; do
        extra=""; [[ "${mode}" == fixed-fps ]] && extra=" --fps ${FIXED_FPS}"
        printf '%s\n' "[dry-run] SSH Pi: python mas-main.py --engine pade --mode ${mode}${extra} --run-id <unique> --output-dir <remote-unique>"
    done
    exit 0
fi

trap stop_power_logger_on_exit EXIT INT TERM
mkdir -p "${PILOT_ROOT}"
require_remote_config
validate_mac_tc66
remote_exec true
validate_remote_runtime || runner_fail "remote runtime/models/vcgencmd preflight failed" || exit 1

log_pilot() { printf '%s\n' "$*" | tee -a "${PILOT_LOG}"; }

run_mode() {
    local ordinal="$1" mode="$2" slug="$3" mode_dir run_id remote_base remote_dir log metadata before_temp before_thr after_temp after_thr t0 t1 started status extra=""
    mode_dir="${PILOT_ROOT}/${ordinal}_${slug}"; mkdir -p "${mode_dir}"
    log="${mode_dir}/remote_run.log"; metadata="${mode_dir}/metadata.txt"
    run_id="pilot_${RUN_TIMESTAMP}_${ordinal}_${slug}"
    remote_base="${PI_REMOTE_OUTPUT_BASE:-${PI_PROJECT_ROOT}/results/mac_controlled_runs}/${run_id}"
    remote_dir="${remote_base}/${run_id}"
    before_temp="$(remote_temperature_c)" || return 1
    before_thr="$(remote_throttled)" || return 1
    {
        printf 'mode=%s\nrun_id=%s\nfixed_fps=%s\n' "${mode}" "${run_id}" "$([[ "${mode}" == fixed-fps ]] && printf '%s' "${FIXED_FPS}" || printf n/a)"
        printf 'recorded_before=%s\n' "$(mac_timestamp)"
        printf 'local_git_commit=%s\nremote_git_commit=%s\npi_hostname=%s\n' "$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || printf unknown)" "$(remote_git_commit)" "$(remote_hostname)"
        printf 'temperature_before_c=%s\nthrottled_before=%s\nremote_run_dir=%s\npower_log_path=%s\n' "${before_temp}" "${before_thr}" "${remote_dir}" "${mode_dir}/power.csv"
    } > "${metadata}"
    ACTIVE_MODE_DIR="${mode_dir}"; ACTIVE_TC66_LOG="${mode_dir}/tc66.log"
    printf '%s\n' "[power] logger start=$(mac_timestamp)" | tee -a "${log}"
    start_power_logger "${mode_dir}" "${ACTIVE_TC66_LOG}"
    t0="$(mac_timestamp)"; started="$(date +%s)"; printf '[T0] %s\n' "${t0}" | tee -a "${log}"; printf 't0=%s\n' "${t0}" >> "${metadata}"
    [[ "${mode}" == fixed-fps ]] && extra=" --fps '${FIXED_FPS}'"
    if run_remote_pipeline "cd '${PI_PROJECT_ROOT}' && '${PI_PYTHON}' mas-main.py --engine pade --mode '${mode}'${extra} --run-id '${run_id}' --output-dir '${remote_base}'" "${log}"; then status=0; else status=$?; fi
    t1="$(mac_timestamp)"; printf '[T1] %s exit=%s\n' "${t1}" "${status}" | tee -a "${log}"; printf 't1=%s\nexit_code=%s\nduration_seconds=%s\n' "${t1}" "${status}" "$(( $(date +%s) - started ))" >> "${metadata}"
    printf '%s\n' "[power] logger stop=$(mac_timestamp)" | tee -a "${log}"
    stop_power_logger "${mode_dir}" "${ACTIVE_TC66_LOG}"; ACTIVE_MODE_DIR=""; ACTIVE_TC66_LOG=""
    after_temp="$(remote_temperature_c)" || return 1
    after_thr="$(remote_throttled)" || return 1
    printf 'temperature_after_c=%s\nthrottled_after=%s\n' "${after_temp}" "${after_thr}" >> "${metadata}"
    printf 'recorded_after=%s\n' "$(mac_timestamp)" >> "${metadata}"
    if remote_log_has_internal_error "${log}"; then
        printf '%s\n' "[pilot] internal runtime error detected in ${log}" | tee -a "${log}" >&2
        return 70
    fi
    [[ "${status}" -eq 0 ]] || return "${status}"
    copy_remote_output "${remote_dir}" "${mode_dir}"
    log_pilot "[pilot] completed ${ordinal}_${slug}"
}

log_pilot "[pilot] checking Pi temperature before first run"
wait_for_remote_cooldown 0 "${COOL_TEMP_C}" "${RECHECK_SECONDS}" "${COOLDOWN_EXPECTED_MAX_SECONDS}" 2>&1 | tee -a "${PILOT_LOG}"
run_mode 01 original-timing original_timing || exit $?
wait_for_remote_cooldown "${COOLDOWN_MIN_SECONDS}" "${COOL_TEMP_C}" "${RECHECK_SECONDS}" "${COOLDOWN_EXPECTED_MAX_SECONDS}" 2>&1 | tee -a "${PILOT_LOG}"
run_mode 02 fixed-fps fixed_fps || exit $?
wait_for_remote_cooldown "${COOLDOWN_MIN_SECONDS}" "${COOL_TEMP_C}" "${RECHECK_SECONDS}" "${COOLDOWN_EXPECTED_MAX_SECONDS}" 2>&1 | tee -a "${PILOT_LOG}"
run_mode 03 visual-adaptive visual_adaptive || exit $?
wait_for_remote_cooldown "${COOLDOWN_MIN_SECONDS}" "${COOL_TEMP_C}" "${RECHECK_SECONDS}" "${COOLDOWN_EXPECTED_MAX_SECONDS}" 2>&1 | tee -a "${PILOT_LOG}"
run_mode 04 visual-gated visual_gated || exit $?
wait_for_remote_cooldown "${COOLDOWN_MIN_SECONDS}" "${COOL_TEMP_C}" "${RECHECK_SECONDS}" "${COOLDOWN_EXPECTED_MAX_SECONDS}" 2>&1 | tee -a "${PILOT_LOG}"
run_mode 05 resource-aware-visual-gated resource_aware_visual_gated || exit $?
log_pilot "[pilot] PASS results=${PILOT_ROOT}"
