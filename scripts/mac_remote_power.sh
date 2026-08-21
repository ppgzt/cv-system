#!/usr/bin/env bash
# Shared mechanics for the Mac-controlled PIBIC experiment runners.
# The Pi only executes mas-main.py; this file owns neither scientific policy nor
# dataset selection.

MAC_RUNNER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SSH_BIN="${SSH_BIN:-ssh}"
SCP_BIN="${SCP_BIN:-scp}"
MAC_PYTHON="${MAC_PYTHON:-${MAC_RUNNER_ROOT}/.venv/bin/python}"
TC66_SCRIPT="${TC66_SCRIPT:-${MAC_RUNNER_ROOT}/TC66C.py}"
TC66_PORT="${TC66_PORT:-}"
TC66_INTERVAL="${TC66_INTERVAL:-1}"
VERIFY_MODEL_SHA256="${VERIFY_MODEL_SHA256:-1}"
SELECTOR_SHA256="f0886d0f01a1b48ccb836da7ea139caa58f0e0e445ee27ef2ec2a07abd9adca7"
WEIGHT_SHA256="15b9d310c8deffc4629a107b62e889d13c8fb55186759c595ee7b0c192e50d4a"
TC66_PID=""

mac_timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

runner_fail() {
    printf '%s\n' "[mac-runner] ERROR: $*" >&2
    return 1
}

require_remote_config() {
    : "${PI_HOST:?set PI_HOST (or define it in the environment)}"
    : "${PI_USER:?set PI_USER (or define it in the environment)}"
    : "${PI_PROJECT_ROOT:?set PI_PROJECT_ROOT to the absolute project path on the Pi}"
    PI_PYTHON="${PI_PYTHON:-${PI_PROJECT_ROOT}/.venv/bin/python}"
    PI_TARGET="${PI_TARGET:-${PI_USER}@${PI_HOST}}"
    case "${PI_PROJECT_ROOT}${PI_PYTHON}${PI_TARGET}" in
        *"'"*|*$'\n'*|*$'\r'*) runner_fail "remote variables cannot contain quotes or newlines" ;;
    esac
}

remote_exec() {
    "${SSH_BIN}" -o BatchMode=yes -o ConnectTimeout=10 "${PI_TARGET}" "$@"
}

parse_temperature_c() {
    local raw="$1"
    if [[ "${raw}" =~ temp=([0-9]+([.][0-9]+)?) ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    printf '%s\n' "[temperature] unparseable vcgencmd value: ${raw}" >&2
    return 1
}

remote_temperature_c() {
    local raw
    raw="$(remote_exec "vcgencmd measure_temp" 2>&1)" || {
        printf '%s\n' "[temperature] remote measure_temp failed: ${raw}" >&2
        return 1
    }
    parse_temperature_c "${raw}"
}

remote_throttled() {
    local raw
    raw="$(remote_exec "vcgencmd get_throttled" 2>&1)" || {
        printf '%s\n' "[throttling] remote get_throttled failed: ${raw}" >&2
        return 1
    }
    printf '%s\n' "${raw}"
}

remote_git_commit() {
    remote_exec "git -C '${PI_PROJECT_ROOT}' rev-parse HEAD" 2>/dev/null || printf '%s\n' unknown
}

remote_hostname() {
    remote_exec hostname 2>/dev/null || printf '%s\n' unknown
}

validate_mac_tc66() {
    [[ -x "${MAC_PYTHON}" ]] || runner_fail "Mac Python is not executable: ${MAC_PYTHON}" || return
    [[ -f "${TC66_SCRIPT}" ]] || runner_fail "TC66C.py not found: ${TC66_SCRIPT}" || return
    [[ -n "${TC66_PORT}" ]] || runner_fail "set TC66_PORT (for example /dev/cu.usbmodemTC661)" || return
    [[ -e "${TC66_PORT}" ]] || runner_fail "TC66C serial port not found: ${TC66_PORT}" || return
    "${MAC_PYTHON}" -c 'import serial; from Crypto.Cipher import AES' \
        || runner_fail "Mac Python lacks TC66C dependencies (pyserial/pycryptodome)"
}

validate_remote_runtime() {
    local hash_checks=""
    if [[ "${VERIFY_MODEL_SHA256}" == "1" ]]; then
        hash_checks=" && command -v sha256sum >/dev/null && [ \"\$(sha256sum infra/models/frame_selector.tflite | awk '{print \$1}')\" = '${SELECTOR_SHA256}' ] && [ \"\$(sha256sum infra/models/sheep_weight_predictor.tflite | awk '{print \$1}')\" = '${WEIGHT_SHA256}' ]"
    fi
    remote_exec "cd '${PI_PROJECT_ROOT}' && test -x '${PI_PYTHON}' && command -v vcgencmd >/dev/null && test -f infra/models/frame_selector.tflite && test -f infra/models/sheep_weight_predictor.tflite${hash_checks}"
}

start_power_logger() {
    local mode_dir="$1" tc66_log="$2"
    "${MAC_PYTHON}" -u "${TC66_SCRIPT}" "${TC66_PORT}" "${mode_dir}/power.csv" -t "${TC66_INTERVAL}" >"${tc66_log}" 2>&1 &
    TC66_PID=$!
    local ready=0
    for _ in $(seq 1 20); do
        if ! kill -0 "${TC66_PID}" 2>/dev/null; then
            cat "${tc66_log}" >&2 || true
            runner_fail "TC66C logger exited before its first sample"
            return 1
        fi
        if grep -Eq '^[0-9]{4}-[0-9]{2}-[0-9]{2}T.*,.*,.*,.*,' "${tc66_log}" 2>/dev/null; then
            ready=1
            break
        fi
        sleep 0.5
    done
    [[ "${ready}" -eq 1 ]] || { runner_fail "TC66C logger produced no sample within 10 seconds"; return 1; }
}

stop_power_logger() {
    local mode_dir="$1" tc66_log="$2"
    [[ -n "${TC66_PID}" ]] || return 0
    kill -INT "${TC66_PID}" 2>/dev/null || true
    for _ in $(seq 1 40); do
        kill -0 "${TC66_PID}" 2>/dev/null || break
        sleep 0.2
    done
    kill -0 "${TC66_PID}" 2>/dev/null && kill -TERM "${TC66_PID}" 2>/dev/null || true
    wait "${TC66_PID}" 2>/dev/null || true
    TC66_PID=""
    awk 'BEGIN{print "Datetime,Time[S],Volt[V],Current[A],Power[W]"}
         /^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:,.-]+,[^,]+,[^,]+,[^,]+,[^,]+/' \
        "${tc66_log}" > "${mode_dir}/power.csv"
}

stop_power_logger_on_exit() {
    local exit_status="$?"
    if [[ -n "${TC66_PID}" && -n "${ACTIVE_MODE_DIR:-}" && -n "${ACTIVE_TC66_LOG:-}" ]]; then
        stop_power_logger "${ACTIVE_MODE_DIR}" "${ACTIVE_TC66_LOG}"
    fi
    exit "${exit_status}"
}

run_remote_pipeline() {
    local remote_command="$1" log_file="$2" status
    set +e
    remote_exec "${remote_command}" 2>&1 | tee -a "${log_file}"
    status=${PIPESTATUS[0]}
    set -e
    return "${status}"
}

copy_remote_output() {
    local remote_run_dir="$1" mode_dir="$2"
    mkdir -p "${mode_dir}/raspberry_outputs"
    "${SCP_BIN}" -q -o BatchMode=yes -r "${PI_TARGET}:${remote_run_dir}" "${mode_dir}/raspberry_outputs/"
}

is_cool() {
    awk -v temp="$1" -v limit="$2" 'BEGIN { exit !(temp < limit) }'
}

wait_for_remote_cooldown() {
    local mandatory_seconds="$1" cool_temp="$2" recheck_seconds="$3" expected_max_seconds="$4"
    local started="${SECONDS}" elapsed temp warned=0
    if [[ "${mandatory_seconds}" -gt 0 ]]; then
        printf '%s\n' "[cooldown] waiting mandatory ${mandatory_seconds} s"
        sleep "${mandatory_seconds}"
    fi
    while true; do
        temp="$(remote_temperature_c)" || return 1
        elapsed=$((SECONDS - started))
        printf '%s\n' "[cooldown] temp=${temp} C after ${elapsed} s"
        if is_cool "${temp}" "${cool_temp}"; then
            printf '%s\n' "[cooldown] ready after ${elapsed} s"
            return 0
        fi
        if [[ "${elapsed}" -ge "${expected_max_seconds}" && "${warned}" -eq 0 ]]; then
            printf '%s\n' "[cooldown] WARNING: still >= ${cool_temp} C after expected ${expected_max_seconds} s; continuing until cool" >&2
            warned=1
        fi
        printf '%s\n' "[cooldown] temp >= ${cool_temp} C; rechecking in ${recheck_seconds} s"
        sleep "${recheck_seconds}"
    done
}
