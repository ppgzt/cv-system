#!/usr/bin/env bash
# Deterministic no-hardware checks for the Mac orchestration scripts.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT}/scripts/mac_remote_power.sh"
[[ "$(parse_temperature_c "temp=47.8'C")" == "47.8" ]]
! parse_temperature_c "temperature unavailable" >/dev/null 2>&1

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
mkdir -p "${tmp}/bin"
: > "${tmp}/tc66-port"

cat > "${tmp}/bin/python" <<'EOF'
#!/usr/bin/env bash
[[ "$1" == -c ]] && exit 0
[[ "${MOCK_LOGGER_FAIL:-0}" == 1 ]] && exit 11
trap 'exit 0' INT TERM
while true; do echo '2026-08-21T00:00:00Z,0.0,5.0,1.0,5.0'; sleep 1; done
EOF
cat > "${tmp}/bin/ssh" <<'EOF'
#!/usr/bin/env bash
command="${@: -1}"
[[ "${MOCK_SSH_FAIL:-0}" == 1 ]] && exit 23
case "${command}" in
  *measure_temp*) echo "temp=${MOCK_TEMP:-47.8}'C" ;;
  *get_throttled*) echo 'throttled=0x0' ;;
  *rev-parse*) echo deadbeef ;;
  hostname) echo mock-pi ;;
  *mas-main.py*) exit "${MOCK_REMOTE_EXIT:-0}" ;;
  *) exit 0 ;;
esac
EOF
cat > "${tmp}/bin/scp" <<'EOF'
#!/usr/bin/env bash
exit "${MOCK_SCP_EXIT:-0}"
EOF
chmod +x "${tmp}/bin/"*

env_base=(PI_HOST=mock PI_USER=pi PI_PROJECT_ROOT=/srv/cv-system PI_PYTHON=/srv/cv-system/.venv/bin/python TC66_PORT="${tmp}/tc66-port" MAC_PYTHON="${tmp}/bin/python" TC66_SCRIPT="${ROOT}/TC66C.py" SSH_BIN="${tmp}/bin/ssh" SCP_BIN="${tmp}/bin/scp" VERIFY_MODEL_SHA256=0)

smoke_dry_run="$(DRY_RUN=1 "${ROOT}/scripts/smoke_raspberry_from_mac.sh")"
[[ "${smoke_dry_run}" == *resource-aware-visual-gated* ]]
pilot_dry_run="$(DRY_RUN=1 FIXED_FPS=5 "${ROOT}/scripts/pilot_5_modes_from_mac.sh")"
[[ "${pilot_dry_run}" == *"fixed-fps --fps 5"* ]]
env "${env_base[@]}" RESULT_ROOT="${tmp}/success" "${ROOT}/scripts/smoke_raspberry_from_mac.sh" >/dev/null
[[ -f "${tmp}/success/power.csv" && -f "${tmp}/success/metadata.txt" ]]

if env "${env_base[@]}" MOCK_SSH_FAIL=1 RESULT_ROOT="${tmp}/ssh-fail" "${ROOT}/scripts/smoke_raspberry_from_mac.sh" >/dev/null 2>&1; then exit 1; fi
if env "${env_base[@]}" MOCK_REMOTE_EXIT=7 RESULT_ROOT="${tmp}/remote-fail" "${ROOT}/scripts/smoke_raspberry_from_mac.sh" >/dev/null 2>&1; then exit 1; fi
if env "${env_base[@]}" MOCK_SCP_EXIT=9 RESULT_ROOT="${tmp}/scp-fail" "${ROOT}/scripts/smoke_raspberry_from_mac.sh" >/dev/null 2>&1; then exit 1; fi
if env "${env_base[@]}" MOCK_LOGGER_FAIL=1 RESULT_ROOT="${tmp}/logger-fail" "${ROOT}/scripts/smoke_raspberry_from_mac.sh" >/dev/null 2>&1; then exit 1; fi

printf '%s\n' 'mac experiment runner mocks: PASS'
