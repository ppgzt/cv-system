#!/usr/bin/env bash
# ==============================================================================
# run_pibic_recon_native.sh — reconhecimento físico PADE via Mac + TC66C + SSH
#
# Fluxo: preflight descartável (1 passagem), warm-up descartável, REPS runs
# válidas. O pipeline roda no Raspberry; energia e cópia dos resultados ficam
# no Mac por meio de run_power_test.sh.
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPS="${REPS:-1}"
COOLDOWN_MIN="${COOLDOWN_MIN:-8}"
DO_PREFLIGHT="${DO_PREFLIGHT:-1}"
DO_WARMUP="${DO_WARMUP:-1}"
MODE="${MODE:-mas-single}"
# Vazio = todas as passagens disponíveis no Raspberry.
NUM_ANIMALS="${NUM_ANIMALS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

RECON_TS="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="${WORK_DIR:-./recon_runs/recon_${MODE}_pade_native_${RECON_TS}}"
MANIFEST="${WORK_DIR}/manifest.tsv"
REQUIRED_REPORT_FILES=(
    metrics.json report.md debug.log cpu.csv mem.csv temp.csv
    queue_telemetry.csv hardware_telemetry.csv capture_timing.csv
)
REQUIRED_MAC_FILES=(power.csv tc66.log pipeline.log)

if [[ "$MODE" != "mas-single" ]]; then
    echo "[ERROR] Este piloto é restrito a MODE=mas-single (recebido: $MODE)." >&2
    exit 2
fi
if ! [[ "$REPS" =~ ^[1-9][0-9]*$ ]] || ! [[ "$COOLDOWN_MIN" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] REPS deve ser >= 1 e COOLDOWN_MIN deve ser >= 0." >&2
    exit 2
fi
if [[ "$DO_PREFLIGHT" != "0" && "$DO_PREFLIGHT" != "1" ]] || \
   [[ "$DO_WARMUP" != "0" && "$DO_WARMUP" != "1" ]]; then
    echo "[ERROR] DO_PREFLIGHT e DO_WARMUP devem ser 0 ou 1." >&2
    exit 2
fi
if [[ -n "$NUM_ANIMALS" && ! "$NUM_ANIMALS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] NUM_ANIMALS deve ser vazio ou inteiro >= 1." >&2
    exit 2
fi

EXTRA_ARGV=()
if [[ -n "$EXTRA_ARGS" ]]; then
    read -r -a EXTRA_ARGV <<< "$EXTRA_ARGS"
fi
if (( ${#EXTRA_ARGV[@]} > 0 )); then
    for extra_arg in "${EXTRA_ARGV[@]}"; do
        case "$extra_arg" in
            --engine|--engine=*|--native-timestamps|--debug|--num-animals|--num-animals=*)
                echo "[ERROR] EXTRA_ARGS não pode alterar PADE, timing, debug ou NUM_ANIMALS." >&2
                exit 2
                ;;
        esac
    done
fi

COOLDOWN_SEC=$((COOLDOWN_MIN * 60))
mkdir -p "$WORK_DIR"
printf 'kind\trep\tstatus\texit_code\toutput_dir\treport_dir\n' > "$MANIFEST"

countdown() {
    local seconds=$1
    while (( seconds > 0 )); do
        printf '\r[COOLDOWN] %02dm %02ds restantes...   ' \
            "$((seconds / 60))" "$((seconds % 60))"
        sleep 1
        ((seconds -= 1))
    done
    printf '\r[COOLDOWN] concluído.                         \n'
}

find_copied_report() {
    local output_dir=$1
    local -a reports=()
    local path

    while IFS= read -r path; do
        [[ -n "$path" ]] && reports+=("$path")
    done < <(find "$output_dir" -mindepth 1 -maxdepth 1 -type d \
        -name "${MODE}_pade_*" -print)

    if (( ${#reports[@]} != 1 )); then
        echo "[ERROR] Esperado exatamente um report copiado ${MODE}_pade_*; encontrados ${#reports[@]}." >&2
        return 1
    fi
    printf '%s\n' "${reports[0]}"
}

validate_artifacts() {
    local output_dir=$1
    local report_dir=$2
    local missing=0
    local filename

    echo "[CHECK] Resultado Mac: $output_dir"
    echo "[CHECK] Report Pi:      $report_dir"
    for filename in "${REQUIRED_MAC_FILES[@]}"; do
        if [[ -f "$output_dir/$filename" ]]; then
            printf '  [OK] %s\n' "$filename"
        else
            printf '  [AUSENTE] %s\n' "$filename" >&2
            ((missing += 1))
        fi
    done
    for filename in "${REQUIRED_REPORT_FILES[@]}"; do
        if [[ -f "$report_dir/$filename" ]]; then
            printf '  [OK] %s\n' "$filename"
        else
            printf '  [AUSENTE] %s\n' "$filename" >&2
            ((missing += 1))
        fi
    done
    (( missing == 0 ))
}

run_pipeline() {
    local kind=$1
    local requested_animals=$2
    local output_dir="${WORK_DIR}/${MODE}_native_${kind}"
    local runner_extra="--debug"
    local exit_code
    local report_dir=""

    if (( ${#EXTRA_ARGV[@]} > 0 )); then
        runner_extra+=" ${EXTRA_ARGS}"
    fi

    echo "[RUN] $kind → PADE/Original-Timing no Raspberry via SSH"
    ENGINE=pade \
    NATIVE_TIMESTAMPS=1 \
    RUN_TAG="$kind" \
    WORK_DIR="$WORK_DIR" \
    NUM_ANIMALS="$requested_animals" \
    EXTRA_ARGS="$runner_extra" \
        ./run_power_test.sh
    exit_code=$?

    RUN_EXIT_CODE=$exit_code
    RUN_OUTPUT_DIR=$output_dir
    RUN_REPORT_DIR=""

    if [[ -d "$output_dir" ]]; then
        if report_dir="$(find_copied_report "$output_dir")"; then
            RUN_REPORT_DIR=$report_dir
        fi
    fi

    if [[ "$exit_code" -eq 0 && -n "$RUN_REPORT_DIR" ]] && \
       validate_artifacts "$output_dir" "$RUN_REPORT_DIR"; then
        return 0
    fi
    return 1
}

record_manifest() {
    local kind=$1
    local rep=$2
    local status=$3
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$kind" "$rep" "$status" "$RUN_EXIT_CODE" \
        "$RUN_OUTPUT_DIR" "${RUN_REPORT_DIR:-}" >> "$MANIFEST"
}

discard_successful_run() {
    rm -rf "$RUN_OUTPUT_DIR"
}

echo "=========================================================="
echo " PIBIC — RECONHECIMENTO FÍSICO PADE / ORIGINAL-TIMING"
echo "=========================================================="
echo "Orquestrador     : Mac + TC66C"
echo "Pipeline         : Raspberry via SSH"
echo "Mode / Engine    : $MODE / pade"
echo "Schedule / Debug : native timestamps / enabled"
echo "Passagens        : ${NUM_ANIMALS:-todas disponíveis no Raspberry}"
echo "Preflight        : $DO_PREFLIGHT"
echo "Warm-up          : $DO_WARMUP"
echo "Runs válidas     : $REPS"
echo "Cooldown         : ${COOLDOWN_MIN} min"
echo "Work dir         : $WORK_DIR"
echo "=========================================================="

if [[ "$DO_PREFLIGHT" == "1" ]]; then
    if run_pipeline "preflight" "1"; then
        record_manifest "preflight" "-" "DISCARDED_OK"
        discard_successful_run
        echo "[PREFLIGHT] OK e descartado."
    else
        record_manifest "preflight" "-" "FAILED"
        echo "[PREFLIGHT] FALHOU; reconhecimento abortado." >&2
        exit 1
    fi
fi

if [[ "$DO_WARMUP" == "1" ]]; then
    if run_pipeline "warmup" "$NUM_ANIMALS"; then
        record_manifest "warmup" "-" "DISCARDED_OK"
        discard_successful_run
        echo "[WARM-UP] OK e descartado."
    else
        record_manifest "warmup" "-" "FAILED"
        echo "[WARM-UP] FALHOU; reconhecimento abortado." >&2
        exit 1
    fi
    (( COOLDOWN_SEC > 0 )) && countdown "$COOLDOWN_SEC"
fi

failed_runs=0
for ((rep = 1; rep <= REPS; rep++)); do
    kind="recon_r${rep}"
    if run_pipeline "$kind" "$NUM_ANIMALS"; then
        record_manifest "$kind" "$rep" "OK"
        echo "[RECON] OK: $RUN_REPORT_DIR"
    else
        record_manifest "$kind" "$rep" "FAILED"
        echo "[RECON] FALHOU; continuando para a próxima repetição." >&2
        ((failed_runs += 1))
    fi
    (( rep < REPS && COOLDOWN_SEC > 0 )) && countdown "$COOLDOWN_SEC"
done

echo "=========================================================="
echo " RECONHECIMENTO CONCLUÍDO — falhas: $failed_runs"
echo " Manifest: $MANIFEST"
cat "$MANIFEST"
echo "=========================================================="

(( failed_runs == 0 ))
