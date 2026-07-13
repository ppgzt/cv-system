#!/usr/bin/env bash
# ==============================================================================
# run_power_battery_native.sh — BATERIA de timestamps originais com potência.
# PIBIC - CV System
# ------------------------------------------------------------------------------
# Baseado em run_power_battery.sh. Executa:
#   1) preflight descartável de 1 animal (opcional);
#   2) 1 warm-up COMPLETO descartável;
#   3) 5 execuções COMPLETAS válidas;
#   4) cooldown padrão de 8 minutos entre execuções completas.
#
# Cada execução usa NATIVE_TIMESTAMPS=1, portanto o Raspberry executa:
#   mas-main.py <MODE> --native-timestamps
#
# RODA NO MAC. O TC66C fica no Mac, o pipeline roda no Raspberry via SSH e cada
# run puxa imediatamente seus relatórios para WORK_DIR via scp.
#
# Uso recomendado (mantém o Mac acordado):
#   caffeinate -dimsu ./run_power_battery_native.sh
#
# Ajustes úteis:
#   NUM_ANIMALS=20 ./run_power_battery_native.sh
#   EXTRA_ARGS="" ./run_power_battery_native.sh
#   DO_PREFLIGHT=0 COOLDOWN_MIN=8 ./run_power_battery_native.sh
# ============================================================================
set -uo pipefail

# ============================== CONFIG =======================================
REPS="${REPS:-5}"                       # execuções válidas
COOLDOWN_MIN="${COOLDOWN_MIN:-8}"       # cooldown entre execuções completas
COOLDOWN_SEC=$((COOLDOWN_MIN * 60))

MODE="${MODE:-mas-single}"              # mas-single | mas-batch
NUM_ANIMALS="${NUM_ANIMALS:-}"          # vazio = todos os animais disponíveis
EXTRA_ARGS="${EXTRA_ARGS:---debug}"

# Preflight curto: valida TC66C + SSH + pipeline sem consumir uma execução completa.
DO_PREFLIGHT="${DO_PREFLIGHT:-1}"

# Warm-up COMPLETO descartável, antes das 5 execuções válidas.
DO_WARMUP="${DO_WARMUP:-1}"

BATTERY_TS="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="${WORK_DIR:-./power_runs/battery_${MODE}_native_${BATTERY_TS}}"
MANIFEST="${WORK_DIR}/_manifest.txt"
export WORK_DIR MODE NUM_ANIMALS EXTRA_ARGS
# ============================================================================== 

countdown() {
    local secs=$1
    while [ "$secs" -gt 0 ]; do
        local mins=$((secs / 60))
        local s=$((secs % 60))
        printf "\r[Aguardando] Próxima run em: %02dm %02ds...   " "$mins" "$s"
        sleep 1
        secs=$((secs - 1))
    done
    printf "\r[Pronto] Iniciando agora!                                \n"
}

# Roda uma execução nativa e registra no manifest. $1=rep $2=RUN_TAG
run_one() {
    local rep=$1 tag=$2
    local out="${WORK_DIR}/${MODE}_native_${tag}"
    echo "----------------------------------------------------------"
    echo "[RUN] ${MODE} @ native timestamps (rep ${rep}) -> ${out}"
    echo "----------------------------------------------------------"
    NATIVE_TIMESTAMPS=1 RUN_TAG="$tag" ./run_power_test.sh
    local st=$?
    if [ "$st" -eq 0 ] && [ -d "$out" ]; then
        echo "[RUN] OK  -> ${out}"
        printf 'native\t%s\tOK\t%s\n' "$rep" "$out" >> "$MANIFEST"
    else
        echo "[RUN] FALHA (exit=${st}) — veja os logs em ${out:-(sem pasta)}"
        printf 'native\t%s\tFALHA(%s)\t%s\n' "$rep" "$st" "${out:-}" >> "$MANIFEST"
    fi
    return 0                         # continue-on-error, como na bateria original
}

mkdir -p "$WORK_DIR"
: > "$MANIFEST"

echo "=========================================================="
echo "   BATERIA DE POTÊNCIA — NATIVE TIMESTAMPS"
echo "=========================================================="
echo "Modo             : $MODE"
echo "Execuções válidas: $REPS"
echo "Cooldown/run     : $COOLDOWN_MIN min"
echo "Warm-up completo : $([ "$DO_WARMUP" = 1 ] && echo sim || echo não)"
echo "Preflight        : $([ "$DO_PREFLIGHT" = 1 ] && echo sim\ (1\ animal) || echo não)"
echo "Num animais      : ${NUM_ANIMALS:-todos os disponíveis no Raspberry}"
echo "Pasta            : $WORK_DIR"
echo "=========================================================="
echo

# --- Preflight descartável ---------------------------------------------------
if [ "$DO_PREFLIGHT" = 1 ]; then
    PREFLIGHT_LOG="${WORK_DIR}/_preflight.log"
    PREFLIGHT_OUT="${WORK_DIR}/${MODE}_native__preflight"
    echo "[CHECK] Validando voltímetro + SSH + pipeline nativo com 1 animal..."
    NATIVE_TIMESTAMPS=1 RUN_TAG="_preflight" NUM_ANIMALS=1 EXTRA_ARGS="" \
        ./run_power_test.sh >"$PREFLIGHT_LOG" 2>&1
    PF_ST=$?
    if [ "$PF_ST" -ne 0 ] || [ ! -d "$PREFLIGHT_OUT" ]; then
        echo "[CHECK] FALHOU (exit=${PF_ST}) — log do preflight:"
        sed 's/^/    /' "$PREFLIGHT_LOG"
        echo "        (TC66C desconectado? Raspberry off? SSH sem chave? pipeline travou?)"
        exit 1
    fi
    rm -rf "$PREFLIGHT_OUT" "$PREFLIGHT_LOG"
    echo "[CHECK] OK — voltímetro, SSH e modo nativo respondem."
    echo
fi

# --- Warm-up COMPLETO descartável --------------------------------------------
if [ "$DO_WARMUP" = 1 ]; then
    echo "========== WARM-UP COMPLETO (descartável) =========="
    NATIVE_TIMESTAMPS=1 RUN_TAG="_warmup" ./run_power_test.sh
    WARMUP_ST=$?
    WARMUP_OUT="${WORK_DIR}/${MODE}_native__warmup"
    if [ "$WARMUP_ST" -ne 0 ]; then
        echo "[WARM-UP] FALHOU (exit=${WARMUP_ST}); bateria não iniciada."
        exit "$WARMUP_ST"
    fi
    rm -rf "$WARMUP_OUT"
    echo "[WARM-UP] completo e descartado. Esfriando ${COOLDOWN_MIN} min..."
    countdown "$COOLDOWN_SEC"
    echo
fi

# --- Cinco execuções válidas -------------------------------------------------
done_runs=0
for ((r=1; r<=REPS; r++)); do
    run_one "$r" "r${r}"
    done_runs=$((done_runs + 1))
    if [ "$done_runs" -lt "$REPS" ]; then
        echo
        echo "Run ${done_runs}/${REPS} concluída. Esfriando ${COOLDOWN_MIN} min..."
        countdown "$COOLDOWN_SEC"
        echo
    fi
done

echo "=========================================================="
echo "   BATERIA NATIVE CONCLUÍDA — ${done_runs}/${REPS} runs válidas"
echo "=========================================================="
echo "Pasta    : $WORK_DIR"
echo "Manifest : $MANIFEST"
echo
echo "Status por run:"
cat "$MANIFEST"
echo "=========================================================="
