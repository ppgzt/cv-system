#!/usr/bin/env bash
# ==============================================================================
# run_power_battery.sh — BATERIA de experimentos com medição de potência.
# PIBIC - CV System
# ------------------------------------------------------------------------------
# Empilha chamadas do run_power_test.sh. Ordem INTERCALADA (rep-externo,
# FPS-interno): varre FPS_LIST inteira na rep 1, de novo na rep 2, ... — nunca
# agrupa as N reps do mesmo FPS (distribui drift térmico entre os níveis).
# Cooldown entre runs (esfriar o Pi) + warm-up opcional descartável no início.
# Cada run mede potência (TC66C no Mac) + pipeline no Pi.
#
# RODA NO MAC (mesmo orquestrador do run_power_test.sh). Pré-requisitos idem:
# veja README-power.md (SSH por chave, pyenv no Pi, pyserial/pycryptodome no Mac).
#
# Uso:
#   ./run_power_battery.sh                     # usa FPS_LIST/REPS/COOLDOWN do topo
#   REPS=3 COOLDOWN_MIN=10 ./run_power_battery.sh
#   FPS_LIST="1 2 3" MODE=mas-batch ./run_power_battery.sh
#
# Saídas: ./power_runs/battery_<MODE>_<ts>/<MODE>_<FPS>fps_r<R>/  (uma pasta por run)
#         + ./power_runs/battery_<MODE>_<ts>/_manifest.txt        (status de cada run)
# ==============================================================================
set -uo pipefail

# ============================== CONFIG ========================================
# Lista de FPS a variar (9 valores típicos p/ o estudo de capacidade):
FPS_LIST="${FPS_LIST:-1 2 3 4 5 10 15 20 30}"

REPS="${REPS:-5}"                 # repetições por FPS
COOLDOWN_MIN="${COOLDOWN_MIN:-5}"  # minutos de espera entre runs (esfriar o Pi c/ ventoinha ativa)
COOLDOWN_SEC=$((COOLDOWN_MIN * 60))

MODE="${MODE:-mas-single}"        # mas-single | mas-batch
NUM_ANIMALS="${NUM_ANIMALS:-}"    # vazio = rebanho completo (recomendado p/ comparar FPS)
EXTRA_ARGS="${EXTRA_ARGS:---debug}"

# Warm-up descartável (1 run que não entra na bateria — zera caches frios):
DO_WARMUP="${DO_WARMUP:-1}"
WARMUP_FPS="${WARMUP_FPS:-10}"

# Pasta raiz das runs: uma pasta por bateria (evita colisão entre baterias)
BATTERY_TS="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="${WORK_DIR:-./power_runs/battery_${MODE}_${BATTERY_TS}}"
MANIFEST="${WORK_DIR}/_manifest.txt"
export WORK_DIR MODE NUM_ANIMALS EXTRA_ARGS
# ==============================================================================

# Contagem regressiva com atualização inline (igual ao run_experiments.sh)
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

# Roda UMA run e registra no manifest. $1=FPS $2=rep $3=RUN_TAG
run_one() {
    local fps=$1 rep=$2 tag=$3
    local out="${WORK_DIR}/${MODE}_${fps}fps_${tag}"
    echo "----------------------------------------------------------"
    echo "[RUN] ${MODE} @ ${fps} fps  (rep ${rep})  -> ${out}"
    echo "----------------------------------------------------------"
    FPS="$fps" RUN_TAG="$tag" ./run_power_test.sh
    local st=$?
    if [ "$st" -eq 0 ] && [ -d "$out" ]; then
        echo "[RUN] OK  -> ${out}"
        printf '%s\t%s\tOK\t%s\n' "$fps" "$rep" "$out" >> "$MANIFEST"
    else
        echo "[RUN] FALHA (exit=${st}) — veja os logs em ${out:-(sem pasta)}"
        printf '%s\t%s\tFALHA(%s)\t%s\n' "$fps" "$rep" "$st" "${out:-}" >> "$MANIFEST"
    fi
    return 0          # continue-on-error: uma run falha não aborta a bateria
}

mkdir -p "$WORK_DIR"
: > "$MANIFEST"

# Cabeçalho
echo "=========================================================="
echo "   BATERIA DE POTÊNCIA (run_power_test.sh × N)"
echo "=========================================================="
echo "Modo           : $MODE"
echo "FPS_LIST       : $FPS_LIST"
echo "Repetições/FPS : $REPS"
echo "Cooldown/run   : $COOLDOWN_MIN min"
echo "Warm-up        : $([ "$DO_WARMUP" = 1 ] && echo "sim (${WARMUP_FPS} fps, descartável)" || echo "não")"
echo "Num animais    : ${NUM_ANIMALS:-rebanho completo}"
echo "Pasta          : $WORK_DIR"
echo "=========================================================="
echo

# Garante que o voltímetro e o Pi respondem ANTES de comprometer horas de bateria.
# USA WARMUP_FPS (5), NUNCA 999 — FPS alto demais induz o hang λ≫μ do thread engine
# na RPi5 (ver memória), que é justamente o que o preflight quer evitar. E loga num
# arquivo (não /dev/null) pra a gente enxergar o erro se falhar.
PREFLIGHT_LOG="${WORK_DIR}/_preflight.log"
echo "[CHECK] Validando voltímetro + SSH+pipeline com 1 animal (descartável, FPS=${WARMUP_FPS})..."
FPS="$WARMUP_FPS" RUN_TAG="_preflight" NUM_ANIMALS=1 EXTRA_ARGS="" \
    ./run_power_test.sh >"$PREFLIGHT_LOG" 2>&1
PF_ST=$?
if [ "$PF_ST" -ne 0 ] || [ ! -d "${WORK_DIR}/${MODE}_${WARMUP_FPS}fps__preflight" ]; then
    echo "[CHECK] FALHOU (exit=${PF_ST}) — abortando a bateria. Log do preflight:"
    sed 's/^/    /' "$PREFLIGHT_LOG"
    echo "        (voltímetro desconectado? Pi off? pyenv errado? SSH sem chave? pipeline travou?)"
    exit 1
fi
rm -rf "${WORK_DIR}/${MODE}_${WARMUP_FPS}fps__preflight" "$PREFLIGHT_LOG"
echo "[CHECK] OK — voltímetro e pipeline respondem."
echo

TOTAL_FPS=$(echo "$FPS_LIST" | wc -w | tr -d ' ')
TOTAL_RUNS=$((TOTAL_FPS * REPS))
done_runs=0

# --- Warm-up descartável -----------------------------------------------------
if [ "$DO_WARMUP" = 1 ]; then
    echo "========== WARM-UP (descartável) =========="
    FPS="$WARMUP_FPS" RUN_TAG="_warmup" ./run_power_test.sh
    rm -rf "${WORK_DIR}/${MODE}_${WARMUP_FPS}fps__warmup"
    echo "[WARM-UP] descartado. Esfriando ${COOLDOWN_MIN} min antes da bateria..."
    countdown "$COOLDOWN_SEC"
    echo
fi

# --- Bateria: REPS × FPS -----------------------------------------------------
# Ordem INTERCALADA (rep-externo / FPS-interno): 1,2,…,30,  1,2,…,30, …
# Distribui o drift térmico/carga uniformemente entre os FPS em vez de
# concentrar todo o aquecimento num único nível. (User pref.: não agrupar.)
for ((r=1; r<=REPS; r++)); do
    for fps in $FPS_LIST; do
        run_one "$fps" "$r" "r${r}"
        done_runs=$((done_runs + 1))
        # cooldown entre runs, exceto após a última de todas
        if [ "$done_runs" -lt "$TOTAL_RUNS" ]; then
            echo
            echo "Run ${done_runs}/${TOTAL_RUNS} concluída. Esfriando ${COOLDOWN_MIN} min..."
            countdown "$COOLDOWN_SEC"
            echo
        fi
    done
done

# --- Resumo ------------------------------------------------------------------
echo "=========================================================="
echo "   BATERIA CONCLUÍDA — ${done_runs}/${TOTAL_RUNS} runs"
echo "=========================================================="
echo "Pasta    : $WORK_DIR"
echo "Manifest : $MANIFEST"
echo
echo "Status por run:"
cat "$MANIFEST"
echo "=========================================================="
