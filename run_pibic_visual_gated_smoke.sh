#!/usr/bin/env bash
# ==============================================================================
# run_pibic_visual_gated_smoke.sh
# PIBIC - Smoke test do runtime Visual-Gated no Raspberry Pi
# ------------------------------------------------------------------------------
# RODA NO MAC.
#
# Objetivo:
#   - validar integração real PADE no Raspberry;
#   - validar Visual Event + Orchestrator + Selection Hold;
#   - validar trigger forwarding;
#   - validar LOW -> HIGH -> LOW;
#   - validar END semantics;
#   - validar geração/cópia das telemetrias.
#
# NÃO É EXECUÇÃO EXPERIMENTAL OFICIAL.
# NÃO usar os resultados como medição final de CPU/temperatura/energia,
# pois o --debug permanece habilitado neste smoke test.
#
# Uso:
#   chmod +x run_pibic_visual_gated_smoke.sh
#   caffeinate -dimsu ./run_pibic_visual_gated_smoke.sh
#
# Exemplos:
#   NUM_ANIMALS=20 ./run_pibic_visual_gated_smoke.sh
#   NUM_ANIMALS=40 LOW_FPS=5 ./run_pibic_visual_gated_smoke.sh
# ==============================================================================

set -uo pipefail

# ============================== CONFIG ========================================

MODE="${MODE:-mas-single}"
ENGINE="${ENGINE:-pade}"

# LOW ainda NÃO está congelado cientificamente.
# Para o smoke test usamos 4 FPS provisoriamente.
LOW_FPS="${LOW_FPS:-4}"

# Selection Hold já congelado.
HOLD_N="${HOLD_N:-2}"

# Smoke curto.
# Pode aumentar se as primeiras passagens não exercitarem bem as transições.
NUM_ANIMALS="${NUM_ANIMALS:-20}"

# Debug propositalmente ligado neste smoke test.
EXTRA_ARGS="${EXTRA_ARGS:---low-fps ${LOW_FPS} --visual-gated --selection-hold-n ${HOLD_N} --debug}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-visual_gated_smoke_low${LOW_FPS}_${RUN_TS}}"

WORK_DIR="${WORK_DIR:-./power_runs/${RUN_TAG}}"

export MODE
export ENGINE
export LOW_FPS
export HOLD_N
export NUM_ANIMALS
export EXTRA_ARGS
export RUN_TAG
export WORK_DIR

# ==============================================================================

echo
echo "======================================================================"
echo " PIBIC — SMOKE TEST VISUAL-GATED"
echo "======================================================================"
echo "Modo             : ${MODE}"
echo "Engine           : ${ENGINE}"
echo "Visual-Gated     : SIM"
echo "LOW              : ${LOW_FPS} FPS (provisório)"
echo "HIGH             : trace temporal nativo"
echo "Selection Hold   : N=${HOLD_N}"
echo "Animais          : ${NUM_ANIMALS}"
echo "Debug            : SIM"
echo "Run tag          : ${RUN_TAG}"
echo "Saída local      : ${WORK_DIR}"
echo
echo "ATENÇÃO:"
echo "  Esta execução é apenas funcional."
echo "  Não utilizar CPU/temp/energia como resultado experimental final."
echo "======================================================================"
echo

mkdir -p "${WORK_DIR}"

LOG_FILE="${WORK_DIR}/_smoke_launcher.log"

echo "[SMOKE] Iniciando execução no Raspberry via run_power_test.sh..."
echo

ENGINE="${ENGINE}" \
MODE="${MODE}" \
LOW_FPS="${LOW_FPS}" \
NUM_ANIMALS="${NUM_ANIMALS}" \
EXTRA_ARGS="${EXTRA_ARGS}" \
RUN_TAG="${RUN_TAG}" \
WORK_DIR="${WORK_DIR}" \
./run_power_test.sh 2>&1 | tee "${LOG_FILE}"

ST=${PIPESTATUS[0]}

echo
echo "======================================================================"

if [ "${ST}" -ne 0 ]; then
    echo "[SMOKE] FALHOU — exit=${ST}"
    echo
    echo "Verifique:"
    echo "  - ${LOG_FILE}"
    echo "  - conexão SSH"
    echo "  - inicialização dos agentes PADE"
    echo "  - carregamento do frame_selector.tflite"
    echo "  - exceções do VisualEventAgent / OrchestratorAgent"
    echo "  - leases/FrameStore"
    echo "  - EndPassageEvent / EndPipelineEvent"
    echo "======================================================================"
    exit "${ST}"
fi

echo "[SMOKE] Execução terminou sem erro de processo."
echo
echo "Agora verificar nos artefatos/logs:"
echo
echo "  [ ] passagens solicitadas finalizadas"
echo "  [ ] Visual IDLE observado"
echo "  [ ] Visual ACTIVE observado"
echo "  [ ] pelo menos uma transição IDLE -> ACTIVE"
echo "  [ ] frame trigger encaminhado ao Selection"
echo "  [ ] trigger encaminhado exatamente uma vez"
echo "  [ ] frames HIGH chegando ao Selection"
echo "  [ ] Selection Hold ativado"
echo "  [ ] Selection Hold liberado após N=${HOLD_N} rejeições"
echo "  [ ] retorno HIGH -> LOW"
echo "  [ ] nenhum deadlock"
echo "  [ ] EndPassageEvent propagado"
echo "  [ ] EndPipelineEvent propagado"
echo "  [ ] telemetrias geradas"
echo "  [ ] nenhum erro relacionado a FrameStore/leases"
echo
echo "Saída local:"
echo "  ${WORK_DIR}"
echo
echo "Se esses itens estiverem corretos, o Visual-Gated está apto"
echo "para a próxima etapa de execução física/calibração."
echo "======================================================================"

exit 0