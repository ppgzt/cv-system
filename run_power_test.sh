#!/usr/bin/env bash
# ==============================================================================
# run_power_test.sh — UMA execução de teste com medição de potência (TC66C).
# PIBIC - CV System
# ------------------------------------------------------------------------------
# RODA NO MAC (orquestrador). O voltímetro TC66C está conectado via USB ao MAC;
# o pipeline mas-main.py roda no RASPBERRY, disparado via SSH.
#
# Protocolo (mesmo do dono do rasp):
#   1. liga TC66C.py (grava power.csv)
#   2. espera a 1ª leitura de potência (~1s) — só então começa o experimento
#   3. ssh Pi -> python mas-main.py ...   (BLOQUEIA até o pipeline acabar)
#   4. kill -INT no TC66C.py              (fecha/flusha o CSV)
#   5. scp puxa infra/reports/ do Pi de volta para o Mac
#
# Pré-requisitos no MAC:  python com `pyserial` e `pycryptodome` (TC66C.py).
# Pré-requisitos no PI :  SSH por chave (sem senha) ou ssh-agent destravado.
#
# Uso:
#   ./run_power_test.sh                     # defaults: mas-single, 5 fps, rebanho completo
#   FPS=10 NUM_ANIMALS=5 ./run_power_test.sh
#   MODE=mas-batch FPS=2 RUN_TAG=r1 ./run_power_test.sh
#   NATIVE_TIMESTAMPS=1 ./run_power_test.sh  # timestamps originais do dataset
# ==============================================================================
set -uo pipefail

# ============================== CONFIG ========================================
# --- Voltímetro TC66C (USB -> ESTE Mac) ---
# Descubra a porta com:  ls /dev/cu.usbmodem*
PORT="/dev/cu.usbmodemTC661"          # porta do TC66C no Mac (confirmada via ls /dev/cu.usbmodem*)
TC66_INT=1                            # intervalo de polling (s) — "1ª leitura em 1s"

# --- Raspberry (roda o pipeline) ---
PI_HOST="ewertonsjp@192.168.0.141"     # usuario@host (IP é mais confiável que o nome mDNS)
PI_DIR="~/projects/cv-system/"  # <-- AJUSTE: caminho do repo no Pi
# Python no Pi (pyenv "global" c/ as deps do pipeline). Shell não-interativo não
# carrega o pyenv do .bashrc, por isso usamos o caminho absoluto do interpretador.
PY_PI="${PY_PI:-/home/ewertonsjp/.pyenv/versions/cv_vend_mas/bin/python}"

# --- Experimento (defaults = TESTE rápido) ---
MODE="${MODE:-mas-single}"           # mas-single | mas-batch (env-overridable p/ bateria)
ENGINE="${ENGINE:-thread}"           # thread | pade
FPS="${FPS:-5}"
LOW_FPS="${LOW_FPS:-}"
NUM_ANIMALS="${NUM_ANIMALS:-}"         # vazio = TODOS os animais (rebanho completo)
EXTRA_ARGS="${EXTRA_ARGS:---debug}"  # --debug grava debug.log no Pi; vazio p/ desligar
NATIVE_TIMESTAMPS="${NATIVE_TIMESTAMPS:-0}"

# --- Python no Mac (para o TC66C.py) ---
# Precisa de pyserial + pycryptodome instalados neste interpretador.
PY="${PY:-.venv/bin/python}"

# --- Saída ---
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"  # env-overridable: a bateria passa "r1","r2",...
WORK_DIR="${WORK_DIR:-./power_runs}"
OUT_DIR="${WORK_DIR}/${MODE}_${FPS}fps_${RUN_TAG}"
if [ "$NATIVE_TIMESTAMPS" = "1" ]; then
    OUT_DIR="${WORK_DIR}/${MODE}_native_${RUN_TAG}"
elif [ -n "$LOW_FPS" ]; then
    OUT_DIR="${WORK_DIR}/${MODE}_low${LOW_FPS}fps_${RUN_TAG}"
fi
POWER_CSV="${OUT_DIR}/power.csv"
TC66_LOG="${OUT_DIR}/tc66.log"
SSH_LOG="${OUT_DIR}/pipeline.log"
# ==============================================================================

ts() { date "+%Y-%m-%dT%H:%M:%S%z"; }

echo "=========================================================="
if [ "$NATIVE_TIMESTAMPS" = "1" ]; then
    echo "  POWER TEST — ${MODE}/${ENGINE} @ native timestamps, ${NUM_ANIMALS:-todos} animais"
elif [ -n "$LOW_FPS" ]; then
    echo "  POWER TEST — ${MODE}/${ENGINE} @ adaptive (LOW=${LOW_FPS} fps, HIGH=native), ${NUM_ANIMALS:-todos} animais"
else
    echo "  POWER TEST — ${MODE}/${ENGINE} @ ${FPS} fps, ${NUM_ANIMALS:-todos} animais"
fi
echo "  Voltímetro : ${PORT}   (este Mac)"
echo "  Pipeline   : ${PI_HOST}:${PI_DIR}  (Raspberry, via SSH)"
echo "  Saída      : ${OUT_DIR}"
echo "=========================================================="

mkdir -p "$OUT_DIR"

# --- 0. Sanidade: a porta existe? ---
if [ ! -e "$PORT" ]; then
    echo "[ERROR] Porta '$PORT' não encontrada. Candidatos:"
    ls /dev/cu.usbmodem* 2>/dev/null || ls /dev/cu.* 2>/dev/null | head
    echo "Ajuste PORT= no topo deste script e rode novamente."
    exit 1
fi

# --- 1. Liga o TC66C em background (python -u => stdout sem buffer) ---
echo "[1/5] TC66C ligado (pid a seguir) -> $POWER_CSV"
"$PY" -u TC66C.py "$PORT" "$POWER_CSV" -t "$TC66_INT" >"$TC66_LOG" 2>&1 &
TC66_PID=$!
echo "      TC66C pid=$TC66_PID — esperando a 1ª leitura..."

# --- 2. Espera a 1ª leitura de potência (protocolo do dono do rasp) ---
READY=0
for _ in $(seq 1 20); do                          # até ~10s
    if ! kill -0 "$TC66_PID" 2>/dev/null; then
        echo "[ERROR] TC66C.py morreu no início. Log:"
        cat "$TC66_LOG"
        exit 1
    fi
    # linha de dado real: começa com timestamp ISO e tem >=4 vírgulas
    if grep -Eq '^[0-9]{4}-[0-9]{2}-[0-9]{2}T.*,.*,.*,.*,' "$TC66_LOG" 2>/dev/null; then
        READY=1
        FIRST=$(grep -m1 -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}T' "$TC66_LOG" | cut -d, -f1)
        echo "      1ª leitura capturada em $FIRST — liberando o experimento."
        break
    fi
    sleep 0.5
done
if [ "$READY" -ne 1 ]; then
    echo "[ERROR] Nenhuma leitura após ~10s. Log do TC66C:"
    cat "$TC66_LOG"
    kill -INT "$TC66_PID" 2>/dev/null
    exit 1
fi

# --- 3. Roda o pipeline no Raspberry (SSH bloqueia até terminar) ---
echo "[2/5] Disparando pipeline no Raspberry via SSH..."
case "$ENGINE" in
    thread|pade) ;;
    *)
        echo "[ERROR] ENGINE deve ser thread ou pade (recebido: $ENGINE)"
        kill -INT "$TC66_PID" 2>/dev/null
        exit 2
        ;;
esac

# O marcador torna a descoberta do report específica desta invocação, em vez
# de copiar silenciosamente a pasta mais recente de uma execução antiga.
REMOTE_MARKER=".run_power_test_${RUN_TAG}_$$_marker"
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$PI_HOST" \
    "cd ${PI_DIR} && : > '${REMOTE_MARKER}'"; then
    echo "[ERROR] Não foi possível criar marcador remoto para esta run."
    kill -INT "$TC66_PID" 2>/dev/null
    exit 1
fi

if [ "$NATIVE_TIMESTAMPS" = "1" ]; then
    CMD="cd ${PI_DIR} && fuser -k 8000/tcp 2>/dev/null || true; ${PY_PI} mas-main.py '${MODE}' --engine '${ENGINE}' --native-timestamps"
    [ -n "$NUM_ANIMALS" ] && CMD+=" --num-animals '${NUM_ANIMALS}'"
elif [ -n "$LOW_FPS" ]; then
    CMD="cd ${PI_DIR} && fuser -k 8000/tcp 2>/dev/null || true; ${PY_PI} mas-main.py '${MODE}' --engine '${ENGINE}'"
    [ -n "$NUM_ANIMALS" ] && CMD+=" --num-animals '${NUM_ANIMALS}'"
else
    CMD="cd ${PI_DIR} && fuser -k 8000/tcp 2>/dev/null || true; ${PY_PI} mas-main.py '${MODE}' '${FPS}' --engine '${ENGINE}'"
    [ -n "$NUM_ANIMALS" ] && CMD+=" '${NUM_ANIMALS}'"
fi
[ -n "$EXTRA_ARGS" ]  && CMD+=" ${EXTRA_ARGS}"



echo "      ssh ${PI_HOST} \"${CMD}\""
echo "[T0] início do pipeline: $(ts)" | tee -a "$SSH_LOG"
ssh -o BatchMode=yes "$PI_HOST" "$CMD" 2>&1 | tee -a "$SSH_LOG"
SSH_STATUS=${PIPESTATUS[0]}
echo "[T1] fim do pipeline:    $(ts) (exit=${SSH_STATUS})" | tee -a "$SSH_LOG"

# --- 4. Para o TC66C e RECONSTRÓI o power.csv a partir do tc66.log ---
# O TC66C.py grava o power.csv em buffer de bloco e só flusha no f.close() do
# KeyboardInterrupt — mas o SIGINT quase nunca é tratado a tempo (Poll() bloqueia
# até 5s na leitura serial), e o SIGTERM mata o processo sem flush, truncando o
# CSV. Solução robusta (sem mexer no TC66C.py): o tc66.log é o stdout do
# `python -u` (sem buffer) e contém TODAS as amostras, no mesmo formato. Depois
# que o processo morre de fato, regeneramos o power.csv a partir dele.
echo "[3/5] Parando TC66C e reconstruindo power.csv do tc66.log..."
kill -INT "$TC66_PID" 2>/dev/null
for _ in $(seq 1 40); do                # grace ~8s (cobre 1 ciclo de Poll)
    kill -0 "$TC66_PID" 2>/dev/null || break
    sleep 0.2
done
kill -0 "$TC66_PID" 2>/dev/null && kill -TERM "$TC66_PID" 2>/dev/null
# garante que o processo está realmente morto antes de ler o log
for _ in $(seq 1 10); do
    kill -0 "$TC66_PID" 2>/dev/null || break
    sleep 0.2
done
# recria o power.csv: cabeçalho + linhas de dado do tc66.log (formato idêntico)
awk 'BEGIN{print "Datetime,Time[S],Volt[V],Current[A],Power[W]"}
     /^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:,.-]+,[^,]+,[^,]+,[^,]+,[^,]+/' \
    "$TC66_LOG" > "$POWER_CSV"
echo "      power.csv reconstruído: $(($(wc -l < "$POWER_CSV")-1)) amostras"

# --- 5. Puxa a pasta de relatório do Pi de volta para o Mac (best-effort) ---
echo "[4/5] Puxando relatório do Raspberry..."
# Caminho ABSOLUTO: scp resolve relativo ao HOME remoto, não ao repo, então
# usamos $PWD (expandido no shell remoto) pra prefixar o diretório de reports.
REMOTE_DIRS=()
while IFS= read -r remote_dir; do
    [ -n "$remote_dir" ] && REMOTE_DIRS+=("$remote_dir")
done < <(ssh -o BatchMode=yes "$PI_HOST" \
    "cd ${PI_DIR} && find \"\$PWD\"/infra/reports -mindepth 1 -maxdepth 1 -type d -newer '${REMOTE_MARKER}' -name '${MODE}_${ENGINE}_*' -print 2>/dev/null")
ssh -o BatchMode=yes "$PI_HOST" \
    "cd ${PI_DIR} && rm -f '${REMOTE_MARKER}'" >/dev/null 2>&1 || true

if [ "${#REMOTE_DIRS[@]}" -eq 1 ]; then
    REMOTE_DIR="${REMOTE_DIRS[0]}"
    if scp -q -o BatchMode=yes -r "${PI_HOST}:${REMOTE_DIR}" "$OUT_DIR/"; then
        echo "      puxado: ${REMOTE_DIR} -> ${OUT_DIR}/"
    else
        echo "      [WARN] scp falhou (relatórios continuam no Pi em ${REMOTE_DIR})"
    fi
else
    echo "      [WARN] esperado exatamente um report novo ${MODE}_${ENGINE}_*; encontrados ${#REMOTE_DIRS[@]}"
    [ "${#REMOTE_DIRS[@]}" -gt 0 ] && printf '             %s\n' "${REMOTE_DIRS[@]}"
fi

# --- Resumo ---
echo "[5/5] CONCLUÍDO."
echo "----------------------------------------------------------"
echo "  power.csv  : ${POWER_CSV}  ($(wc -l < "$POWER_CSV" 2>/dev/null | tr -d ' ') linhas)"
echo "  pipeline   : ${SSH_LOG}    (ssh exit=${SSH_STATUS})"
[ "$SSH_STATUS" -ne 0 ] && echo "  [WARN] pipeline saiu com erro — veja ${SSH_LOG}"
echo "  pasta      : ${OUT_DIR}"
echo "=========================================================="
exit "$SSH_STATUS"
