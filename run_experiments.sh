#!/bin/bash

# ==============================================================================
# Script de Automação de Experimentos (mas-main.py)
# PIBIC - CV System
# ==============================================================================
# Uso:
#   ./run_experiments.sh [mode] [extra_args...]
#
# Exemplos:
#   ./run_experiments.sh mas-single
#   ./run_experiments.sh mas-batch
#   ./run_experiments.sh mas-single 3 --debug (roda apenas 3 animais com debug log)
# ==============================================================================

# Configurações de tempo
INTERVAL_MINUTES=12
INTERVAL_SECONDS=$((INTERVAL_MINUTES * 60))

# Parâmetros
MODE="mas-single"
if [ $# -gt 0 ]; then
    if [ "$1" = "mas-single" ] || [ "$1" = "mas-batch" ]; then
        MODE=$1
        shift 1
    fi
fi
EXTRA_ARGS=("$@")

# Lista de FPS para os experimentos reais
FPS_LIST=(3 4 5 10 15 20 30)

# Interpretador Python (execução normal, sem venv)
PYTHON_EXEC="python"

# Função para contagem regressiva com atualização inline
countdown() {
    local secs=$1
    while [ $secs -gt 0 ]; do
        local mins=$((secs / 60))
        local s=$((secs % 60))
        printf "\r[Aguardando] Próximo experimento em: %02dm %02ds..." $mins $s
        sleep 1
        secs=$((secs - 1))
    done
    printf "\r[Pronto] Iniciando próximo experimento agora!                         \n"
}

# Cabeçalho informativo
echo "=========================================================="
echo "   INICIANDO BATERIA DE EXPERIMENTOS (mas-main.py)"
echo "=========================================================="
echo "Modo de execução: $MODE"
echo "Argumentos extras: ${EXTRA_ARGS[*]:-(nenhum)}"
echo "Lista de FPS:      ${FPS_LIST[*]}"
echo "Intervalo:         $INTERVAL_MINUTES minutos ($INTERVAL_SECONDS segundos)"
echo "=========================================================="
echo

# ------------------------------------------------------------------------------
# 1. EXPERIMENTO DE WARM-UP (5 FPS) - PULADO (SISTEMA JÁ AQUECIDO)
# ------------------------------------------------------------------------------
# echo "----------------------------------------------------------"
# echo "[WARM-UP] Rodando experimento inicial de 5 FPS (Warm-up)..."
# echo "----------------------------------------------------------"
# echo "Comando: $PYTHON_EXEC mas-main.py \"$MODE\" 5 ${EXTRA_ARGS[*]}"
# echo
# 
# $PYTHON_EXEC mas-main.py "$MODE" 5 "${EXTRA_ARGS[@]}"
# WARMUP_STATUS=$?
# 
# # Localizar e deletar a pasta gerada pelo warm-up
# NEW_DIR=$(ls -td infra/reports/"${MODE}"_* 2>/dev/null | head -n 1)
# 
# if [ -n "$NEW_DIR" ] && [ -d "$NEW_DIR" ]; then
#     echo
#     echo "[WARM-UP] Descartando pasta gerada pelo warm-up: $NEW_DIR"
#     rm -rf "$NEW_DIR"
#     echo "[WARM-UP] Arquivos descartados com sucesso."
# else
#     echo
#     echo "[WARM-UP] AVISO: Não foi possível localizar a pasta de relatórios do warm-up."
# fi
# 
# if [ $WARMUP_STATUS -ne 0 ]; then
#     echo "[WARM-UP] AVISO: O comando de warm-up terminou com código de saída $WARMUP_STATUS."
# fi
# 
# echo "[WARM-UP] Warm-up concluído."
# echo
# 
# # Intervalo após o warm-up
# echo "Aguardando $INTERVAL_MINUTES minutos de intervalo após o warm-up..."
# countdown $INTERVAL_SECONDS
# echo

# ------------------------------------------------------------------------------
# 2. LOOP DE EXPERIMENTOS PRINCIPAIS
# ------------------------------------------------------------------------------
TOTAL_EXP=${#FPS_LIST[@]}
for i in "${!FPS_LIST[@]}"; do
    FPS=${FPS_LIST[$i]}
    EXP_NUM=$((i + 1))
    
    echo "----------------------------------------------------------"
    echo "[EXPERIMENTO $EXP_NUM/$TOTAL_EXP] Rodando com FPS: $FPS"
    echo "----------------------------------------------------------"
    echo "Comando: $PYTHON_EXEC mas-main.py \"$MODE\" \"$FPS\" ${EXTRA_ARGS[*]}"
    echo
    
    # Executa o script python
    $PYTHON_EXEC mas-main.py "$MODE" "$FPS" "${EXTRA_ARGS[@]}"
    RUN_STATUS=$?
    
    # Localiza a pasta mais recente gerada para este modo
    NEW_DIR=$(ls -td infra/reports/"${MODE}"_* 2>/dev/null | head -n 1)
    
    if [ -n "$NEW_DIR" ] && [ -d "$NEW_DIR" ]; then
        TARGET_DIR="infra/reports/pre_${FPS}_FPS"
        echo
        echo "[EXPERIMENTO $EXP_NUM/$TOTAL_EXP] Renomeando pasta de relatórios:"
        echo "  De:   $NEW_DIR"
        echo "  Para: $TARGET_DIR"
        
        # Limpar destino anterior para evitar conflito/aninhamento no rename
        if [ -d "$TARGET_DIR" ]; then
            echo "[EXPERIMENTO $EXP_NUM/$TOTAL_EXP] Pasta destino já existe. Removendo antiga..."
            rm -rf "$TARGET_DIR"
        fi
        
        mv "$NEW_DIR" "$TARGET_DIR"
        echo "[EXPERIMENTO $EXP_NUM/$TOTAL_EXP] Pasta renomeada com sucesso!"
    else
        echo
        echo "[EXPERIMENTO $EXP_NUM/$TOTAL_EXP] ERRO: Não foi possível localizar a pasta de relatório gerada."
    fi
    
    if [ $RUN_STATUS -ne 0 ]; then
        echo "[EXPERIMENTO $EXP_NUM/$TOTAL_EXP] AVISO: Execução do mas-main.py retornou status de erro $RUN_STATUS."
    fi
    
    # Se não for o último experimento, aguardar o intervalo
    if [ $EXP_NUM -lt $TOTAL_EXP ]; then
        echo
        echo "Experimento $EXP_NUM finalizado. Iniciando intervalo de $INTERVAL_MINUTES minutos..."
        countdown $INTERVAL_SECONDS
        echo
    fi
done

echo "=========================================================="
echo "   TODOS OS EXPERIMENTOS CONCLUÍDOS COM SUCESSO!"
echo "=========================================================="
