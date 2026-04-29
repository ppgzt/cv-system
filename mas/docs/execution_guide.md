# Guia de Execução: MAS (Multi-Agent System)

Este guia descreve como executar as simulações utilizando a arquitetura de Multi-Agentes (MAS) nos modos **Single** e **Batch**.

## 1. Configuração de Ambiente (.env)

O sistema utiliza o arquivo `.env` na raiz do projeto para configurar a rede do framework PADE. Certifique-se de que os valores estão corretos:

| Variável | Descrição | Valor Padrão |
| :--- | :--- | :--- |
| `SMA_AMS_HOST` | Host do Agente de Gerenciamento (AMS) | `localhost` |
| `SMA_AMS_PORT` | Porta do Agente de Gerenciamento | `8000` |
| `SMA_AGENT_HOST` | Host onde os agentes serão executados | `localhost` |
| `SMA_AGENT_BASE_PORT` | Porta inicial para alocação dos agentes | `5010` |

---

## 2. Argumentos da Linha de Comando (CLI)

A execução é feita através do arquivo `main.py`. Os argumentos são **posicionais** e devem seguir a ordem abaixo:

```bash
python main.py <strategy> <herd_size> <passage_time> <arrival_time> <fselection_time> <fselection_window>
```

### Detalhamento dos Parâmetros:

1.  **`strategy`**: Define o modo de execução.
    *   `mas_single`: Execução via MAS com predição frame-a-frame (tempo real).
    *   `mas_batch`: Execução via MAS com predição em lote ao final da passagem.
2.  **`herd_size`**: Quantidade de animais (ciclos) na simulação.
3.  **`passage_time`**: Tempo que o animal permanece "em frente à câmera" (segundos).
4.  **`arrival_time`**: Tempo de espera entre a saída de um animal e a entrada de outro (segundos).
5.  **`fselection_time`**: Intervalo de captura de frames (simulação de FPS). Ex: `0.2` equivale a 5 FPS.
6.  **`fselection_window`**: Janela de adequação para o seletor de frames (limiar de qualidade).

---

## 3. Exemplos de Execução

### Modo MAS Single
Neste modo, cada frame selecionado é enviado imediatamente para predição.
```bash
python main.py mas_single 5 30 5 0.2 0.5
```
*(5 animais, 30s de passagem, 5s de espera, 5 FPS, 0.5 de limiar)*

### Modo MAS Batch
Neste modo, os frames são acumulados e a predição ocorre apenas quando o animal termina de passar.
```bash
python main.py mas_batch 5 30 5 0.2 0.5
```

---

## 4. Saída e Resultados

Após a conclusão de todos os animais (`herd_size`), o Agente de Predição (`PredictWeightAgent`) salvará os resultados em:

`infra/reports/<PID>/metrics.json`

Onde `<PID>` é gerado automaticamente contendo a estratégia e o timestamp da execução (ex: `mas_batch_2024-04-26T10:00:00`).

---

## 5. Observações Importantes

*   **Bloqueio do Reator**: O MAS utiliza o Twisted Reactor. Ao final da execução, o `PredictWeightAgent` encerra o reactor automaticamente.
*   **Logs**: O sistema exibe mensagens em tempo real no terminal via `display_message`, permitindo acompanhar a comunicação entre Capture, Selection e Predict.
*   **Zerar Relatórios**: Antes de rodar novos experimentos científicos, certifique-se de que a pasta `infra/reports/` está organizada para evitar confusão entre execuções.
