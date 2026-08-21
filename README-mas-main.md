# Runtime experimental PIBIC — Raspberry Pi 5

O entrypoint experimental oficial é:

```bash
python mas-main.py --engine pade --mode <modalidade>
```

`main.py`, `domain/pipelines.py` e `mas/agents/capture_agent.py` são caminhos
históricos. Permanecem para reprodução, mas não são o runtime atual nem devem
aparecer no comando de deploy.

## Ambiente e modelos

Use Raspberry Pi 5 16 GB, Python 3.13, o virtualenv `.venv`, TensorFlow Lite,
Twisted, PADE, NumPy/SciPy, psutil e python-dotenv. Configure AMS/PADE em `.env`
por `SMA_AMS_HOST`, `SMA_AMS_PORT`, `SMA_AGENT_HOST` e `SMA_AGENT_BASE_PORT`.
O dataset padrão é `data/exp1` (`--data-root` permite outro cohort).

```bash
source .venv/bin/activate
```

Os TFLite são deliberadamente ignorados pelo Git e devem ser copiados por SCP
para `infra/models/` antes da execução. Não adicione os modelos ao Git/LFS.

| Arquivo | Papel | SHA256 |
|---|---|---|
| `frame_selector.tflite` | selector final v3 ROI10 | `f0886d0f01a1b48ccb836da7ea139caa58f0e0e445ee27ef2ec2a07abd9adca7` |
| `frame_selector_passage_level_antigo.tflite` | selector histórico v2 grouped | `c3ce562b0945b12d168ec4af0654ed332d3941354bbd1c2bce77f89227e1544a` |
| `frame_selector_image_level.tflite` | selector histórico image-level | `2d2b8a2ddde53fa4738d0d93b6a1a67c54e925999c8c8bdc9df7e010589ccfbf` |
| `sheep_weight_predictor.tflite` | regressor de peso | `15b9d310c8deffc4629a107b62e889d13c8fb55186759c595ee7b0c192e50d4a` |

No Pi/Linux:

```bash
sha256sum infra/models/*.tflite
```

No macOS: `shasum -a 256 infra/models/*.tflite`.

## As cinco modalidades oficiais

| Modalidade | Visual adaptive | Visual gate | Resource cap | Scheduler |
|---|---:|---:|---:|---|
| `original-timing` | OFF | OFF | OFF | timestamps originais |
| `fixed-fps` | OFF | OFF | OFF | fixed-FPS histórico |
| `visual-adaptive` | ON | OFF | OFF | causal LOW/MEDIUM/HIGH |
| `visual-gated` | ON | ON | OFF | adaptive + gating |
| `resource-aware-visual-gated` | ON | ON | ON | Visual-Gated + resource cap |

```bash
python mas-main.py --engine pade --mode original-timing
python mas-main.py --engine pade --mode fixed-fps --fps 5
python mas-main.py --engine pade --mode visual-adaptive
python mas-main.py --engine pade --mode visual-gated
python mas-main.py --engine pade --mode resource-aware-visual-gated
```

`fixed-fps` exige `--fps`; não há FPS implícito. LOW=4 FPS, MEDIUM=7 FPS e
HIGH mantém o timing nativo/original do trace, sem upsampling artificial.

## Configuração

Defaults autoritativos: `mas/experiment_config.py`. CLI sobrescreve valores por
run sem editar múltiplos arquivos; expor uma opção não altera o protocolo.

| Parâmetro | Default | Unidade | Modos |
|---|---:|---|---|
| `--mode` | obrigatório | — | todos |
| `--fps` | obrigatório em fixed | fps | fixed-fps |
| `--low-fps`, `--medium-fps` | 4.0, 7.0 | fps | visuais |
| `--selector-threshold` | 0.5 | prob. | todos |
| selector ROI | y/x 10–90% | fração | todos, congelado v3 |
| `--visual-pdi-threshold` | 0.08747855917667238 | PDI | visuais |
| `--visual-roi` | `0.30 0.675 0.00 1.00` | fração | visuais |
| `--visual-idle-patience` | 3 | observações | visuais |
| visual depth diff | 200 | mm | visuais |
| quality p99 / fraction | 2230 / 0.0027473958333333335 | mm / fração | visuais |
| `--resource-warning-temperature` | 75.0 | °C | resource-aware |
| `--resource-critical-temperature` | 80.0 | °C | resource-aware |
| `--resource-warning-backlog` | 7 | eventos | resource-aware |
| `--resource-stale-after-seconds` | 10.0 | s | resource-aware |
| `--aggregation-mode` | single | — | todos |
| `--num-animals`, `--data-root` | todos / `data/exp1` | — | todos |
| `--output-dir`, `--run-id`, `--repetition` | `infra/reports` / timestamp / vazio | — | todos |

Exemplo:

```bash
python mas-main.py --engine pade --mode resource-aware-visual-gated \
  --run-id pibic-pi5-001 --repetition 1 --num-animals 3 \
  --output-dir infra/reports
```

## Resource Management

Só `resource-aware-visual-gated` permite que ResourceState limite aquisição.
CPU e RAM continuam em telemetria/blackboard, mas não classificam estado.

```text
throttling atual ativo                 -> CRITICAL -> máximo LOW
temperatura >= 80 °C                   -> CRITICAL -> máximo LOW
temperatura >= 75 °C                   -> WARNING  -> máximo MEDIUM
prediction_backlog >= 7                -> WARNING  -> máximo MEDIUM
outro caso                             -> SAFE     -> sem cap adicional
```

`prediction_backlog` é a ocupação da borda **Preprocessing → Prediction**
(`PredictWeightAgent.inbox`), a mesma métrica
`preprocessing_to_prediction_qsize` em `queue_telemetry.csv`. Ela contém apenas
trabalho aceito e pré-processado aguardando Prediction; não soma filas upstream
nem inclui inferência já em execução. É lida na publicação Resource, a cada
aproximadamente 5 s.

`vcgencmd get_throttled` é coletado pelo monitor de hardware com timeout de
0,5 s. Telemetria preserva raw/máscara/flags. Qualquer flag *atual* de
undervoltage, frequency-capped, throttled ou soft-temperature-limit é CRITICAL.
Se o comando faltar/falhar/expirar, o sinal fica `None`/unavailable — nunca
False silenciosamente — e o Resource Agent não cai.

Cada estado tem sequência e timestamp monotônico. Após mais de 10 s sem amostra
fresca, o Orchestrator congela conservadoramente a taxa/cap efetiva e não permite
novo upshift até uma amostra fresca. Resource Manager nunca envia FPS ao Capture.

## Outputs, shutdown e deploy

Cada execução salva em `<output-dir>/<run-id>/`: `metrics.json`, `report.md`,
`capture_timing.csv`, `queue_telemetry.csv`, `hardware_telemetry.csv`,
`control_activity.csv`, `cpu.csv`, `mem.csv`, `temp.csv`, e nos modos visuais
`visual_activity.csv`. `--debug` adiciona `debug.log`.

Finalize com `Ctrl-C` e aguarde o drain de END/Prediction; não mate o processo
antes da finalização se precisar de resultados completos.

Sequência recomendada no Raspberry:

```bash
git pull
source .venv/bin/activate
# instalar/sincronizar dependências quando necessário
# copiar os quatro TFLite por SCP para infra/models/ quando necessário
sha256sum infra/models/*.tflite
python mas-main.py --engine pade --mode resource-aware-visual-gated --num-animals 3 --run-id smoke-pi5
# revisar CSVs/metrics; então executar a coorte planejada
```

Antes da coorte, confirme no Pi `vcgencmd get_throttled`, AMS/PADE e shutdown.

## Scripts operacionais no Raspberry

Os scripts usam somente o entrypoint oficial, descobrem a raiz do projeto a
partir do próprio arquivo e exigem Python 3.13 no `.venv`. Dê permissão uma vez:

```bash
chmod +x scripts/smoke_raspberry.sh scripts/pilot_5_modes_raspberry.sh
```

Smoke curto (três primeiras tags da ordem oficial do dataset; a CLI atual não
aceita filtrar tags individuais):

```bash
./scripts/smoke_raspberry.sh
```

Ele exige os dois modelos ativos, valida seus SHA256 por padrão, exige
`vcgencmd`, registra `measure_temp`/`get_throttled`, executa apenas
`resource-aware-visual-gated --num-animals 3` e escreve em
`results/smoke_raspberry_<timestamp>/`. Para desabilitar apenas a conferência de
hash em uma investigação local, use `VERIFY_MODEL_SHA256=0`; isso não é
recomendado para o piloto.

Piloto ordenado, uma execução do cohort oficial por modalidade:

```bash
./scripts/pilot_5_modes_raspberry.sh
FIXED_FPS=5 ./scripts/pilot_5_modes_raspberry.sh
```

Os resultados ficam em `results/pilot_5_modes_<timestamp>/01_original_timing`
até `05_resource_aware_visual_gated`, cada um com `run.log`, além de
`pilot.log` com o cooldown. Antes e depois de
cada run o script registra timestamp, modo, temperatura, `get_throttled`, commit
e hostname. Em qualquer falha ele para e preserva os logs.

Entre as primeiras quatro execuções há cooldown obrigatório de 180 s. Depois,
o script só continua quando `measure_temp` for estritamente menor que 50 °C;
reconfere a cada 60 s. Após 600 s ele apenas emite warning e continua esperando:
10 minutos não autorizam iniciar o próximo modo quente. Ajustes explícitos são
`COOLDOWN_MIN_SECONDS`, `COOL_TEMP_C`, `RECHECK_SECONDS` e
`COOLDOWN_EXPECTED_MAX_SECONDS`. Falha ou parsing inválido de `measure_temp`
interrompe o piloto.

Os dois scripts possuem `DRY_RUN=1` para validar a montagem de paths/comandos
sem requerer Pi, modelos ou `vcgencmd`:

```bash
DRY_RUN=1 ./scripts/smoke_raspberry.sh
DRY_RUN=1 ./scripts/pilot_5_modes_raspberry.sh
```

Exemplo de cópia manual dos modelos, sem fixar usuário/IP:

```bash
scp infra/models/frame_selector.tflite <user>@<raspberry>:/caminho/do/projeto/infra/models/
scp infra/models/sheep_weight_predictor.tflite <user>@<raspberry>:/caminho/do/projeto/infra/models/
```
