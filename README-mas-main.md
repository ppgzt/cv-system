# `mas-main.py` — Pipeline MAS data-driven por tag

Entry point do pipeline de pesagem de ovinos via Sistema Multi-Agente (PADE +
Twisted), com **captura data-driven**: os frames depth são lidos diretamente do
dataset real em `data/exp1`, simulando uma câmera real definida por **FPS**.

Este é o pipeline **novo** e paralelo ao `main.py` / `domain/pipelines.py`
(baseline `SingleStream`/`Batch` e o `MASStrategy` antigo seguem intocados e
utilizáveis). O `mas-main.py` não passa pelo `main.py` — ele instancia
`MASStrategy` de `mas_pipeline.py`.

---

## 1. Visão geral

O fluxo dos agentes é uma cadeia linear (ontologias FIPA-ACL INFORM):

```
DatasetCaptureAgent  ──frame-capture──▶  FrameSelectionAgent  ──frame-selected──▶  DataEnhanceAgent  ──frame-enhanced──▶  PredictWeightAgent
        │                                       │
        └────────── passage-complete ──────────▶│
                                                └────────── batch-ready ──────────▶ PredictWeightAgent
```

- **`DatasetCaptureAgent`** — lê os frames depth reais de `DEPTH/<tag>/`,
  ritmados por FPS (nearest-neighbor no `relative_time_ms`), e publica apenas
  metadados leves (a imagem vai para um buffer em memória, não na mensagem).
- **`FrameSelectionAgent`** — roda o seletor TFLite no depth **raw**; encaminha
  só os frames *suitable* ao enhance e descarta o resto (liberando o buffer).
- **`DataEnhanceAgent`** — realça o frame *suitable* (só ~os que passaram) e
  sobrescreve o raw no buffer.
- **`PredictWeightAgent`** — roda o regressor de peso TFLite no frame
  enhanced; agrega por animal e, ao finalizar todos, salva `metrics.json` e
  para o reator.
- **`ResourceManagerAgent`** — monitora CPU/RAM em background e escreve
  `cpu.csv` / `mem.csv`; para no shutdown do reator.

A imagem (pesada) trafega por um **buffer em memória** (`FRAME_BUFFER`,
chaveado por `frame_id`); as mensagens FIPA-ACL carregam só JSON leve
(`frame_id`, `animal_id`, `elapsed_time`, `label`, …).

---

## 2. Modelo de captura (data-driven, FPS-paced)

Cada animal tem, em `data/exp1/animal-tags/<tag>/simulation_index.json`, uma
lista de frames reais com `relative_time_ms` (tempo de captura real dentro da
passagem, irregular — nativo ~10 fps), `depth_filename` e `label`
(ground-truth ∈ {`background`, `parcial`, `suited`, `ruido`}).

O `DatasetCaptureAgent` simula uma câmera real definida por **FPS**:

1. Um `TimedBehaviour` (Twisted) **pulsa a cada `1/FPS` segundos de wall-clock**.
2. Mantém um **relógio virtual** em *ms* que avança `1000/FPS ms` por pulso,
   iniciando em `t0` do animal (primeiro `relative_time_ms`).
3. A cada pulso, captura o frame cujo `relative_time_ms` é o **mais próximo**
   do relógio virtual (nearest-neighbor, `np.searchsorted`, O(log n)):
   - **FPS < frequência nativa** → alguns frames reais nunca são capturados
     (perda de frames, desejada na simulação).
   - **FPS > frequência nativa** → o mesmo frame real é capturado mais de uma
     vez (duplicação, desejada na simulação).
4. A **duração da passagem** de cada animal vem do dado (`tmax` = maior
   `relative_time_ms`); não há `passage_time`/`arrival_time` fixos. Quando o
   relógio virtual ultrapassa `tmax`, o agente emite `passage-complete` e avança
   para o próximo animal (ordem alfabética da tag), **pré-carregando** o
   `simulation_index.json` do próximo na memória.
5. Ao terminar o último animal, a captura para; o reator para só depois que o
   `PredictWeightAgent` finaliza todos (garantia de que as predições são salvas).

> Por que wall-clock e não "o mais rápido possível": preserva o ritmo real de
> uma câmera, mantendo válidas as medições de throughput/latência e o teto de
> capacidade (stall λ>μ) do MAS. Veja [§10. Capacidade](#10-capacidade).

### 2.1. Reprodução pelos timestamps originais (engine `thread`)

Para reproduzir cada passagem usando todos os frames e os intervalos originais
de `relative_time_ms`, sem nearest-neighbor, use:

```bash
python mas-main.py mas-single --native-timestamps
python mas-main.py mas-single --native-timestamps --num-animals 3
```

Esse modo está disponível somente no engine padrão `thread`. O `fps` pode ser
omitido; se for informado por compatibilidade, não participa do agendamento.
Cada frame do `simulation_index.json` é emitido uma única vez, usando deadlines
baseados em relógio monotônico. O pipeline mantém as mesmas quatro threads e as
mesmas filas do modo FPS; nenhum worker adicional é criado. O modo original
continua sendo ativado pelos comandos existentes, por exemplo:

```bash
python mas-main.py mas-single 5 3
```

Os **metadados** entre agentes (payload `frame-capture`) passam a carregar:
`frame_id`, `animal_id` = **tag**, `frame_index`, `elapsed_time` (relógio
virtual ms ou timestamp original), `label` (ground-truth) e `depth_filename`.
No modo nativo, `dataset_timestamp_ms` também registra explicitamente o
`relative_time_ms` usado no agendamento.

### Single vs Batch
- **`mas-single`**: cada frame *suitable* é inferido **imediatamente**
  (`PredictWeight` com batch=1). Latência por frame, peso por frame no log.
- **`mas-batch`**: os frames *suitable* são acumulados por animal; ao receber
  `batch-ready`, o `PredictWeight` roda **uma** inferência em lote sobre todos.
  Menos overhead, peso só ao final do animal.

O critério de **término** é por conjunto: o `PredictWeightAgent` conta
finalizações até atingir o total de animais do rebanho (`herd_size` =
quantidade de tags), robusto a qualquer ordem de finalização.

---

## 3. Pré-requisitos

- Python 3.13 + venv do projeto (`.venv`), com `tensorflow`, `twisted`, `pade`,
  `scikit-image`, `psutil`, `python-dotenv`, `click`, `numpy`.
- Dataset em `data/exp1`:
  - `animal-tags/<tag>/simulation_index.json`
  - `DEPTH/<tag>/<depth_filename>` — PNG depth **uint16 mm** (shape `240×320`).
    O loader **não reescala**: o seletor clipa em 4000 e o DataEnhance em 1950
    (esperam milímetros crus, igual ao treino).
- Modelos em `infra/models/`:
  - `frame_selector.tflite` (seletor, 4 classes, decisão `prob(class0) > 0.5`).
  - `sheep_weight_predictor.tflite` (regressor de peso, entrada enhanced `300×300`).
- Arquivo `.env` (veja `.env.example`): `SMA_AMS_HOST`, `SMA_AMS_PORT`,
  `SMA_AGENT_HOST`, `SMA_AGENT_BASE_PORT`.

---

## 4. Uso

```
python mas-main.py <mode> <fps> [num_animals] [max_passage_seconds] [--debug]
```

| Argumento             | Obrigatório | Descrição                                                        |
|-----------------------|-------------|------------------------------------------------------------------|
| `mode`                | sim         | `mas-single` \| `mas-batch`                                       |
| `fps`                 | sim         | taxa de captura simulada (frames por segundo)                    |
| `num_animals`         | não         | quantas tags processar (default: **todas**, em ordem alfabética) |
| `max_passage_seconds` | não         | cap do span de cada animal em segundos (default: sem cap)        |
| `--debug`             | não         | modo verbose + grava todo o log em `debug.log`                   |

> Os dois posicionais opcionais também aceitam nomes longos via argparse
> (`num_animals`, `max_passage_seconds`); `--debug` é flag.

### Exemplos

```bash
# 3 animais a 5 fps (teste rápido)
python mas-main.py mas-single 5 3

# Todos os animais a 10 fps, em modo batch
python mas-main.py mas-batch 10

# 3 animais com log detalhado (label real × classificação × peso)
python mas-main.py mas-single 5 3 --debug

# Todos os animais, cap de 30s por animal (rede de segurança p/ anomalias), debug
python mas-main.py mas-single 5 192 30 --debug

# Apenas ajuda
python mas-main.py --help
```

---

## 5. Modo `--debug`

Além do log normal, o `--debug`:

1. **Salva TODO o log** em `infra/reports/<pid>/debug.log` via um *Tee* de
   stdout/stderr (captura mensagens dos agentes, prints e tracebacks do
   reator). O `pid` é gerado como `<mode>_<ISO timestamp>`.
2. Liga o **log verbose por frame** com a trilha de inspeção:

   - **Captura**: `[CAPTURE] animal=03mf idx=5 t=1234.0ms label=suited -> frame_selection_agent`
   - **Seleção** (label real × decisão do seletor × probabilidade):
     `[SELECT] frame_id=abc animal=03mf label=suited -> SUITABLE (p=0.9164)`
     ou `-> DISCARDED (p=0.0000)`.
   - **Resumo do seletor por animal** (ao finalizar cada animal):
     `[SELECT-SUMMARY] animal=03mf total=N | label 'suited' captados=K (TP=.., FN=..) | não-suited marcados suitable (FP)=..`
   - **Predição**: `[PREDICTION] animal_id=03mf frame_id=abc label=suited weight=16.0064 kg`
   - **Final do animal**: `[FINAL] Animal 03mf: n_suitable=2 | labels_dos_suitable={'suited':2} | peso_medio=16.1000 kg`

Sem `--debug`, o comportamento e os logs são enxutos (como antes).

---

## 6. Saídas

Tudo em `infra/reports/<pid>/`:

| Arquivo        | Conteúdo                                                                 |
|----------------|--------------------------------------------------------------------------|
| `metrics.json` | Por animal (chave = **tag**): timestamps, `total_of_images`, `suitable_images`, `weight_prediction_final` e `imgs{}` (com `label` por frame quando disponível). |
| `cpu.csv`      | Amostras de uso de CPU por core (tempo real).                            |
| `mem.csv`      | Amostras de memória (total/used/percent/…).                              |
| `debug.log`    | Só com `--debug`: log completo da execução.                             |

> **Atenção:** `metrics.json` agora é indexado pela **tag** (ex. `"03mf"`) e
> não mais por índice inteiro (`"1","2"`). A comparação com runs antigas não é
> posicional, mas o conteúdo por animal é paralelo.

---

## 7. Estrutura de dados esperada

```
data/exp1/
├── animal-tags/
│   ├── 03mf/simulation_index.json     # [{relative_time_ms, depth_filename, rgb_filename, label}, ...]
│   ├── 0014/simulation_index.json
│   └── 0014s2/simulation_index.json   # 2ª passagem do 0014 (split, t rezeroado)
└── DEPTH/
    ├── 03mf/<depth_filename>.png      # uint16 mm, 240×320
    ├── 0014/<depth_filename>.png
    └── 0014s2/<depth_filename>.png
```

---

## 8. Considerações práticas

- **Anomalias de dado**: o `0014` original tinha `tmax` ≈ 17 min (duas
  passagens do mesmo animal mescladas); foi dividido em `0014` + `0014s2`. Se
  surgir outra tag com span anômalo (>120 s), o agente emite um `[WARN]` — use
  `max_passage_seconds` para capar.
- **`num_animals`**: para iteração rápida, processe poucos animais
  (ex. `... 5 3`). Default = todas as tags (atualmente 193).
- **Threads/cores**: `TF_INTRA_OP`/`TF_INTER_OP_PARALLELISM_THREADS=1` (em
  `mas-main.py`) e o pool do Twisted em `minthreads=2, maxthreads=4`
  (`mas_pipeline.py`, ajustado para o Pi 5, 4 cores). Os modelos TFLite rodam
  com `num_threads=2` (XNNPACK CPU).
- **Concorrência de inferência serializada**: seletor e regressor de peso são
  despachados via `deferToThread`, mas guardados por um **`DeferredSemaphore(1)`**
  compartilhado (1 inferência por vez) — em `PredictWeightAgent` e
  `FrameSelectionAgent`. Isso impede que vários animais prontos disparem
  inferências TF/TFLite concorrentes que se atropelariam nos 4 cores
  (*thrashing*, que era o que travava o pipeline antigo sem semáforo em
  ≥2 fps). Veja a [§10. Capacidade e concorrência](#10-capacidade).
- **Capacidade (teto de FPS)**: o limite operacional depende do hardware e do
  modelo — os números antigos (1.5/2 fps) medidos no pipeline **pré-semáforo**
  (Keras, threadpool=30) **estão obsoletos** e devem ser re-medidos no código
  atual (semáforo + TFLite). Veja [§10](#10-capacidade) e o documento de
  análise em `logs/docs/analise-fps-capacidade.md`.
- **dtype depth**: uint16 mm — não reescalar. O loader (`AnimalDataset`) usa
  `skimage.io.imread` preservando o tipo.
- **Hardware**: CPU only (M1 para teste, RPi 5 para deploy) — sem GPU.

---

## 9. Diferenças vs `main.py` (antigo)

| Aspecto                | `main.py` / `MASStrategy` antigo                | `mas-main.py` / `mas_pipeline.py`                  |
|------------------------|--------------------------------------------------|----------------------------------------------------|
| Captura                | `sample.png` repetido, `passage/arrival_time` fixos | frames depth reais por tag, FPS-paced (nearest-neighbor) |
| Identidade do animal   | inteiro `1..herd_size`                           | **tag** string (ex. `03mf`)                         |
| Duração da passagem    | parâmetro `passage_time`                         | derivada do dado (`tmax`)                          |
| Label nos metadados    | —                                                | **sim** (ground-truth viaja no payload)            |
| Término                | `animal_id == herd_size`                         | conjunto de finalizações = total de tags           |
| CLI                    | `strategy herd passage arrival fsel fsel_win`    | `mode fps [num_animals] [max_passage] [--debug]`   |

Os agentes `FrameSelectionAgent`, `DataEnhanceAgent` e `PredictWeightAgent`
são **compartilhados**; apenas `DatasetCaptureAgent` é novo e o critério de
término foi tornado compatível com ambos (int e tag-string).

---

<a id="10-capacidade"></a>
## 10. Capacidade, concorrência e limites de FPS

> Análise completa em `logs/docs/analise-fps-capacidade.md`. Esta seção resume
> o que é **fato no código atual** (verificável) e o que ainda **precisa ser
> re-medido**.

### Por que existe um teto de FPS (e ele não é da câmera)

Aumentar o FPS **não** aumenta o número de animais processados por minuto. A
chegada de animais (λ) é ditada pelo ritmo da passagem (cada animal tem um
`tmax` no dado), e o serviço por animal (μ) cresce com o FPS — mais frames
capturados → mais `enhance`/`select`/`predict` por animal. A partir de certo
FPS, **λ > μ** e o backlog cresce sem parar (stall), o que leva o watchdog a
forçar a finalização dos animais restantes como *placeholders* (sem predição
real). É por isso que a captura é **wall-clock paced** (não "o mais rápido
possível"): preserva o ritmo real e mantém a medição de throughput válida.

### Como o código atual controla a concorrência (evita o thrashing)

Versões antigas do MAS disparavam uma inferência `deferToThread` por animal
**sem limite de concorrência** — quando vários animais ficavam prontos juntos
(em ≥2 fps), dezenas de threads TF/TFLite competiam nos 4 cores, e a inferência
degradava de ~2s para até ~50s (*thrashing*). O código atual corrige isso com:

- **`DeferredSemaphore(1)`** em `PredictWeightAgent` (linhas 296/325) e
  `FrameSelectionAgent` (linha 193): **no máximo uma inferência de cada tipo por
  vez**, independentemente de quantos animais estejam prontos.
- **TFLite/XNNPACK** com `num_threads=2` (em vez de Keras multi-thread).
- **`TF_INTRA/INTER_OP_PARALLELISM_THREADS=1`** e pool do Twisted em
  `minthreads=2, maxthreads=4`.

Com isso, a inferência sob carga degrada de forma controlada (próximo do
linear), não catastrófica.

### ⚠️ Números de capacidade obsoletos — re-medir

Os runs em `logs/mas-batch-1_5fps-100a/` (100/100) e
`logs/mas-batch-2fps-100a/` (58/100 reais + 42 *placeholders*) são do pipeline
**pré-semáforo, em Keras, threadpool=30**. Eles **não** representam o
comportamento do `mas-main.py` atual. Antes de concluir qualquer teto de FPS
para o deploy no Pi 5, **re-meça** com o código atual (semáforo + TFLite),
provavelmente o 2 fps passa a ser viável onde antes colapsava.

### Critério para novas medições

- Varre FPS ∈ {1, 1.5, 2, 3, …} e confere se todos os animais finalizam com
  predição **real** (sem *placeholders* do watchdog).
- Monitore `mem.csv`/`cpu.csv` e o `queue_depth` (se disponível): o sistema
  está em λ ≤ μ enquanto o backlog drena no fim; está em λ > μ quando o
  `frame_buffer` cresce monotonicamente e o watchdog dispara.
- Reporte o FPS máximo em que **100% dos animais** têm peso real como o teto
  operacional daquela configuração (hardware + modelo + semáforo).
