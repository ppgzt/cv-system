# Benchmark de componentes do pipeline Edge AI

Microbenchmark **independente e isolado** que mede o tempo de serviço de cada
estágio real do pipeline de estimativa de peso de ovinos no **Raspberry Pi 5**.
Ele **não** repete a latência fim a fim (já medida nos experimentos por FPS) —
mede apenas os tempos internos: seletor de frames, enhancement, preditor de
peso e agregação final.

O benchmark **reusa as funções e interpretadores reais** do pipeline
(`FrameSelection`, `DataEnhance`, `PredictWeight`) com a **mesma configuração**
(`num_threads=2`, delegate XNNPACK default do TFLite, mesmas `preprocess_fn` /
transforms), sem reimplementar nada.

---

## 1. Pré-requisitos

- **No Raspberry Pi 5** (onde o benchmark roda de fato): o pyenv `cv_vend_mas`
  com as mesmas dependências do pipeline (tensorflow, numpy, scipy, psutil,
  scikit-image). É o mesmo ambiente do `mas-main.py`.
- Dataset em `data/exp1/` (animal-tags + DEPTH).
- Modelos em `infra/models/frame_selector.tflite` e
  `infra/models/sheep_weight_predictor.tflite`.
- **No Mac** (orquestrador, opcional): apenas `ssh`/`scp` por chave e o script
  `run_benchmark_components.sh`.

> O benchmark também roda no Mac para **dry-run** (validação de shapes/dtypes);
  nessa máquina sem scikit-image, a leitura de PNG cai para Pillow e o inventário
  lê os `simulation_index.json` diretamente (mesmos dados). **As medições
  válidas devem ser feitas no Pi** — é o alvo do estudo.

## 2. Comandos de execução

Direto no Pi (dentro do repo, com o pyenv ativo):

```bash
python benchmarks/benchmark_components.py --component selector
python benchmarks/benchmark_components.py --component enhancer
python benchmarks/benchmark_components.py --component predictor
python benchmarks/benchmark_components.py --component aggregation
python benchmarks/benchmark_components.py --component all          # sequencial

python benchmarks/benchmark_components.py --component all --dry-run        # só valida
python benchmarks/benchmark_components.py --verify-pipeline-config         # compara c/ pipeline
python benchmarks/benchmark_components.py --component enhancer --decompose-enhancer
```

Parâmetros (defaults conforme especificação): `--warmup 50`, `--iterations 1000`,
`--seed 42`, `--output-dir benchmarks/runs`, `--num-threads 2`, `--threshold 0.5`,
`--pool-size 300`, `--monitor-interval 1.0`, `--bootstrap-resamples 10000`.

Do Mac (orquestra via SSH e puxa o relatório):

```bash
cd benchmarks
./run_benchmark_components.sh                                   # all, defaults
COMPONENT=selector ./run_benchmark_components.sh
COMPONENT=predictor WARMUP=100 ITERATIONS=2000 ./run_benchmark_components.sh
COMPONENT=selector EXTRA_ARGS="--dry-run" ./run_benchmark_components.sh
```

## 3. Configuração dos modelos (idêntica ao pipeline)

| Componente | Modelo | Formato | Threads | Delegate | Entrada | Saída |
|---|---|---|---|---|---|---|
| Seletor | `frame_selector.tflite` (MobileNetV2) | TFLite | 2 | XNNPACK (default) | `[1,224,224,3]` float32 | `prob(class 0)` |
| Enhancement | `DataEnhance` (4 transforms NumPy/TF) | — | — | — | `(240,320)` uint16 mm | `(300,300,3)` float32 |
| Preditor | `sheep_weight_predictor.tflite` (EfficientNet-B3) | TFLite | 2 | XNNPACK (default) | `[1,300,300,3]` float32 | `[N,1]` kg |
| Agregação | `float(np.mean(weights))` | — | — | — | lista de predições | escalar |

Variáveis de ambiente TF setadas **antes** de importar TensorFlow (idênticas ao
`mas-main.py`): `TF_INTRA_OP_PARALLELISM_THREADS=1`,
`TF_INTER_OP_PARALLELISM_THREADS=1`, `TF_ENABLE_ONEDNN_OPTS=0`,
`KERAS_BACKEND=tensorflow`. O `num_threads=2` é o intra-op do XNNPACK passado ao
`Interpreter` (igual ao pipeline).

## 4. Definição EXATA das regiões cronometradas

Todas as durações usam `time.perf_counter_ns()`. Dentro da região cronometrada
**não há** leitura de disco, `print`, escrita de JSON/CSV, coleta de CPU/temp ou
carregamento de modelo — apenas as operações listadas.

### Seletor (`selector_measurements.csv`)
- **`total_stage`**: `_to_single_channel` + `_preprocess_fn` (tf.function real) +
  atribuição no buffer + `set_tensor` + `invoke` + `get_tensor` + classe final.
  (Equivalente ao `FrameSelection.predict()` + threshold; verificado em
  `_verify_correctness`.)
- **`tflite_total`**: `set_tensor` + `invoke` + `get_tensor`.
- **`invoke`**: somente `interpreter.invoke()`.

### Enhancer (`enhancer_measurements.csv`)
- **`total_stage`**: a função REAL `DataEnhance.run(img)` (4 transforms reais).
- **Decomposição opcional** (`--decompose-enhancer`): `noise_removal`,
  `adjust_scale`, `replicate`, `resize_pad` — mesmos objetos `transfs`, com
  timers aninhados. Secundária; não substitui o total.

### Preditor (`predictor_measurements.csv`)
Entrada = imagem **suited pré-enhanceada** pela `DataEnhance` real (fora do
cronômetro, uma vez no setup) — igual ao que chega ao estágio no pipeline.
- **`total_stage`**: `np.asarray([img])` + checagem de batch + `set_tensor` +
  `invoke` + `get_tensor` + `.copy()` + `float()`.
- **`tflite_total`**: `set_tensor` + `invoke` + `get_tensor`.
- **`invoke`**: somente `interpreter.invoke()`.

### Agregação (`aggregation_measurements.csv`)
- **`aggregation`**: `float(np.mean(weights))` (operação real do pipeline) para
  tamanhos 1, 5, 10, 20, 50. Pesos sintéticos com seed fixa. **Não se mistura
  com as latências dos modelos.**

## 5. Metodologia

- **Warm-up separado**: 50 operações (configurável) por componente antes das
  1.000 medições válidas. Os tempos de warm-up ficam em `warmup_<comp>.csv` e
  **nunca** são misturados com as medições válidas.
- **1.000 medições válidas**: só contam operações bem-sucedidas. Falhas vão
  para `failures.csv` e não substituem silenciosamente uma medição.
- **Conjunto representativo**: pool balanceado (suited/not-suited) para o
  seletor; só suited para enhancer/preditor; espalhado por animal (round-robin
  por tag). Ordem embaralhada com **seed fixa** e cíclica (não mede 1.000× a
  mesma imagem). Reuso por imagem é registrado em `metadata.json`.
- **Monitoramento**: thread daemon de 1 Hz (CPU total/proc, RSS, memória, temp,
  freq, throttle) **fora** da região cronometrada, com PID explícito e coluna
  `component` identificando o estágio ativo.
- **Estatísticas**: média, mediana, dp, variância, min/max, amplitude, CV, SEM,
  IC 95% da média (t de Student), IC 95% da mediana e dos percentis p1–p99
  (bootstrap ≥ 10.000, seed fixa), IQR, MAD, skewness, kurtosis, inclinação
  temporal, correlações latência×{temp,freq,CPU}, contagem de outliers IQR
  (sem remoção automática). Uma análise **secundária** sem outliers IQR é
  fornecida, claramente identificada.
- **Análise temporal**: blocos de 100 iterações + comparação primeira metade
  vs. segunda / primeiras 100 vs. últimas 100.

## 6. Estrutura dos resultados

```
benchmark_components_YYYY-MM-DD_HH-MM-SS/
├── metadata.json              # config + hw + sw + modelos (sha256/size) + pools + bootstrap
├── selector_measurements.csv  # 1 linha por medição válida
├── enhancer_measurements.csv
├── predictor_measurements.csv
├── aggregation_measurements.csv
├── warmup_<component>.csv     # medições de warm-up (separadas)
├── system_monitor.csv         # amostras 1 Hz (CPU/mem/temp/freq/throttle)
├── failures.csv               # falhas de validação/runtime
├── summary.json               # estatística completa por componente/métrica
├── summary.csv                # versão tabular compacta
├── report.md                  # relatório legível
├── article_table.{csv,md,tex}         # tabela compacta p/ o artigo
├── article_table_detailed.{csv,md,tex}
└── plots/                     # PNG + PDF (opcional, se matplotlib presente)
```

## 7. Como validar que a configuração é idêntica à do pipeline

```bash
python benchmarks/benchmark_components.py --verify-pipeline-config
```

Verifica: caminhos dos modelos e dataset (iguais ao pipeline), existência +
SHA-256, `num_threads=2`, `threshold=0.5`, shapes/dtypes dos interpretadores
reais, constantes de preprocessing (`_IMG_SIZE=224`, `_CLIP_MAX=4000`,
`_NORM_SCALE=2000`), parâmetros dos transforms do `DataEnhance`, env vars do
TF e delegates. Emite `OK`/`WARN`/`FAIL` por item e retorna código diferente de
zero quando há falhas; onde não é possível verificar automaticamente (ex.:
delegate XNNPACK não é introspectável), emite `WARN` explícito.

A verificação de **corretude** também roda automaticamente antes das medições
de seletor/preditor: o caminho instrumentado é comparado com a função real
`predict()` (diff de prob/peso registrada em `summary.json → correctness`).

## 8. Limitações metodológicas

- O benchmark mede cada componente **isoladamente e sequencialmente**. No
  pipeline real, os estágios rodam sobrepostos em threads (1 worker por estágio
  no `thread_pipeline`). Portanto os tempos aqui são de **serviço isolado**, não
  de throughput concorrente — o que é exatamente o objetivo deste benchmark.
- Decomposição selector/preditor (`tflite_total`, `invoke`) é obtida executando
  a **mesma sequência do `predict()` real** no mesmo interpretador, apenas
  intercalando `perf_counter_ns()`. O overhead dos timers (~poucas dezenas de ns)
  é consistente entre as medições e desprezível frente às latências (ms).
- A correlação latência×ambiente alinha séries de 1 Hz (monitor) com séries por
  iteração (latência) pelo timestamp monotônico mais próximo — é uma
  aproximação, não uma medição simultânea estrita.
- Delegates: selector/preditor usam o delegate CPU padrão do TFLite (XNNPACK
  quando disponível); o pipeline também não seta delegates explicitamente.
  A lista de delegates automáticos não é verificável de forma estável pela API
  pública do interpretador.
- Não removemos outliers: a análise sem outliers (IQR) é secundária e marcada.
