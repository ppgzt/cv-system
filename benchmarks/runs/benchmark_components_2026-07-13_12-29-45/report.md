# Benchmark de componentes do pipeline Edge AI

- Gerado em (UTC): 2026-07-13T15:29:50.879110Z
- Semente: 42 | warm-up: 5 | iterações: 30
- Host: MacBook-Air-de-Gabriel.local | dispositivo: not_available
- Temperatura inicial/final: None °C / None °C

> ⚠ **Throttling ocorreu** desde o boot (raw=not_available). Ver system_monitor.csv.

## 1. Metodologia

- Cada componente é medido de forma **sequencial e isolada**, após warm-up próprio.
- Reuso das **funções/interpretadores reais** do pipeline (FrameSelection, DataEnhance, PredictWeight), com num_threads=2 e delegate XNNPACK (default TFLite).
- `time.perf_counter_ns()` em todas as durações. Nenhuma escrita em disco/print dentro das regiões cronometradas.
- Decomposição selector/predictor: mesma sequência do `predict()` real instrumentada (verificação de equivalência em cada componente abaixo).

## 2. Regiões cronometradas (resumo)

| Componente | total_stage inclui | tflite_total | invoke |
|---|---|---|---|
| selector | to_single_channel + preprocess_fn + set_tensor + invoke + get_tensor + classe | set+invoke+get | invoke |
| enhancer | `DataEnhance.run(img)` (4 transforms) | — | — |
| predictor | asarray + set_tensor + invoke + get_tensor + copy + float | set+invoke+get | invoke |
| aggregation | `float(np.mean(weights))` | — | — |

## 3. Resultados por componente

### selector

- solicitadas: 30 | concluídas: 30 | válidas: 30 | falhas: 0 | warm-up: 5
- corretude (instrumentado vs real): diff=0.0 match=True
- razões: proportion_tflite_of_total=0.8646, proportion_prep_of_total=0.1354, prep_overhead_mean_ms=1.436, ratio_prep_over_tflite=0.1566, proportion_invoke_of_tflite=0.9956

| métrica | n | média ms | mediana ms | dp ms | p95 ms | p99 ms | CV | outliers IQR |
|---|---|---|---|---|---|---|---|---|
| total_stage_ns | 30 | 10.6 | 9.646 | 3.426 | 13.17 | 23.57 | 0.3231 | 4 |
| tflite_total_ns | 30 | 9.168 | 8.489 | 1.679 | 11.94 | 15.09 | 0.1831 | 4 |
| invoke_ns | 30 | 9.128 | 8.459 | 1.656 | 11.9 | 14.95 | 0.1814 | 4 |
- inclinação temporal: -9.717e+04 ns/iter (r=-0.2497)

### enhancer

- solicitadas: 30 | concluídas: 30 | válidas: 30 | falhas: 0 | warm-up: 5

| métrica | n | média ms | mediana ms | dp ms | p95 ms | p99 ms | CV | outliers IQR |
|---|---|---|---|---|---|---|---|---|
| total_stage_ns | 30 | 3.274 | 2.742 | 2.632 | 3.66 | 13.21 | 0.8038 | 2 |
- inclinação temporal: -9.131e+04 ns/iter (r=-0.3054)

### predictor

- solicitadas: 30 | concluídas: 30 | válidas: 30 | falhas: 0 | warm-up: 5
- corretude (instrumentado vs real): diff=0.0 match=True
- razões: proportion_tflite_of_total=0.9939, proportion_prep_of_total=0.006084, prep_overhead_mean_ms=0.6498, ratio_prep_over_tflite=0.006121, proportion_invoke_of_tflite=0.9992

| métrica | n | média ms | mediana ms | dp ms | p95 ms | p99 ms | CV | outliers IQR |
|---|---|---|---|---|---|---|---|---|
| total_stage_ns | 30 | 106.8 | 94.9 | 27.93 | 166.4 | 194.2 | 0.2615 | 5 |
| tflite_total_ns | 30 | 106.2 | 94.36 | 27.85 | 164.7 | 193.8 | 0.2624 | 5 |
| invoke_ns | 30 | 106.1 | 94.32 | 27.86 | 164.6 | 193.7 | 0.2626 | 5 |
- inclinação temporal: -1.037e+06 ns/iter (r=-0.327)

### aggregation

- solicitadas: 150 | concluídas: 150 | válidas: 150 | falhas: 0 | warm-up: 5

| métrica | n | média ms | mediana ms | dp ms | p95 ms | p99 ms | CV | outliers IQR |
|---|---|---|---|---|---|---|---|---|
| aggregation_ns | 150 | 0.006113 | 0.005708 | 0.001383 | 0.007046 | 0.01382 | 0.2263 | 4 |
- inclinação temporal: 6.772 ns/iter (r=0.2127)
