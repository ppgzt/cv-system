# Ablacao de pré-processamento do PDI

Esta análise offline responde apenas à pergunta: filtros espaciais muito baratos
melhoram o PDI atual sem modificar o detector de produção?

Ela usa o cohort operacional de 184 passagens e o target provisório já usado nas
auditorias anteriores:

```text
parcial + suited → positivo
background       → negativo
ruido            → quality gate / INVALID
```

O quality gate é `depth_p99_mm >= 2230`. Um frame inválido limpa o histórico;
o próximo válido é baseline e não gera comparação temporal. Não há comparação
atravessando passagens ou frames inválidos.

## Protocolo fechado

O baseline usa ROI B (`y=30–70%`, `x=20–80%`), `absdiff`, threshold de 200 mm,
máscara binária e `largest_component_area / changed_pixels`.

São avaliadas exatamente 11 variantes:

1. V0 baseline;
2. V1 Gaussian 3×3 e 5×5;
3. V2 median 3×3 e 5×5;
4. V3 opening, closing e opening+closing 3×3;
5. V4 Gaussian 3×3 + somente a melhor morfologia V3;
6. V5 mediana causal de três scores e média causal de três scores.

O threshold é escolhido pela mesma regra do baseline: ponto de Youden no gate
oráculo; as métricas principais e operacionais usam o gate P99 executável.
Não há busca de ROI, pixel threshold, kernels maiores, flow, background model,
downsampling ou modelos aprendidos.

## Executar

```bash
.venv/bin/python data-analysis/visual_event_preprocessing_ablation/run_ablation.py
```

Os resultados são gravados em `output/`:

- `variant_results.csv`: qualidade frame-level, métricas operacionais e custo;
- `microbenchmark.csv`: fases de latência no Mac;
- `configuration.json`: protocolo, baseline reproduzido e variante V4 escolhida.

O benchmark é somente ranking local no Mac; não representa uma medição no
Raspberry Pi.
