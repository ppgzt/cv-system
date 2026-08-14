# Correções da análise baseline de cinco runs

## Problemas encontrados

- O notebook anterior filtrava `SUITABLE` antes de associar decisões às capturas, deslocando os tempos adequados para o início da passagem.
- A taxa anteriormente chamada de throughput ocupado usava a janela global de inferências e incluía períodos sem trabalho.
- A capacidade isolada e o tempo de serviço observado no pipeline não estavam apresentados como referências distintas.
- Não havia auditoria por animal, matriz de confusão ordinal, testes sintéticos completos ou comparação equivalente com 10 FPS.

## Alterações realizadas

- Pareamento completo captura–seleção por animal e `log_order`, seguido do filtro `SUITABLE`.
- Auditoria de capturas, seleções, pareamento, monotonicidade, duplicações, summaries e `[FINAL]`, com warnings explícitos.
- O campo `forwarded` do `[SELECT-SUMMARY]` é preservado como contador cumulativo; o delta por animal é calculado antes da comparação com `SUITABLE`, evitando warnings falsos.
- A interpretação cumulativa de `forwarded` foi corrigida usando deltas sucessivos por run; os deltas são comparados ao número de adequados pareados.
- A correlação entre carga e atraso residual foi refeita após agregação pela mediana de cada animal entre as cinco runs.
- A diferença entre taxa média de rajada e pico em janela de 1 s foi explicitada no notebook.
- A taxa média global de adequados foi retirada da tabela baseline versus 10 FPS; o pico local por animal foi mantido.
- Foi exportada a tabela curta `article_indicators` para o artigo.
- Rajadas com limiar configurável: 1,5× mediana, 2,0× mediana e 250 ms; taxa `(N-1)/Δt`, taxa por mediana dos intervalos e tratamento `NaN` para rajada unitária.
- Janelas móveis de 250, 500, 1000 e 2000 ms com `searchsorted` e teste sintético `[0, 0.1, 0.2, 0.9, 1.1]` s.
- Razões `rho_isolated` e `rho_integrated_service`; a taxa global foi renomeada para `run_level_prediction_completion_rate`.
- Throughput em blocos ocupados reportado apenas como blocos observáveis e com sensibilidade a 0,25/0,50 s.
- Trabalho residual separado em medições diretas e evidência ordinal; nenhuma sobreposição temporal é inferida apenas da ordem textual.
- Comparação baseline/10 FPS com definições, unidades e agregação por run equivalentes.
- Resumo científico separado em observação, interpretação e hipótese.

## Validações e limitações

- Foram adicionados asserts para pareamento, filtro posterior, rajadas, janelas, atraso residual, capacidade, bloco unitário, animal sem adequado e divergência de contagens.
- Os mesmos animais aparecem nas cinco runs; análises por animal são descritivas/exploratórias e não tratam 920 linhas como réplicas independentes.
- A associação de tempo entre captura e seleção permanece ordinal porque os logs não fornecem timestamp absoluto para cada decisão.
- Arquivos brutos das runs não foram alterados.

## Outputs

As tabelas e figuras derivadas foram regeneradas em `outputs_baseline/`; o notebook corrigido é `baseline_5runs_analysis_corrected.ipynb`.
