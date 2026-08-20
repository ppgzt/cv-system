# Workflow de relabeling RGB do Visual Event

Este diretório mantém as anotações do Visual Event separadas dos labels
originais. A revisão humana é feita **exclusivamente nos arquivos RGB do Google
Drive**. Os scripts não procuram, abrem, baixam ou copiam imagens RGB; usam
somente os filenames presentes em `simulation_index.json`.

Os painéis depth gerados anteriormente podem permanecer em `depth_panels/`, mas
não fazem parte do workflow e não são necessários para executar os comandos.

## Escopo congelado

O builder reutiliza, sem recalibrar a PDI:

- `background` a até 3 source frames de `parcial` ou `suited`;
- `background` com o sinal PDI aprovado acima de `0.08747855917667238`, a até
  1000 ms de `parcial` ou `suited`;
- união deduplicada dos dois conjuntos.

O resultado é uma fila de 1.301 candidatos das 184 passagens operacionais:
1.077 `boundary`, 146 `high_pdi` e 78 `boundary+high_pdi`.

`candidate_reason` é apenas metadado de proveniência. Ele não é score nem
sugestão de label.

## 1. Fila principal RGB

`review_manifest.csv` é a fonte autoritativa da revisão. Ela já está ordenada
por:

```text
passage_id
capture_index
```

Assim, todos os candidatos de uma passagem ficam juntos e em ordem temporal.
Use `rgb_filename` como alvo e `rgb_prev_3` até `rgb_next_3` como referências
para navegar manualmente pela pasta correspondente no Google Drive.

Para reconstruir a fila antes de iniciar a anotação:

```bash
.venv/bin/python data-analysis/visual_event_relabeling/build_relabeling_queue.py build --force
```

Não use `--force` depois de começar a preencher `final_review`, pois ele recria
o manifesto vazio. Ao abrir o CSV em Excel/Numbers, importe `passage_id` como
texto para preservar zeros à esquerda.

## 2. Labels humanos

Preencha `final_review` diretamente com um destes três valores:

- `CLEAR_EMPTY`: nenhuma parte identificável do animal está visível;
- `ANIMAL_VISIBLE`: alguma parte identificável está visível, inclusive
  entrada ou saída parcial;
- `AMBIGUOUS`: o RGB não permite uma decisão confiável.

Não existe etapa intermediária e `NEEDS_RGB` não é aceito.

Use `notes` somente para observações relevantes. Não altere `candidate_reason`,
filenames, passagem ou índice.

## 3. Resumo por passagem

`passage_summary.csv` contém:

```text
passage_id
candidate_count
min_capture_index
max_capture_index
```

Ele permite planejar a navegação: abra uma passagem no Drive, revise o intervalo
indicado e avance para a próxima linha do resumo.

## 4. Piloto de 150 candidatos

`pilot_manifest.csv` contém 150 candidatos — 50 por `candidate_reason` — e
também está ordenado por passagem e índice para reduzir trocas de pasta.

O piloto mede apenas:

- tempo de anotação;
- clareza dos critérios;
- distribuição `CLEAR_EMPTY` / `ANIMAL_VISIBLE` / `AMBIGUOUS`.

Para preservar uma única fonte de verdade, use o `review_order` do piloto para
preencher a mesma linha em `review_manifest.csv`. O piloto não decide se RGB é
necessário: todos os 1.301 candidatos são revisados via RGB.

## 5. Validar e consolidar

Validação parcial:

```bash
.venv/bin/python data-analysis/visual_event_relabeling/build_relabeling_queue.py validate
```

Validação exigindo todos os 1.301 labels:

```bash
.venv/bin/python data-analysis/visual_event_relabeling/build_relabeling_queue.py validate --require-complete
```

Consolidação final:

```bash
.venv/bin/python data-analysis/visual_event_relabeling/build_relabeling_queue.py consolidate --require-complete
```

`relabels.csv` recebe:

- `CLEAR_EMPTY` -> `NEGATIVE`;
- `ANIMAL_VISIBLE` -> `POSITIVE`;
- `AMBIGUOUS` -> `EXCLUDE`, fora do treinamento inicial.

O ground truth futuro combina essas revisões com os labels originais:

- `ruido` -> `INVALID`;
- `parcial` e `suited` -> `POSITIVE`;
- `background + ANIMAL_VISIBLE` -> `POSITIVE`;
- `background + CLEAR_EMPTY` -> `NEGATIVE`;
- `background + AMBIGUOUS` -> `EXCLUDE`.

O workflow nunca modifica `simulation_index.json` e não atribui automaticamente
um target aos backgrounds que não fazem parte da revisão dirigida.

## Arquivos

- `review_manifest.csv`: fila RGB principal e fonte editável;
- `passage_summary.csv`: resumo para navegação por pasta/passagem;
- `pilot_manifest.csv`: amostra estratificada de 150 candidatos;
- `relabels.csv`: saída consolidada, inicialmente somente com header;
- `build_summary.json`: contagens e proveniência do escopo;
- `depth_panels/`: artefatos legados opcionais, fora do workflow atual.
