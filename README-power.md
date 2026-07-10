# Medição de Consumo Energético — `run_power_test.sh` / `run_power_battery.sh`

Este documento descreve **como medimos o consumo de potência** do pipeline de
pesagem de ovinos enquanto ele roda no Raspberry Pi 5, e **como o `power.csv` é
produzido** (inclusive o porquê da reconstrução a partir do `tc66.log`).

> Pipeline em si: veja [`README-mas-main.md`](README-mas-main.md).
> Análise de capacidade FPS: veja [`logs/docs/analise-fps-capacidade.md`](logs/docs/analise-fps-capacidade.md).

---

## 1. Visão geral — quem mede o quê, e onde

O experimento envolve **dois dispositivos**, e o segredo é: **o Mac é o maestro**,
porque ele é dono do voltímetro.

```
   WALL ──USB-C──▶ [ TC66C ] ──USB-C power──▶ RASPBERRY Pi 5
                     │
                     └──USB data──▶ MACBOOK  (orquestrador)
                                      │
                                      ├── roda TC66C.py   (lê o voltímetro -> power.csv)
                                      └── ssh Pi          (roda mas-main.py)
```

- **TC66C** = voltímetro USB-C. Fisicamente no **caminho de alimentação**
  (wall → TC66C → Pi), então ele mede o consumo **total do Pi** (CPU, RAM, TODO).
  O cabo de **dados** do TC66C vai pro **Mac**.
- **MacBook** = orquestrador. Roda o `TC66C.py` (lê o voltímetro pela serial) e
  dispara o pipeline no Pi via SSH. Tudo num script só (`run_power_test.sh`).
- **Raspberry Pi 5** = só executa o `mas-main.py`. Não sabe que está sendo medido.

> Por que o Mac comanda? Porque a porta serial do TC66C está nele. SSH é a
> ferramenta padrão pra coordenar dois dispositivos — a preocupação "não dá pra
> automatizar por serem dois dispositivos" é infundada.

### Pré-requisitos

| Dispositivo | Requisito |
|---|---|
| **Mac** | `pyserial` + `pycryptodome` instalados (o `TC66C.py` precisa). No repo, estão no `.venv` (`.venv/bin/python`). |
| **Mac→Pi** | SSH **por chave** (sem senha). Configurar uma vez: `ssh-copy-id ewertonsjp@192.168.0.141`. |
| **Pi** | Repo em `~/projects/cv-system`; Python do pipeline no pyenv `cv_vend_mas` (`/home/ewertonsjp/.pyenv/versions/cv_vend_mas/bin/python`). |

> ⚠️ **pyenv não aparece em SSH não-interativo.** `ssh host 'cmd'` não carrega o
> `.bashrc`, então `python` cai pra `/usr/bin/python3.13` (sem deps). Por isso o
> script usa o **caminho absoluto** do interpretador pyenv. Não tente trocar por
> `python` nu. (Detalhes em [§6 — Troubleshooting](#6-troubleshooting).)

---

## 2. O protocolo de uma run (`run_power_test.sh`)

O script roda **no Mac** e executa 5 passos, espelhando o protocolo que o dono do
rasp usava manualmente:

```
[1/5] liga TC66C.py em background            → grava tc66.log (e power.csv)
[2/5] espera a 1ª leitura de potência (~1s)  ← só então começa o experimento
[3/5] ssh Pi: mas-main.py ...                ← BLOQUEIA até o pipeline acabar
[4/5] para o TC66C + reconstrói power.csv    (do tc66.log — ver §3)
[5/5] scp puxa infra/reports/ do Pi          → metrics.json, cpu.csv, mem.csv, ...
```

### Por que cada passo

- **[2/5] esperar a 1ª leitura:** garante que o voltímetro está respondendo
  *antes* de começar o pipeline. Se a porta falhou, o `TC66C.py` morre e o script
  aborta em ~10s em vez de rodar o experimento todo sem medir nada.
- **[3/5] SSH bloqueia:** a chamada `ssh ... mas-main.py` só retorna quando o
  pipeline termina. Esse é o **limite natural** da janela de medição — começa a
  gravar antes do SSH e para logo depois dele retornar. Sem cronômetro manual.
- **[4/5] reconstrução:** ver §3 (é a parte não-óbvia).
- **[5/5] scp:** os relatórios do pipeline nascem **no Pi** (`infra/reports/<pid>/`).
  O script os puxa pro Mac pra análise junto com o `power.csv`.

---

## 3. Como o `power.csv` é medido (e por que é reconstruído do `tc66.log`)

### O que o TC66C mede
O `TC66C.py` faz polling do voltímetro a cada `TC66_INT` segundos (default **1s**).
Cada leitura vem num pacote AES-criptografado de 192 bytes, decifrado e expandido
em: `Volt[V]`, `Current[A]`, `Power[W]` (e mais, se `--all`). A cada leitura ele:

1. monta a string `s = <ISO datetime>,<Time[S]>,<Volt>,<Current>,<Power>`;
2. **`f.write(s+'\n')`** no `power.csv`;
3. **`print(s)`** no stdout.

### O problema do buffer (truncamento)
O `power.csv` é aberto com `open(out_name,'w')` — **buffer de bloco** (~8 KB). E o
`TC66C.py` só chama `f.close()` (que flushearia o buffer) dentro do
`except KeyboardInterrupt`. O problema é que o `Poll()` faz uma leitura serial que
**bloqueia até 5s**, e o `KeyboardInterrupt` disparado por `SIGINT` quase nunca é
tratado a tempo — quando o script mandava `SIGTERM` (fallback), o processo morria
**sem flushear**, e o `power.csv` ficava **truncado** no último bloco de 8 KB.

**Sintoma observado** (antes do fix): `power.csv` com 143 amostras enquanto o
`tc66.log` tinha 168 — perdíamos as ~25 últimas leituras de cada run.

### A solução: reconstruir do `tc66.log`
O `tc66.log` é o **stdout do `python -u`** (sem buffer — write-through). Cada
`print(s)` cai no arquivo **imediatamente**, no **mesmo formato** do `power.csv`.
Então, depois que o processo do TC66C está morto de fato, o script regenera o
`power.csv` a partir do `tc66.log`:

```bash
awk 'BEGIN{print "Datetime,Time[S],Volt[V],Current[A],Power[W]"}
     /^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:,.-]+,[^,]+,[^,]+,[^,]+,[^,]+/' \
    "$TC66_LOG" > "$POWER_CSV"
```

Isso (a) recria o cabeçalho, (b) filtra só as linhas de dado válidas (descarta
qualquer lixo de stderr que porventura caia no log), (c) preserva a ordem.

### Fidelidade (verificada, não assumida)
Validado numa run de 20 animais @ 5 fps comparando `tc66.log` × `power.csv`
reconstruído:

| Checagem | Resultado |
|---|---|
| `diff` entre dados do `tc66.log` e o corpo do `power.csv` | **vazio (idênticos)** |
| Linhas do `tc66.log` descartadas pelo awk | **0** |
| Duplicatas no `power.csv` | **0** |
| `Time[S]` monotônico / maior gap | ✅ / **1.1s** |
| Campos por linha | **5 em todas** |
| Última linha íntegra (sem corte do SIGTERM) | ✅ |

**Resumo:** o `tc66.log` é a fonte da verdade (unbuffered, completo); o `power.csv`
é uma cópia fiel e limpa dele, com cabeçalho. Não há dado conflitante.

---

## 4. Semântica da janela de medição (importante pra análise)

A janela de potência **não é exatamente `[início do pipeline, fim do pipeline]`**:

- **começa** na 1ª leitura do TC66C (logo antes do `[T0]`, início do pipeline);
- **termina** quando o TC66C é parado — alguns segundos **depois** do `[T1]`
  (durante a graça do SIGINT, o Pi já está em idle).

Ou seja, o `power.csv` cobre `[~T0, T1 + alguns segundos de idle]`. Isso é
**saudável**: dá o piso de idle logo após o pipeline parar. Mas se você quiser a
**energia gasta só pelo pipeline**, recorte o `power.csv` pelos timestamps
**`[T0, T1]`**, que estão gravados no `pipeline.log`:

```
[T0] início do pipeline: 2026-07-07T17:10:53-0300
[T1] fim do pipeline:    2026-07-07T17:13:39-0300 (exit=0)
```

> **Energia da run** ≈ Σ(Power[W] · Δt) sobre as amostras dentro de `[T0, T1]`,
> com Δt ≈ 1s (cadência de polling). O idle fora dessa janela é descartado.

---

## 5. Saídas

Cada run cria uma pasta `./power_runs/<MODE>_<FPS>fps_<RUN_TAG>/` (no Mac) com:

| Arquivo | Conteúdo |
|---|---|
| `power.csv` | `Datetime,Time[S],Volt[V],Current[A],Power[W]` — reconstruído do `tc66.log` (completo). |
| `tc66.log` | stdout cru do `TC66C.py` (fonte da verdade, sem buffer). |
| `pipeline.log` | saída do `mas-main.py` via SSH + marcadores `[T0]`/`[T1]`. |
| `mas-single_thread_<pid>/` | relatório puxado do Pi: `metrics.json`, `cpu.csv`, `mem.csv`, `temp.csv`, `report.md`, `debug.log`. |

---

## 6. Troubleshooting

| Sintoma | Causa / Solução |
|---|---|
| `[ERROR] Porta '/dev/...' não encontrada` | Rode `ls /dev/cu.usbmodem*` no Mac e ajuste `PORT=` no topo do script. |
| `ModuleNotFoundError: No module named 'serial'` (ou `'psutil'` no Pi) | Mac: instale `pyserial`+`pycryptodome` no `.venv`. Pi: confirme que está usando o pyenv `cv_vend_mas` (caminho absoluto em `PY_PI`), não o `python` nu. |
| `ssh: Could not resolve hostname` | Use IP (`ewertonsjp@192.168.0.141`), não nome mDNS sem `.local`. |
| SSH pede senha | Falta a chave: `ssh-copy-id ewertonsjp@192.168.0.141` (uma vez). |
| `cd: ...: No such file or directory` (no Pi) | `PI_DIR` errado. Confirme com `ssh host 'ls ~/projects/cv-system/mas-main.py'`. O `~` deve ir cru (o shell do Pi expande). |
| `scp: ...: No such file or directory` | Já corrigido: o script agora emite caminho **absoluto** (`$PWD` remoto). Não use caminho relativo no `scp`. |
| `power.csv` com poucas linhas (truncado) | Já corrigido pela reconstrução do `tc66.log` (§3). Se voltar a acontecer, verifique se o `tc66.log` tem mais linhas que o `power.csv` — ele é a fonte da verdade. |
| `Terminated: 15` no output | **É cosmético.** É o shell reportando que o processo bg do TC66C foi morto por SIGTERM. Como o `power.csv` é reconstruído do `tc66.log`, não afeta o resultado. |

---

## 7. Como rodar

### Uma run isolada (`run_power_test.sh`)
```bash
./run_power_test.sh                              # mas-single, 5 fps, rebanho completo
FPS=2 NUM_ANIMALS=20 ./run_power_test.sh         # 2 fps, só 20 animais
MODE=mas-batch FPS=3 ./run_power_test.sh         # modo batch
```

**Variáveis de ambiente** (todas opcionais — defaults no topo do script):

| Var | Default | Descrição |
|---|---|---|
| `MODE` | `mas-single` | `mas-single` \| `mas-batch` |
| `FPS` | `5` | taxa de captura simulada |
| `NUM_ANIMALS` | *(vazio = todos)* | limita o rebanho |
| `EXTRA_ARGS` | `--debug` | extras do `mas-main.py` (vazio p/ desligar) |
| `RUN_TAG` | timestamp | sufixo do nome da pasta (a bateria usa `r1`,`r2`,…) |
| `WORK_DIR` | `./power_runs` | pasta raiz das runs |
| `PORT` | `/dev/cu.usbmodemTC661` | porta serial do TC66C no Mac |
| `PI_HOST` | `ewertonsjp@192.168.0.141` | alvo SSH |
| `PI_DIR` | `~/projects/cv-system/` | repo no Pi |
| `PY` | `.venv/bin/python` | Python do Mac (TC66C.py) |
| `PY_PI` | `.../cv_vend_mas/bin/python` | Python do Pi (pyenv) |

### Bateria completa (`run_power_battery.sh`)
```bash
./run_power_battery.sh                  # usa FPS_LIST/REPS/COOLDOWN do topo do script
```
Ver [`run_power_battery.sh`](run_power_battery.sh) — percorre `FPS_LIST` inteira e
repete (`REPS` rounds), na ordem **intercalada** (1→30, 1→30, …) pra distribuir o
drift térmico entre os níveis de FPS em vez de agrupá-los. Tem cooldown entre
runs, warm-up opcional descartável, e naming com índice de repetição
(`<FPS>fps_r<R>`) sem colisão.
