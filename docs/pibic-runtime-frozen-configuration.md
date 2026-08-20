# PIBIC Runtime — Frozen Configuration

Documento canônico e fonte de verdade operacional do runtime de borda (Edge AI MAS) para o projeto PIBIC. Registra exclusivamente as semânticas, invariantes, parâmetros e regras matemáticas congeladas no código de produção.

---

## 1. Visual Event (Detector de Atividade Visual Online)

- **Quality Gate Conjuntivo:**
  $$\text{INVALID} = (p_{99} \ge 2230.0\text{ mm}) \land (\text{fraction}_{2500\text{mm}} \ge 0.0027473958333333335)$$
- **Região de Interesse (ROI Central B):**
  $$\text{ROI}_B = [y_0=30\%, y_1=70\%, x_0=20\%, x_1=80\%] \implies \text{slice}(72..168, 64..256)\text{ para } (240, 320)$$
- **Diferença Temporal:**
  $$\text{diff} = |\text{curr\_roi}_{\text{float32}} - \text{prev\_roi}_{\text{float32}}|$$
- **Threshold por Pixel na ROI:**
  $$\text{mask} = (\text{diff} \ge 200.0\text{ mm})$$
- **Rotulagem Espacial e Conectividade:**
  $$\text{8-conectividade com elemento estruturante } 3 \times 3 \text{ de uns: } \texttt{ndimage.label(mask, structure=np.ones((3, 3)))}$$
- **PDI / Coherence Score:**
  $$\text{score} = \frac{\text{área da maior componente conexa}}{\text{total de pixels alterados na ROI}} \quad (0.0\text{ se } \text{total\_alterados} = 0)$$
- **Threshold de Decisão PDI:**
  $$\text{moving} = (\text{score} \ge 0.08747855917667238)$$
- **Visual Idle Patience (Histerese):**
  $$\text{patience} = 3 \text{ observações consecutivas com } \text{moving} == \text{False para transitar } \text{ACTIVE} \to \text{IDLE}$$
- **Semântica Estrita de Frame INVALID:**
  - Não conta como movimento ($\text{moving} = \text{None}$);
  - Não conta como ausência de movimento ($\text{no\_motion\_count} = 0$);
  - Limpa histórico temporal ($\text{previous\_raw} = \text{None}, \text{previous\_valid} = \text{False}$);
  - Preserva o estado $\text{visual\_state}$ ($\text{ACTIVE}/\text{IDLE}$) anterior sem disparar transição;
  - Próximo frame $\text{VALID}$ vira baseline ($\text{previous\_raw} = \text{current}, \text{score} = \text{None}$);
  - Somente o segundo frame $\text{VALID}$ consecutivo volta a computar score temporal.

---

## 2. Selection Hold

- **Janela de Rejeições:** $N = 2$ rejeições consecutivas.
- **Regras Operacionais:**
  - $\text{accepted} == \text{True} \implies \text{hold\_active} = \text{True}, \text{consecutive\_rejections} = 0$;
  - $1^{\text{a}}$ rejeição com hold ativo $\implies \text{consecutive\_rejections} = 1$, mantém hold ativo;
  - $2^{\text{a}}$ rejeição consecutiva $\implies \text{consecutive\_rejections} = 2$, libera hold ($\text{hold\_active} = \text{False}$);
  - Frame $\text{accepted}$ entre rejeições $\implies \text{consecutive\_rejections} = 0$, renova o hold.
- **Invariantes:**
  - O Selection Hold **NUNCA** provoca upshift $\text{LOW} \to \text{HIGH}$ isoladamente;
  - Atua unicamente impedindo o downshift prematuro quando o Visual transita para $\text{IDLE}$;
  - Protegido por $\text{passage\_id}$: evidências tardias de passagens anteriores são descartadas sem afetar a passagem ativa.

---

## 3. Capture Policy (Políticas de Taxa de Amostragem)

- **Modo LOW:** Valor em aberto (avaliado operacionalmente entre 4 FPS e 5 FPS);
- **Modo MEDIUM:** Reservado na CLI para extensões futuras; **NÃO** participa do controle atual;
- **Modo HIGH:** Rastreamento do **Original Temporal Trace**, respeitando os timestamps relativos armazenados de cada passagem, sem nenhum limitador artificial global.

---

## 4. Visual-Gated Routing (Roteamento Reativo de Frames)

- **Em Estado IDLE:**
  - Frame adquirido em $\text{LOW}$ é registrado no $\text{FrameStore}$ e tem lease retida para o Visual;
  - O evento para o pipeline pesado é retido em $\text{\_pending\_low\_frames}$ (não vai ao Selection).
- **Trigger Forwarding ($\text{IDLE} \to \text{ACTIVE}$):**
  - O Visual processa o frame e detecta transição $\text{IDLE} \to \text{ACTIVE}$;
  - O capturador recebe o $\text{VisualStateEvent}$ com $\text{is\_trigger} == \text{True}$;
  - O **mesmo** $\text{frame\_id}$ é imediatamente emitido ao Selection (zero-copy);
  - O capturador sobe para $\text{HIGH}$ para capturas subsequentes;
  - A lease do Visual é liberada sem destruir o array no $\text{FrameStore}$ (mantido pela posse do pipeline principal).
- **Em Estado ACTIVE / HIGH:**
  - Todos os frames capturados no trace nativo são emitidos em paralelo ao Selection e ao Visual.
- **Downshift ($\text{HIGH} \to \text{LOW}$):**
  - Ocorre somente quando $\text{Visual} == \text{IDLE}$ **E** $\text{Selection Hold} == \text{False}$;
  - Ao efetivar o downshift, o próximo deadline $\text{LOW}$ é agendado a partir do tempo do último frame emitido ($t_{\text{atual}} + 1000/\text{LOW\_FPS}$).

---

## 5. Orchestrator

- **Prioridade de Controle:**
  $$\text{Taxa} = \begin{cases} \text{HIGH}, & \text{se } \text{Visual} == \text{ACTIVE} \lor (\text{Rate} == \text{HIGH} \land \text{Hold} == \text{True}) \\ \text{LOW}, & \text{caso contrário} \end{cases}$$
- **Controle de Recursos:**
  - $\text{ResourceManagerAgent}$ permanece puramente observacional (telemetria e blackboard em memória). Não emite comandos de throttling/downshift.

---

## 6. FrameStore e Gestão de Memória

- **Mapeamento:** Dicionário $\text{frame\_id} \to \text{ndarray}$ associado a contador de referências por leases nomeadas;
- **Zero-Copy:** Leitura via views somente-leitura ($\text{readonly\_view}$);
- **Ciclo de Vida:**
  - Leases independentes para $\text{visual}$ e $\text{main}$;
  - $\text{discard}(frame\_id)$ decremente a posse principal;
  - Remoção física do $\text{ndarray}$ ocorre estritamente quando a posse principal e todas as leases registradas atingem zero;
  - Garantia de ausência de frames órfãos após o encerramento completo do fluxo correspondente, validada pelos testes de cleanup.

---

## 7. Semântica de Encerramento (END Semantics) e Concorrência

- **Pipeline Sem Barreiras (No-Barrier):**
  - Agentes não bloqueiam à espera do encerramento de estágios downstream;
  - Passagens consecutivas transitam livremente através do anel multiagente;
  - Isolamento garantido por $\text{passage\_id}$ e sequenciador estritamente monotônico ($\text{stream\_seq}$).
- **Propagation de EndPassage:**
  - $\text{EndPassageEvent}$ atravessa todo o pipeline mesmo em passagens com zero frames admitidos no Selection;
  - Garante agregação e salvamento de métricas em $\text{PredictWeightAgent}$.
- **Finalização Limpa (EndPipeline):**
  - $\text{EndPipelineEvent}$ drena inboxes ordenadas, encerra threads de monitoramento, salva CSVs e finaliza o Twisted reactor com segurança.

---

## 8. CLI e Parâmetros Vigentes

| Parâmetro CLI | Tipo | Default de Produção | Descrição |
| :--- | :---: | :---: | :--- |
| `--low-fps` | `float` | `None` (requerido se adaptativo) | Frequência de amostragem em modo IDLE (ex: 4 ou 5 FPS) |
| `--medium-fps` | `float` | `None` | Reservado para extensão futura |
| `--visual-gated` / `--no-visual-gated` | `bool` | `False` | Habilita gating no ramo pesado durante modo LOW |
| `--selection-hold-n` | `int` | `2` | Janela de rejeições consecutivas do Selection Hold |
| `--visual-pdi-threshold` | `float` | `0.08747855917667238` | Threshold de coerência espacial da maior componente conexa |
| `--visual-pixel-threshold-mm` | `float` | `200.0` | Threshold de diferença absoluta por pixel na ROI (mm) |
| `--visual-idle-patience` | `int` | `3` | Observações consecutivas sem movimento para transitar a IDLE |

*Nota:* O parâmetro legado `--visual-mad-threshold` foi completamente removido do fluxo de execução.

---

## 9. Evidência de Paridade Numérica Estrita (Offline vs. Runtime)

- **Cohort Auditado:** 1.488 frames reais de passagens de profundidade do dataset (`data/exp1`);
- **Discrepâncias de Score:** $0$ (erro absoluto $< 10^{-7}$);
- **Discrepâncias de Decisão ($\text{moving}$):** $0$;
- **Discrepâncias de Estado ($\text{ACTIVE}/\text{IDLE}$):** $0$;
- **Suíte de Testes Automatizados:** 162 testes unitários e de integração com 100% de aprovação.

---

## 10. Parâmetros e Decisões Científicas em Aberto

1. **Taxa Ótima de LOW:** Escolha final entre 4 FPS e 5 FPS (dependente do balanço experimental energia vs. latência de detecção);
2. **MEDIUM FPS:** Configuração e política ainda não ativas;
3. **Controle Ativo de Recursos:** Thresholds e máquinas de estado térmico/CPU ($\text{SAFE}/\text{WARNING}/\text{CRITICAL}$) ainda não participam do laço fechado de controle.
