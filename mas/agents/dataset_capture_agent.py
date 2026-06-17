"""Dataset Capture Agent — captura data-driven por tag, ritmada por FPS.

Substitui o simulador biológico sintético (sample.png + passage/arrival
fixos) por uma captura fiel à realidade, lendo os frames depth reais de
data/exp1/DEPTH/<tag>/.

Mecânica (FPS-paced, wall-clock):
  - Um TimedBehaviour pulsa a cada 1/FPS segundos reais.
  - Mantém um relógio virtual em ms que avança 1000/FPS ms por pulso.
  - A cada pulso captura o frame cujo relative_time_ms é o MAIS PRÓXIMO
    do relógio virtual (nearest-neighbor no tempo):
      * FPS < frequência nativa  -> alguns frames reais são pulados.
      * FPS > frequência nativa  -> o mesmo frame real é repetido.
    Ambos são desejados para a simulação.
  - A duração da passagem de cada animal vem do dado (tmax do
    simulation_index.json); sem passage_time/arrival_time.
  - O simulation_index.json é PRÉ-CARREGADO na memória do agente assim
    que ele começa a capturar aquele animal (nearest-neighbor O(log n)).

Os metadados entre agentes (payload do frame-capture) carregam:
  frame_id, animal_id=<tag>, frame_index, elapsed_time=<relógio virtual ms>,
  label (ground-truth), depth_filename.
"""

import json
import threading
import uuid
from datetime import datetime

import numpy as np

from twisted.internet import reactor

import mas  # noqa: F401  (sys.path hack para pade/infra)

from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from pade.behaviours.protocols import TimedBehaviour
from pade.core.agent import Agent
from pade.misc.utility import display_message

from mas.utils.animal_dataset import AnimalDataset
from mas.utils.globals import FRAME_BUFFER


class DatasetCaptureBehaviour(TimedBehaviour):
    """Pulsa a 1/FPS s; a cada pulso captura o frame depth mais próximo do
    relógio virtual do animal corrente."""

    ANOMALY_SPAN_SECONDS = 120.0

    def __init__(
        self,
        agent: Agent,
        dataset: AnimalDataset,
        next_agent_aid: str,
        selection_agent_aid: str,
        animal_tags: list[str],
        fps: float,
        max_passage_seconds: float | None = None,
        predict_agent=None,
        verbose: bool = False,
    ):
        super().__init__(agent, 1.0 / fps)
        self.dataset = dataset
        self.next_agent_aid = next_agent_aid          # FrameSelectionAgent
        self.selection_agent_aid = selection_agent_aid  # recebe passage-complete
        self.animal_tags = animal_tags
        self.fps = fps
        self.step_ms = 1000.0 / fps
        self.max_passage_seconds = max_passage_seconds
        self.predict_agent = predict_agent            # notify in-process ao Predict
        self.verbose = verbose

        self.tag_idx = 0
        self._times: np.ndarray | None = None   # relative_time_ms ordenado
        self._frames: list[dict] = []           # paralelo a _times
        self._virtual_clock = 0.0               # ms, dentro da passagem do animal

        self.captured_count = 0
        self.first_capture = None
        self.last_capture = None
        self._finished = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def _load_animal(self, tag: str):
        """Pré-carrega o simulation_index.json do animal na memória."""
        index = self.dataset.load_index(tag)
        index.sort(key=lambda x: x["relative_time_ms"])
        self._times = np.array(
            [x["relative_time_ms"] for x in index], dtype=float
        )
        self._frames = index
        self._virtual_clock = float(self._times[0])
        self.captured_count = 0
        self.first_capture = None
        self.last_capture = None

        span_s = (self._times[-1] - self._times[0]) / 1000.0
        if span_s > self.ANOMALY_SPAN_SECONDS:
            display_message(
                self.agent.aid.name,
                f"[WARN] Animal {tag} tem span anômalo: {span_s:.1f}s "
                f"({len(index)} frames). Considere --max_passage_seconds.",
            )
        display_message(
            self.agent.aid.name,
            f"[START] Animal {tag} ({self.tag_idx + 1}/{len(self.animal_tags)}) "
            f"- {len(index)} frames, span {span_s:.2f}s",
        )

    @staticmethod
    def _nearest_index(times: np.ndarray, value: float) -> int:
        """Índice do element de `times` mais próximo de `value`."""
        j = int(np.searchsorted(times, value))
        if j <= 0:
            return 0
        if j >= len(times):
            return len(times) - 1
        # desempata entre j-1 e j pelo mais próximo
        if abs(times[j - 1] - value) <= abs(times[j] - value):
            return j - 1
        return j

    def _passage_end_ms(self) -> float:
        """Limite superior do relógio virtual para o animal corrente."""
        tmax = float(self._times[-1])
        if self.max_passage_seconds is not None:
            cap = float(self._times[0]) + self.max_passage_seconds * 1000.0
            return min(tmax, cap)
        return tmax

    # ------------------------------------------------------------------ #
    def on_time(self):
        super().on_time()

        if self._finished:
            return
        if not self.agent.simulation_started:
            return

        # Primeiro pulso: pré-carrega o primeiro animal
        if self._times is None:
            self._load_animal(self.animal_tags[self.tag_idx])

        tag = self.animal_tags[self.tag_idx]
        end_ms = self._passage_end_ms()

        # Relógio virtual além do fim da passagem -> fecha o animal
        if self._virtual_clock > end_ms:
            self._close_animal(tag)
            return

        # --- Captura o frame depth mais próximo do relógio virtual ---
        j = self._nearest_index(self._times, self._virtual_clock)
        frame = self._frames[j]

        img = self.dataset.load_depth(tag, frame["depth_filename"])
        if img is None:
            display_message(
                self.agent.aid.name,
                f"[ERROR] load_depth retornou None para {tag}/{frame['depth_filename']}",
            )
            self._virtual_clock += self.step_ms
            return

        frame_id = str(uuid.uuid4())[:12]
        now_iso = datetime.now().isoformat()
        if self.captured_count == 0:
            self.first_capture = now_iso
        self.last_capture = now_iso
        self.captured_count += 1

        with self._lock:
            FRAME_BUFFER[frame_id] = img

        if self.verbose:
            display_message(
                self.agent.aid.name,
                f"[CAPTURE] animal={tag} idx={self.captured_count} "
                f"t={self._virtual_clock:.1f}ms label={frame.get('label')} "
                f"-> {self.next_agent_aid}",
            )

        msg = ACLMessage(ACLMessage.INFORM)
        msg.set_ontology("frame-capture")
        msg.add_receiver(AID(self.next_agent_aid))
        msg.set_content(json.dumps({
            "frame_id": frame_id,
            "animal_id": tag,
            "frame_index": self.captured_count,
            "elapsed_time": round(self._virtual_clock, 2),
            "label": frame.get("label"),
            "depth_filename": frame.get("depth_filename"),
        }, ensure_ascii=True))
        self.agent.send(msg)

        self._virtual_clock += self.step_ms

    def _close_animal(self, tag: str):
        """Emite passage-complete e avança para o próximo animal (ou termina)."""
        msg = ACLMessage(ACLMessage.INFORM)
        msg.set_ontology("passage-complete")
        msg.add_receiver(AID(self.selection_agent_aid))
        msg.set_content(json.dumps({
            "animal_id": tag,
            "total_frames": self.captured_count,
            "first_capture": self.first_capture,
            "last_capture": self.last_capture,
        }, ensure_ascii=True))
        self.agent.send(msg)

        display_message(
            self.agent.aid.name,
            f"[PASSAGE-COMPLETE] Animal {tag}: {self.captured_count} frames capturados.",
        )

        # Aviso IN-PROCESS ao PredictWeightAgent: a captura deste animal acabou
        # (último frame capturado). Mesmo processo/mesma thread do reator => 100%
        # confiável, não depende do canal FIPA (fire-and-forget). É o gatilho
        # deterministicamente confiável para o cálculo do peso final do animal.
        if self.predict_agent is not None:
            self.predict_agent.notify_capture_done(
                tag,
                total_frames=self.captured_count,
                first_capture=self.first_capture,
                last_capture=self.last_capture,
            )

        self.tag_idx += 1
        if self.tag_idx >= len(self.animal_tags):
            display_message(
                self.agent.aid.name,
                f"[FINISH] Captura concluída para {len(self.animal_tags)} animais.",
            )
            self._finished = True
            self._times = None
            return

        # Pré-carrega o próximo animal imediatamente
        self._load_animal(self.animal_tags[self.tag_idx])


class DatasetCaptureAgent(Agent):
    """Capture PADE agent — captura data-driven por tag, ritmada por FPS."""

    def __init__(
        self,
        aid,
        dataset: AnimalDataset,
        next_agent_aid: str,
        selection_agent_aid: str,
        animal_tags: list[str],
        fps: float,
        max_passage_seconds: float | None = None,
        wait_for_aids: list[str] = None,
        predict_agent=None,
        debug: bool = False,
        verbose: bool = False,
    ):
        super().__init__(aid=aid, debug=debug)
        self.dataset = dataset
        self.next_agent_aid = next_agent_aid
        self.selection_agent_aid = selection_agent_aid
        self.animal_tags = animal_tags
        self.fps = fps
        self.max_passage_seconds = max_passage_seconds
        self.predict_agent = predict_agent
        self.verbose = verbose
        self.wait_for_aids = set(wait_for_aids) if wait_for_aids else set()
        self.ready_agents = set()
        self.simulation_started = False

    def on_start(self):
        super().on_start()

        self.capture_behaviour = DatasetCaptureBehaviour(
            agent=self,
            dataset=self.dataset,
            next_agent_aid=self.next_agent_aid,
            selection_agent_aid=self.selection_agent_aid,
            animal_tags=self.animal_tags,
            fps=self.fps,
            max_passage_seconds=self.max_passage_seconds,
            predict_agent=self.predict_agent,
            verbose=self.verbose,
        )
        self.behaviours.append(self.capture_behaviour)

        if not self.wait_for_aids:
            self.simulation_started = True
            display_message(
                self.aid.name,
                f"DatasetCaptureAgent iniciando — {len(self.animal_tags)} animais a {self.fps} fps",
            )
        else:
            display_message(
                self.aid.name,
                f"DatasetCaptureAgent aguardando agentes: {self.wait_for_aids}",
            )

    def _start_simulation(self):
        if self.simulation_started:
            return
        self.simulation_started = True
        display_message(
            self.aid.name,
            "DatasetCaptureAgent IGNITED. Modelos carregados! Iniciando captura.",
        )

    def react(self, message):
        super().react(message)
        if message.ontology == "agent-ready":
            try:
                data = json.loads(message.content)
                agent_name = data.get("agent")
                self.ready_agents.add(agent_name)
                display_message(self.aid.name, f"Agent {agent_name} is READY.")

                if self.wait_for_aids.issubset(self.ready_agents):
                    display_message(
                        self.aid.name,
                        "Todos os agentes requeridos estão prontos! Igniting...",
                    )
                    self._start_simulation()
            except Exception as e:
                display_message(self.aid.name, f"[ERROR] Processing agent-ready: {e}")
