"""Pipeline MAS data-driven por tag (paralelo a domain/pipelines.py:MASStrategy).

Diferente do MASStrategy antigo (simulador biológico sintético com
sample.png + passage/arrival_time fixos), esta versão usa captura
data-driven: o DatasetCaptureAgent lê os frames depth reais de
data/exp1/DEPTH/<tag>/, ritmados por FPS (nearest-neighbor no
relative_time_ms), com o label ground-truth viajando nos metadados.

Cadeia de agentes (inalterada):
    DatasetCaptureAgent -> FrameSelectionAgent -> DataEnhanceAgent -> PredictWeightAgent

Apenas o agente de captura e o critério de término mudam em relação ao
MASStrategy antigo. main.py, domain/pipelines.py e as estratégias
SingleStream/Batch seguem intocados.
"""

import os
import time


class MASStrategy:
    """Strategy Multi-Agent System data-driven por tag.

    O tamanho do rebanho é derivado da quantidade de tags em data/exp1
    (opcionalmente limitado por `num_animals`). A duração da passagem de
    cada animal vem do próprio dado (tmax do simulation_index.json).
    """

    def __init__(
        self,
        pid: str,
        mode: str,
        fps: float | None = None,
        low_fps: float | None = None,
        medium_fps: float | None = None,
        visual_gated: bool = False,
        selection_hold_n: int = 2,
        num_animals: int | None = None,
        max_passage_seconds: float | None = None,
        data_root: str = "data/exp1",
        native_timestamps: bool = False,
        capture_timing_enabled: bool = True,
        visual_event_enabled: bool = False,
        visual_pdi_threshold: float | None = None,
        visual_pixel_threshold_mm: float = 200.0,
        visual_idle_patience: int | None = None,
        verbose: bool = False,
    ):
        if visual_event_enabled:
            if visual_pdi_threshold is None or visual_pdi_threshold < 0:
                raise ValueError(
                    "visual_pdi_threshold must be provided and non-negative"
                )
            if visual_pixel_threshold_mm <= 0:
                raise ValueError(
                    "visual_pixel_threshold_mm must be positive"
                )
            if visual_idle_patience is None or visual_idle_patience <= 0:
                raise ValueError(
                    "visual_idle_patience must be provided and positive"
                )
        self.pid = pid
        self.mode = mode
        self.fps = fps
        self.low_fps = low_fps
        self.medium_fps = medium_fps
        self.visual_gated = visual_gated
        self.selection_hold_n = selection_hold_n
        self.num_animals = num_animals
        self.max_passage_seconds = max_passage_seconds
        self.data_root = data_root
        self.native_timestamps = native_timestamps
        self.capture_timing_enabled = capture_timing_enabled
        self.visual_event_enabled = visual_event_enabled
        self.visual_pdi_threshold = visual_pdi_threshold
        self.visual_pixel_threshold_mm = visual_pixel_threshold_mm
        self.visual_idle_patience = visual_idle_patience
        self.verbose = verbose

    def run(self):
        """Inicia os agentes PADE e o loop principal do reator."""
        run_monotonic_origin_ns = time.monotonic_ns()
        from dotenv import load_dotenv
        load_dotenv(override=True)

        import mas  # noqa: F401  (sys.path hack para pade/infra)
        from pade.acl.aid import AID
        from pade.misc.utility import display_message
        from pade.core.new_ams import AMS
        from twisted.internet import reactor

        from mas.utils.animal_dataset import AnimalDataset
        from mas.infrastructure.frame_store import FRAME_STORE
        from mas.agents.resource_manager_agent import ResourceManagerAgent
        from mas.agents.dataset_capture_agent import DatasetCaptureAgent
        from mas.agents.data_enhance_agent import DataEnhanceAgent
        from mas.agents.frame_selection import FrameSelectionAgent
        from mas.agents.predict_weight_agent import PredictWeightAgent
        from mas.agents.visual_event_agent import VisualEventAgent

        from mas.adapters.data_enhance_adapter import DataEnhanceAdapter
        from mas.adapters.frame_selection_adapter import FrameSelectionAdapter
        from mas.adapters.inference_adapter import InferenceAdapter
        from mas.infrastructure.pade_telemetry import PadeTelemetrySession

        # 1. Dataset + ordem dos animais (alfabética por tag)
        dataset = AnimalDataset(self.data_root)
        animal_tags = dataset.list_tags(limit=self.num_animals)
        if not animal_tags:
            display_message("MASStrategy", f"[ERROR] nenhuma tag encontrada em {self.data_root}")
            return

        from mas.utils.report_collector import ReportCollector
        ReportCollector().reset()
        FRAME_STORE.clear()

        # 2. Configuração via .env
        ams_host = os.getenv("SMA_AMS_HOST", "localhost")
        ams_port = int(os.getenv("SMA_AMS_PORT", 8000))
        agent_host = os.getenv("SMA_AGENT_HOST", "localhost")
        base_port = int(os.getenv("SMA_AGENT_BASE_PORT", 5003))

        display_message("MASStrategy", f"Iniciando MAS data-driven para PID: {self.pid}")
        display_message(
            "MASStrategy",
            f"Configuração: AMS={ams_host}:{ams_port}, BasePort={base_port}, "
            f"animais={len(animal_tags)}, "
            f"timing={'original' if self.native_timestamps else f'{self.fps} fps'}, "
            f"mode={self.mode}",
        )

        # 3. AMS Agent (standalone)
        ams_agent = AMS(host=ams_host, port=ams_port)
        ams_agent.register_user("admin", "admin@pade.com", "admin")
        ams_agent._initialize_database()

        # 4. Port layout
        # base_port+0: DatasetCaptureAgent
        # base_port+1: DataEnhanceAgent
        # base_port+2: FrameSelectionAgent
        # base_port+3: PredictWeightAgent
        # base_port+4: VisualEventAgent (opcional)
        # base_port+5: OrchestratorAgent (opcional/adaptativo)
        # base_port+6: ResourceManagerAgent

        def aid(name, offset):
            port = base_port + offset
            return AID(name=f"{name}@{agent_host}:{port}")

        capture_aid = aid("capture_agent", 0)
        enhance_aid = aid("data_enhance_agent", 1)
        selection_aid = aid("frame_selection_agent", 2)
        predict_aid = aid("predict_weight_agent", 3)
        visual_aid = aid("visual_event_agent", 4)
        orch_aid = aid("orchestrator_agent", 5)
        rm_aid = aid("resource_manager_agent", 6)

        weight_model_path = "infra/models/sheep_weight_predictor.tflite"
        selection_model_path = "infra/models/frame_selector.tflite"

        # 5. Adapters (lógica de domínio compartilhada, paridade com baseline)
        enhance_adapter = DataEnhanceAdapter()
        # suitable_window é vestigial (não usado em FrameSelection.evaluate).
        selection_adapter = FrameSelectionAdapter(
            suitable_window=None,
            model_path=selection_model_path,
        )
        inference_adapter = InferenceAdapter(weight_model_path)

        if self.low_fps is not None:
            condition = (
                "pade_visual_gated"
                if self.visual_gated
                else "pade_visual_adaptive"
            )
            telemetry_condition = condition
            telemetry_capture_fps = None
            capture_mode_str = "visual-gated" if self.visual_gated else "visual-adaptive"
            predict_agent_fps = None
        else:
            condition = (
                "pade_original_timing"
                if self.native_timestamps
                else "pade_fixed_fps"
            )
            telemetry_condition = condition
            telemetry_capture_fps = None if self.native_timestamps else self.fps
            capture_mode_str = "native-timestamps" if self.native_timestamps else None
            predict_agent_fps = self.fps

        # 6. Agents
        predict_agent = PredictWeightAgent(
            aid=predict_aid,
            inference_adapter=inference_adapter,
            mode=self.mode,
            pid=self.pid,
            capture_agent_aid=capture_aid.name,
            frame_store=FRAME_STORE,
            verbose=self.verbose,
            fps=predict_agent_fps,
            capture_mode=capture_mode_str,
        )


        orchestrator_agent = None
        if self.visual_event_enabled or self.low_fps is not None:
            from mas.agents.orchestrator_agent import OrchestratorAgent
            orchestrator_agent = OrchestratorAgent(
                aid=orch_aid,
                capture_agent_aid=capture_aid.name,
                n_hold=self.selection_hold_n,
                verbose=self.verbose,
            )

        wait_for_aids = [selection_aid.name, predict_aid.name]
        if self.visual_event_enabled:
            wait_for_aids.append(visual_aid.name)
        if orchestrator_agent is not None:
            wait_for_aids.append(orch_aid.name)

        capture_agent = DatasetCaptureAgent(
            aid=capture_aid,
            dataset=dataset,
            next_agent_aid=selection_aid.name,
            selection_agent_aid=selection_aid.name,
            animal_tags=animal_tags,
            fps=self.fps,
            low_fps=self.low_fps,
            medium_fps=self.medium_fps,
            visual_gated=self.visual_gated,
            orchestrator_agent_aid=(
                orch_aid.name if orchestrator_agent is not None else None
            ),
            max_passage_seconds=self.max_passage_seconds,
            native_timestamps=self.native_timestamps,
            wait_for_aids=wait_for_aids,
            frame_store=FRAME_STORE,
            visual_agent_aid=(
                visual_aid.name if self.visual_event_enabled else None
            ),
            verbose=self.verbose,
        )

        enhance_agent = DataEnhanceAgent(
            aid=enhance_aid,
            data_enhance_adapter=enhance_adapter,
            next_agent_aid=predict_aid.name,
            frame_store=FRAME_STORE,
        )

        selection_agent = FrameSelectionAgent(
            aid=selection_aid,
            frame_selection_adapter=selection_adapter,
            next_agent_aid=enhance_aid.name,
            capture_agent_aid=capture_aid.name,
            orchestrator_agent_aid=(
                orch_aid.name if orchestrator_agent is not None else None
            ),
            frame_store=FRAME_STORE,
            verbose=self.verbose,
        )

        resource_agent = ResourceManagerAgent(
            aid=rm_aid,
            pid=self.pid,
            reports_dir="infra/reports",
            debug=False,
        )
        resource_agent.ams = {"name": ams_host, "port": ams_port}

        visual_agent = None
        if self.visual_event_enabled:
            visual_agent = VisualEventAgent(
                aid=visual_aid,
                capture_agent_aid=capture_aid.name,
                orchestrator_agent_aid=(
                    orch_aid.name if orchestrator_agent is not None else None
                ),
                pdi_threshold=self.visual_pdi_threshold,
                pixel_threshold_mm=self.visual_pixel_threshold_mm,
                idle_patience_frames=self.visual_idle_patience,
                pid=self.pid,
                frame_store=FRAME_STORE,
            )

        telemetry = PadeTelemetrySession(
            run_id=self.pid,
            condition=telemetry_condition,
            capture_fps=telemetry_capture_fps,
            monotonic_origin_ns=run_monotonic_origin_ns,
            selection_inbox=selection_agent.inbox,
            enhance_inbox=enhance_agent.inbox,
            prediction_inbox=predict_agent.inbox,
            capture_timing_enabled=self.capture_timing_enabled,
        )
        capture_agent.telemetry_context = telemetry.context
        capture_agent.capture_timing_recorder = telemetry.capture_timing_recorder
        selection_agent.telemetry_context = telemetry.context
        selection_agent.capture_timing_recorder = telemetry.capture_timing_recorder

        # 7. Hooks de shutdown: EndPipeline continua sendo o gatilho lógico;
        # estes callbacks apenas persistem observabilidade no shutdown global.
        reactor.addSystemEventTrigger('before', 'shutdown', telemetry.stop)
        reactor.addSystemEventTrigger(
            'before', 'shutdown', resource_agent.stop_monitoring
        )
        if visual_agent is not None:
            reactor.addSystemEventTrigger(
                'before', 'shutdown', visual_agent.stop_visual_monitoring
            )

        # 8. Conecta todos os agentes ao AMS e ao reator
        all_agents = [
            resource_agent, capture_agent, enhance_agent,
            selection_agent, predict_agent,
        ]
        if visual_agent is not None:
            all_agents.append(visual_agent)
        if orchestrator_agent is not None:
            all_agents.append(orchestrator_agent)


        # Ambos iniciam antes de qualquer on_start que possa admitir o primeiro
        # frame. Permanecem ativos até o shutdown posterior ao drain global.
        telemetry.start()

        for agent in all_agents:
            agent.update_ams(resource_agent.ams)
            agent.on_start()
            reactor.listenTCP(agent.aid.port, agent.agentInstance)

        display_message("MASStrategy", "Todos os agentes iniciados. Reator Twisted rodando.")
        # Thread pool enxuto (Pi 5: 4 cores) — evita context switching excessivo
        # com deferToThread concorrentes.
        reactor.getThreadPool().adjustPoolsize(minthreads=2, maxthreads=4)
        reactor.run()
