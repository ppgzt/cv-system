import os, keras, time, json
import numpy as np

from datetime import datetime

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"

def dummy_npwarn_decorator_factory():
  def npwarn_decorator(x):
    return x
  return npwarn_decorator
np._no_nep50_warning = getattr(np, '_no_nep50_warning', dummy_npwarn_decorator_factory)

from domain.modules.frame_selection import FrameSelection
from domain.modules.image_capture   import ImageCapture
from domain.modules.predict_weight  import PredictWeight
from domain.modules.data_enhance    import DataEnhance

class SingleStreamStrategy:

    '''
    Docstring for SingleStreamStrategy
    '''
    def __init__(self, pid: str,
        herd_size: int, arrival_time: int, passage_time: int, fselection_time: float, fselection_window:float):

        self.pid = pid
        self.metrics = {
            'pid':pid,
            'load_model_start':datetime.now().isoformat(),
        }

        self.herd_size = herd_size
        self.arrival_time = arrival_time
        self.passage_time = passage_time

        self.model = keras.models.load_model(f'infra/models/model_run1_epoch029.keras')
        self.metrics['load_model_final'] = datetime.now().isoformat()

        self.frame_selection = FrameSelection(
            suitable_window=fselection_window, 
            model=self.model
        )

        self.image_capture = ImageCapture()
        self.data_enhance = DataEnhance()
        self.predict_weight = PredictWeight(model=self.model)

    def run(self):
        self.metrics['animals'] = {}

        for animal in range(1, self.herd_size + 1):
            print(f'animal: {animal}')
            start_at = datetime.now()

            self.metrics['animals'][animal] = {
                'first_image_capture_time':datetime.now().isoformat(),
                'imgs':{}
            }

            weights = []
            i = 0

            elapsed_time = (datetime.now() - start_at).total_seconds()
            last_image_capture = None
            while elapsed_time < self.passage_time:
                i += 1
                print(f'image: {i}')

                img = self.image_capture.get_frame()
                last_image_capture = datetime.now().isoformat()

                img = self.data_enhance.run(img)
                
                suitable = self.frame_selection.evaluate(
                    elapsed_time=elapsed_time,
                    img=img
                )
                if suitable:
                    inference_metrics = {
                        'weight_prediction_start':datetime.now().isoformat()
                    }

                    weight = self.predict_weight.predict(imgs=[img])[0][0]
                    weights.append(weight)

                    inference_metrics['weight_prediction_final'] = datetime.now().isoformat()
                    self.metrics['animals'][animal]['imgs'][i] = inference_metrics

                elapsed_time = (datetime.now() - start_at).total_seconds()

            self.metrics['animals'][animal]['last_image_capture_time'] = last_image_capture
            print(weights)

            predicted_weight = np.mean(weights)
            self.metrics['animals'][animal]['weight_prediction_final'] = datetime.now().isoformat()
            print(predicted_weight)

            # wait for the next animal
            time.sleep(self.arrival_time)

        with open(f"infra/reports/{self.pid}/metrics.json", "w") as json_file:
            json.dump(self.metrics, json_file, indent=4)

class BatchStreamStrategy:

    '''
    Docstring for BatchStreamStrategy
    '''
    def __init__(self, pid: str,
        herd_size: int, arrival_time: int, passage_time: int, fselection_time: float, fselection_window:float):

        self.pid = pid
        self.metrics = {
            'pid':pid,
            'load_model_start':datetime.now().isoformat(),
        }

        self.herd_size = herd_size
        self.arrival_time = arrival_time
        self.passage_time = passage_time

        self.model = keras.models.load_model(f'infra/models/model_run1_epoch029.keras')
        self.metrics['load_model_final'] = datetime.now().isoformat()

        self.frame_selection = FrameSelection(
            suitable_window=fselection_window, 
            model=self.model
        )

        self.image_capture = ImageCapture()
        self.data_enhance = DataEnhance()
        self.predict_weight = PredictWeight(model=self.model)

    def run(self):
        self.metrics['animals'] = {}

        for animal in range(1, self.herd_size + 1):
            print(f'animal: {animal}')
            start_at = datetime.now()

            self.metrics['animals'][animal] = {
                'first_image_capture_time':datetime.now().isoformat(),
                'imgs':{}
            }

            imgs = []
            i = 0
            s = 0
            
            elapsed_time = (datetime.now() - start_at).total_seconds()
            last_image_capture = None

            while elapsed_time < self.passage_time:
                i += 1
                print(f'image: {i}')

                img = self.image_capture.get_frame()
                last_image_capture = datetime.now().isoformat()

                img = self.data_enhance.run(img)

                suitable = self.frame_selection.evaluate(elapsed_time=elapsed_time, img=img)
                if suitable:
                    imgs.append(img)
                    s += 1

                elapsed_time = (datetime.now() - start_at).total_seconds()
                
            self.metrics['animals'][animal]['total_of_images'] = i
            self.metrics['animals'][animal]['suitable_images'] = s

            self.metrics['animals'][animal]['last_image_capture_time'] = last_image_capture
            inference_metrics = {
                'weight_prediction_start':datetime.now().isoformat()
            }

            weights = self.predict_weight.predict(imgs=imgs)
            print(weights)

            inference_metrics['weight_prediction_final'] = datetime.now().isoformat()
            self.metrics['animals'][animal]['imgs'][i] = inference_metrics

            predicted_weight = np.mean(weights)
            self.metrics['animals'][animal]['weight_prediction_final'] = datetime.now().isoformat()
            print(predicted_weight)

            # wait for the next animal
            time.sleep(self.arrival_time)

        with open(f"infra/reports/{self.pid}/metrics.json", "w") as json_file:
            json.dump(self.metrics, json_file, indent=4)

class MASStrategy:
    """Strategy for Multi-Agent System (MAS) execution.

    Wires the 4 PADE agents into a linear pipeline:

        CaptureAgent -> DataEnhanceAgent -> FrameSelectionAgent -> PredictWeightAgent

    All image data flows through FRAME_BUFFER (in-memory dict) keyed by
    frame_id.  FIPA-ACL messages carry only lightweight JSON metadata
    (frame_id, animal_id, elapsed_time).

    The reactor is stopped by CaptureAgent when all animals have been
    processed.
    """

    def __init__(self, pid: str, mode: str,
                 herd_size: int, arrival_time: int, passage_time: int,
                 fselection_time: float, fselection_window: float):
        self.pid = pid
        self.mode = mode
        self.herd_size = herd_size
        self.arrival_time = arrival_time
        self.passage_time = passage_time
        self.fselection_time = fselection_time
        self.fselection_window = fselection_window

    def run(self):
        """Starts the PADE agents and the main reactor loop."""
        from dotenv import load_dotenv
        load_dotenv(override=True)

        import mas  # IMPORTANTE: Adiciona as pastas ao sys.path
        from pade.acl.aid import AID
        from pade.misc.utility import display_message
        from pade.core.new_ams import AMS
        from twisted.internet import reactor
        from mas.agents.resource_manager_agent import ResourceManagerAgent

        from mas.agents.capture_agent import CaptureAgent
        from mas.agents.data_enhance_agent import DataEnhanceAgent
        from mas.agents.frame_selection import FrameSelectionAgent
        from mas.agents.predict_weight_agent import PredictWeightAgent

        from mas.adapters.capture_adapter import CaptureAdapter
        from mas.adapters.data_enhance_adapter import DataEnhanceAdapter
        from mas.adapters.frame_selection_adapter import FrameSelectionAdapter
        from mas.adapters.inference_adapter import InferenceAdapter

        # 1. Configuração via Variáveis de Ambiente (.env)
        ams_host = os.getenv("SMA_AMS_HOST", "localhost")
        ams_port = int(os.getenv("SMA_AMS_PORT", 8000))
        agent_host = os.getenv("SMA_AGENT_HOST", "localhost")
        base_port = int(os.getenv("SMA_AGENT_BASE_PORT", 5003))

        display_message("MASStrategy", f"Iniciando MAS Strategy para PID: {self.pid}")
        display_message("MASStrategy", f"Configuração: AMS={ams_host}:{ams_port}, BasePort={base_port}")

        # 2. Configuração do AMS Agent (standalone)
        ams_agent = AMS(host=ams_host, port=ams_port)
        ams_agent.register_user("admin", "admin@pade.com", "admin")
        ams_agent._initialize_database()

        # 3. Port layout
        # base_port+0: CaptureAgent
        # base_port+1: DataEnhanceAgent
        # base_port+2: FrameSelectionAgent
        # base_port+3: PredictWeightAgent
        # base_port+6: ResourceManagerAgent

        def aid(name, offset):
            port = base_port + offset
            return AID(name=f"{name}@{agent_host}:{port}")

        capture_aid       = aid("capture_agent", 0)
        enhance_aid       = aid("data_enhance_agent", 1)
        selection_aid     = aid("frame_selection_agent", 2)
        predict_aid       = aid("predict_weight_agent", 3)
        rm_aid            = aid("resource_manager_agent", 6)

        model_path = "infra/models/model_run1_epoch029.keras"

        # 4. Adapters (shared domain logic, parity with baseline)
        capture_adapter    = CaptureAdapter()
        enhance_adapter    = DataEnhanceAdapter()
        selection_adapter  = FrameSelectionAdapter(
            suitable_window=self.fselection_window,
            model_path=model_path,
        )

        # We pass the path to allow lazy/async loading in the agent
        inference_adapter = InferenceAdapter(model_path)

        # 5. Agents
        capture_agent = CaptureAgent(
            aid=capture_aid,
            capture_adapter=capture_adapter,
            next_agent_aid=enhance_aid.name,
            selection_agent_aid=selection_aid.name,
            interval_seconds=self.fselection_time,
            herd_size=self.herd_size,
            passage_time=self.passage_time,
            arrival_time=self.arrival_time,
            wait_for_aids=[selection_aid.name, predict_aid.name]
        )

        enhance_agent = DataEnhanceAgent(
            aid=enhance_aid,
            data_enhance_adapter=enhance_adapter,
            next_agent_aid=selection_aid.name,
        )

        selection_agent = FrameSelectionAgent(
            aid=selection_aid,
            frame_selection_adapter=selection_adapter,
            next_agent_aid=predict_aid.name,
            capture_agent_aid=capture_aid.name
        )

        predict_agent = PredictWeightAgent(
            aid=predict_aid,
            inference_adapter=inference_adapter,
            mode=self.mode,
            pid=self.pid,
            herd_size=self.herd_size,
            capture_agent_aid=capture_aid.name
        )

        resource_agent = ResourceManagerAgent(
            aid=rm_aid,
            pid=self.pid,
            reports_dir="infra/reports",
            debug=False
        )
        resource_agent.ams = {"name": ams_host, "port": ams_port}

        # 6. Setup Reactor shutdown hook for clean monitor stop
        reactor.addSystemEventTrigger(
            'before', 'shutdown', resource_agent.stop_monitoring
        )

        # 7. Wire all agents to AMS and reactor
        all_agents = [
            resource_agent, capture_agent, enhance_agent, 
            selection_agent, predict_agent
        ]

        for agent in all_agents:
            agent.update_ams(resource_agent.ams)
            agent.on_start()
            reactor.listenTCP(agent.aid.port, agent.agentInstance)

        display_message("MASStrategy", "Todos os agentes iniciados. Reator Twisted rodando.")
        reactor.run()
