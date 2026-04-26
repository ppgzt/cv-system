# MAS Integration Architecture Design

## Contexto

Este documento descreve a arquitetura de integração do sistema Multi-Agent System (MAS) no projeto `cv-system-multiagent`, estruturado em DDD (Domain-Driven Design).

O objetivo é permitir a comparação científica de performance entre duas estratégias de pipeline:

- **`single`**: Pipeline linear/sequencial (abordagem tradicional)
- **`agentic`**: Sistema multi-agente baseado em PADE (SMA - Multi-Agent System)

---

## Principio Fundamental: Comparação Válida

Para que a comparação de métricas (CPU, RAM, Vazão) seja cientificamente válida, os componentes individuais de processamento devem ser idênticos em ambas as estratégias.

**O que muda**: Apenas a orquestração (sequencial vs. coordenação por agentes).
**O que permanece igual**: Captura, inferência, seleção de frames, etc.

---

## Estrutura de Pastas

```
cv-system-multiagent/
├── domain/                          # Lógica REAL de Visão Computacional (SHARED)
│   └── modules/
│       ├── image_capture.py         # ImageCapture.get_frame()
│       ├── frame_selection.py       # FrameSelection.evaluate()
│       ├── inference.py             # Modelos de inferência (YOLO, etc.)
│       └── event_detection.py       # Detecção de eventos
│
├── mas/                             # Sistema Multi-Agente (PADE)
│   ├── docs/
│   │   └── architecture-design.md   # Este documento
│   ├── adapters/                    # Wrappers finos (delegam para domain/modules)
│   │   ├── capture_adapter.py       # -> chama ImageCapture.get_frame()
│   │   ├── inference_adapter.py     # -> chama modelo de domain/
│   │   ├── frame_selection_adapter.py
│   │   └── profiling_adapter.py     # -> infra.profiling.agents
│   └── agents/                      # Agentes PADE (apenas lógica de coordenação)
│       ├── capture_agent.py         # Agent PADE de captura
│       ├── inference_agent.py       # Agent PADE de inferência
│       ├── event_detection_agent.py # Agent PADE de detecção de eventos
│       ├── frame_selection_agent.py # Agent PADE de seleção de frames
│       ├── resource_manager_agent.py# Monitoramento CPU/RAM
│       └── storage_agent.py         # Agent PADE de armazenamento
│
├── infra/                           # Infraestrutura compartilhada
│   ├── images/                      # Imagens de teste
│   ├── models/                      # Modelos pré-treinados
│   └── profiling/                   # Adaptadores de monitoramento
│       └── agents.py                # Métricas CPU/RAM
│
└── main.py                          # Orquestrador de experimentos
```

---

## Padrão Adapter: Wrappers Finos

### Conceito

Os adapters na pasta `mas/adapters/` são wrappers finos que:

1. Encapsulam chamadas para `domain/modules/`
2. Adaptam a interface para o formato esperado pelos agentes PADE
3. Não duplicam lógica de processamento

### Exemplo: CaptureAdapter

```python
# mas/adapters/capture_adapter.py
from domain.modules.image_capture import ImageCapture

class CaptureAdapter:
    """Adapter que encapsula ImageCapture para uso por agentes PADE."""
    
    def __init__(self, config: dict = None):
        # Usa a MESMA classe do pipeline linear
        self.impl = ImageCapture()
        
    def get_frame(self):
        """Delega para a implementação real em domain/modules/."""
        return self.impl.get_frame()
```

### Exemplo: FrameSelectionAdapter

```python
# mas/adapters/frame_selection_adapter.py
from domain.modules.frame_selection import FrameSelection

class FrameSelectionAdapter:
    """Adapter para seleção de frames via agentes."""
    
    def __init__(self, suitable_window: float, snooze_duration: float):
        # Usa a MESMA classe do pipeline linear
        self.impl = FrameSelection(suitable_window, snooze_duration)
        
    def evaluate(self, elapsed_time: float) -> bool:
        """Delega para a implementação real em domain/modules/."""
        return self.impl.evaluate(elapsed_time)
```

---

## Padrão Agent: Apenas Coordenação

Os agentes na pasta `mas/agents/` contêm apenas lógica de coordenação PADE:

- Envio/recebimento de mensagens ACL
- Behaviours (comportamentos)
- Sincronização com outros agentes
- Não contêm lógica de processamento de imagem

### Exemplo: CaptureAgent

```python
# mas/agents/capture_agent.py
from pade.core.agent import Agent
from pade.misc.utility import display_message
from mas.adapters.capture_adapter import CaptureAdapter

class CaptureAgent(Agent):
    """Agente PADE responsável pela captura de frames."""
    
    def __init__(self, config: dict = None):
        super().__init__()
        # Injeta o adapter (que delega para domain/modules)
        self.adapter = CaptureAdapter(config)
        
    def on_start(self):
        super().on_start()
        display_message(self.aid, 'CaptureAgent iniciado.')
        
    def capture_and_send(self):
        # Usa o adapter para capturar (MESMA função do pipeline linear)
        frame = self.adapter.get_frame()
        
        # Envia frame via mensagem ACL para o próximo agente
        # ... lógica de coordenação PADE ...
```

### Exemplo: Pipeline Linear (SingleStrategy)

```python
# domain/pipelines.py - SinglePipelineStrategy
from domain.modules.image_capture import ImageCapture

class SinglePipelineStrategy:
    """Pipeline linear/sequencial - usa domain/modules diretamente."""
    
    def run(self, video_path: str):
        # Usa a MESMA classe ImageCapture diretamente
        capture = ImageCapture()
        frame = capture.get_frame()  # MESMA função do AgenticStrategy
        
        # Processamento sequencial...
        return results
```

---

## Fluxo de Execução Comparado

### Estratégia single (Pipeline Linear)

```
main.py
  └─ SinglePipelineStrategy.run()
       ├─ ImageCapture.get_frame()          <- domain/modules/image_capture.py
       ├─ FrameSelection.evaluate()         <- domain/modules/frame_selection.py
       ├─ YOLOInference.predict()           <- domain/modules/inference.py
       └─ EventDetection.detect()           <- domain/modules/event_detection.py
```

### Estratégia agentic (Multi-Agent System)

```
main.py
  └─ AgenticStrategy.initialize()
       ├─ PADE AMS Start
       │
       ├─ CaptureAgent
       │    └─ CaptureAdapter.get_frame()   -> domain/modules/image_capture.py
       │
       ├─ FrameSelectionAgent
       │    └─ FrameSelectionAdapter.eval() -> domain/modules/frame_selection.py
       │
       ├─ InferenceAgent
       │    └─ InferenceAdapter.predict()   -> domain/modules/inference.py
       │
       ├─ EventDetectionAgent
       │    └─ EventDetectionAdapter.detect() -> domain/modules/event_detection.py
       │
       └─ ResourceManagerAgent
            └─ ProfilingAdapter.read()      -> infra/profiling/agents.py
```

Em ambos os casos, as funções de `domain/modules/` são as mesmas.

---

## Monitoramento e Métricas

### ResourceManagerAgent

O `ResourceManagerAgent` é responsável por:

1. Coletar métricas de CPU e RAM via `ProfilingAdapter`
2. Armazenar métricas em estrutura thread-safe
3. Expor métricas para coleta externa pelo `main.py`

```python
# mas/agents/resource_manager_agent.py
from threading import Lock
from mas.adapters.profiling_adapter import ProfilingAdapter

class ResourceManagerAgent(Agent):
    """Agente de monitoramento de recursos do sistema."""
    
    def __init__(self, profiling_adapter: ProfilingAdapter):
        super().__init__()
        self.profiling = profiling_adapter
        self._metrics = {}
        self._lock = Lock()
        
    def update_metrics(self):
        """Atualiza métricas de CPU/RAM."""
        cpu, ram = self.profiling.read()
        with self._lock:
            self._metrics['cpu'] = cpu
            self._metrics['ram'] = ram
            
    def get_metrics(self) -> dict:
        """Retorna snapshot thread-safe das métricas."""
        with self._lock:
            return dict(self._metrics)
```

### Coleta no main.py

```python
# main.py
strategy = AgenticStrategy(profiling_adapters)
strategy.initialize()

# Durante o experimento
while experiment_running:
    metrics = resource_manager_agent.get_metrics()
    reporter.log_metrics(metrics)
    time.sleep(1)
```

---

## AgenticStrategy: Integração com main.py

### Estrutura da Classe

```python
# domain/pipelines.py (ou strategies.py)

import threading
from pade.misc.utility import start_loop

class AgenticStrategy:
    """Estratégia agêntica para orquestração via PADE."""
    
    def __init__(self, profiling_adapters: dict, config: dict = None):
        self.profiling = profiling_adapters
        self.pade_loop = None
        self.agent_thread = None
        self.resource_manager = None
        
    def initialize(self):
        """Inicializa o servidor PADE e os agentes."""
        # 1. Criar adapters de profiling injetados
        profiling_adapter = ProfilingAdapter(self.profiling)
        
        # 2. Criar agentes com dependências injetadas
        self.resource_manager = ResourceManagerAgent(
            profiling_adapter=profiling_adapter
        )
        capture_agent = CaptureAgent(config=self.config)
        inference_agent = InferenceAgent(config=self.config)
        # ... outros agentes
        
        # 3. Lista de agentes
        agents = [
            self.resource_manager,
            capture_agent,
            inference_agent,
            # ...
        ]
        
        # 4. Iniciar loop PADE em thread separada (não-bloqueante)
        self.agent_thread = threading.Thread(
            target=self._start_pade, args=(agents,), daemon=True
        )
        self.agent_thread.start()
        
    def _start_pade(self, agents):
        """Inicia o loop de agentes PADE."""
        self.pade_loop = start_loop(agents)
        
    def run_experiment(self, video_path: str):
        """Dispara o experimento via mensagens ACL."""
        # Envia mensagem ACL para CaptureAgent iniciar
        # Agentes processam autonomamente
        pass
        
    def shutdown(self):
        """Encerramento gracioso dos agentes e servidor PADE."""
        if self.pade_loop:
            self.pade_loop.stop()
        if self.agent_thread:
            self.agent_thread.join(timeout=5)
```

### Uso com Context Manager

```python
# main.py
with AgenticStrategy(profiling) as strategy:
    strategy.run_experiment(video_path)
    metrics = collect_metrics()
    save_comparison('agentic', metrics)
# Shutdown automático ao sair do context
```

---

## Encerramento Gracioso

### Abordagens Combinadas

1. Signal Handler (main.py):

```python
import signal

def shutdown_handler(signum, frame):
    strategy.shutdown()
    sys.exit(0)

signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)
```

2. Context Manager (AgenticStrategy):

```python
class AgenticStrategy:
    def __enter__(self):
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
```

3. ShutdownAgent (opcional):

- Agente que escuta mensagem ACL `"shutdown"` e chama `exit_ams()`

---

## Configuração de Imports

Para evitar `ModuleNotFoundError`, o arquivo `mas/__init__.py` deve configurar o `sys.path`:

```python
# mas/__init__.py
import sys
from pathlib import Path

# Adiciona a raiz do projeto ao path para imports relativos
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
```

Isso permite que os agentes importem:

```python
from domain.modules.image_capture import ImageCapture  # funciona
from infra.profiling.agents import ProfilingAdapter    # funciona
```

---

## Checklist de Implementação

- [ ] Copiar `pade/`, `agents/`, `adapters/` do `mas-edge-vision` para `cv-system-multiagent/mas/`
- [ ] Refatorar adapters para delegar para `domain/modules/`
- [ ] Remover lógica de processamento dos agentes (manter apenas coordenação)
- [ ] Configurar `sys.path` no `mas/__init__.py`
- [ ] Implementar `AgenticStrategy` com injeção de dependência
- [ ] Implementar `ResourceManagerAgent` com métricas thread-safe
- [ ] Implementar signal handlers e context manager para shutdown
- [ ] Adaptar `main.py` para rodar ambas as estratégias
- [ ] Validar que as funções de `domain/modules/` são idênticas em ambos os pipelines
- [ ] Rodar experimentos e comparar métricas

---

## Referências

- **mas-edge-vision**: Repositório de origem com implementação PADE completa
- **PADE Framework**: Framework Python para sistemas multi-agent
- **PIBIC**: Projeto de Iniciação Científica - Comparação de performance entre pipeline linear e SMA

---

Documento criado em: 2026-04-14
