"""MAS package — configuração do sys.path para imports do pade e infra."""

import sys
from pathlib import Path

# Adicionar o root do projeto ao PYTHONPATH para que
# `from infra.profiling.agents import ...` funcione
_project_root = Path(__file__).resolve().parent.parent
_project_root_str = str(_project_root)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

# Adicionar mas/ ao PYTHONPATH para que
# `from pade.core.agent import Agent` funcione (pade está em mas/pade/)
_mas_dir = str(Path(__file__).resolve().parent)
if _mas_dir not in sys.path:
    sys.path.insert(0, _mas_dir)
