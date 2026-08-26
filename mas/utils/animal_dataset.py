"""AnimalDataset — carregador dos datasets reais por tag em data/exp1.

Cada animal tem:
  - data/exp1/animal-tags/<tag>/simulation_index.json
      lista de {relative_time_ms, depth_filename, rgb_filename, label}
      label in {background, parcial, suited, ruido}  (ground-truth)
  - data/exp1/DEPTH/<tag>/<depth_filename>
      PNG depth uint16 mm, shape (240, 320). NÃO reescalar: o seletor v3
      aplica ROI10 e clipa em 1950; o DataEnhance também clipa em 1950,
      com transformação espacial distinta (ambos esperam mm crus).
"""

import json
from pathlib import Path

import numpy as np
import skimage.io as ski


class AnimalDataset:
    """Acesso somente-leitura aos frames depth reais indexados por tag."""

    def __init__(self, data_root: str = "data/exp1"):
        self.tags_root = Path(data_root) / "animal-tags"
        self.depth_root = Path(data_root) / "DEPTH"

    def list_tags(self, limit: int | None = None) -> list[str]:
        """Tags (ordem alfabética) que possuem simulation_index.json."""
        tags = sorted(
            p.name
            for p in self.tags_root.iterdir()
            if p.is_dir() and (p / "simulation_index.json").exists()
        )
        return tags[:limit] if limit else tags

    def load_index(self, tag: str) -> list[dict]:
        """Lista de frames do animal: relative_time_ms, depth_filename, label."""
        with open(self.tags_root / tag / "simulation_index.json") as f:
            return json.load(f)

    def load_depth(self, tag: str, depth_filename: str) -> np.ndarray:
        """Carrega o PNG depth cru (uint16 mm preservado)."""
        return ski.imread(self.depth_root / tag / depth_filename)
