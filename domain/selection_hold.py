"""Maquina de estados de Selection Hold por passagem."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SelectionHold:
    """Mantem a retencao de taxa elevada (HIGH) por N rejeicoes consecutivas do Selection.

    Invariantes:
    1. accepted=True ativa o hold (hold_active = True) e zera o contador de rejeicoes.
    2. N rejeicoes consecutivas (default N=2) sem nenhum accepted desativam o hold.
    3. O Selection Hold isoladamente NUNCA provoca upshift LOW -> HIGH.
    4. Reset explicito ao final ou troca de passagem.
    """

    n_rejections_threshold: int = 2
    hold_active: bool = False
    consecutive_rejections: int = 0

    def __post_init__(self) -> None:
        if self.n_rejections_threshold < 0:
            raise ValueError("n_rejections_threshold must be non-negative")

    def observe(self, accepted: bool) -> bool:
        """Processa uma decisao real do Selection e retorna se o hold esta ativo."""
        if self.n_rejections_threshold == 0:
            self.hold_active = False
            self.consecutive_rejections = 0
            return False

        if accepted:
            self.hold_active = True
            self.consecutive_rejections = 0
        else:
            if self.hold_active:
                self.consecutive_rejections += 1
                if self.consecutive_rejections >= self.n_rejections_threshold:
                    self.hold_active = False
        return self.hold_active

    def reset(self) -> None:
        """Limpa o estado do hold."""
        self.hold_active = False
        self.consecutive_rejections = 0
