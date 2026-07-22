"""Resultado padronizado dos estimadores de efeito causal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EffectResult:
    """Efeito estimado por um método causal.

    Args:
        method: Nome do método (ex: ``"ab"``, ``"did"``).
        metric: Nome da métrica avaliada.
        effect: Efeito absoluto estimado (variante − controle).
        relative: Efeito relativo (lift), quando aplicável.
        ci_low: Limite inferior do intervalo (confiança/credibilidade).
        ci_high: Limite superior do intervalo.
        p_value: P-valor (métodos frequentistas); ``None`` em bayesiano.
        significant: Veredito de significância no nível configurado.
        extra: Metadados adicionais (probabilidades, SE, grupos, n, etc.).

    Returns:
        Estrutura imutável com o efeito e sua incerteza.
    """

    method: str
    metric: str
    effect: float
    relative: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    significant: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializa o resultado em dicionário.

        Returns:
            Dicionário com todos os campos do efeito estimado.
        """
        return {
            "method": self.method,
            "metric": self.metric,
            "effect": self.effect,
            "relative": self.relative,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "significant": self.significant,
            "extra": self.extra,
        }
