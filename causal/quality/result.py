"""Resultado padronizado das validações de qualidade.

Toda função ``check_*`` retorna um ou mais :class:`CheckResult`, permitindo que
o relatório (:mod:`causal.quality.report`) renderize de forma uniforme,
independentemente da validação.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckResult:
    """Resultado de uma única validação.

    Args:
        name: Identificador legível da validação (ex: ``"balance[cov_gmv]"``).
        passed: ``True`` se a validação passou no critério.
        value: Valor observado (ex: SMD, p-valor, razão).
        criterion: Descrição textual do critério de aprovação.
        extra: Metadados adicionais (grupos comparados, n, etc.).

    Returns:
        Estrutura imutável com o desfecho da validação.
    """

    name: str
    passed: bool
    value: float
    criterion: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializa o resultado em dicionário.

        Returns:
            Dicionário com ``name``, ``passed``, ``value``, ``criterion`` e
            ``extra``.
        """
        return {
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "criterion": self.criterion,
            "extra": self.extra,
        }
