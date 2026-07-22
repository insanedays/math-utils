"""causal — utilitários de validação e inferência causal em PySpark.

Fundação do domínio: o contrato de configuração (:mod:`causal.spec`) e a
preparação de dados (:mod:`causal.prep`), reutilizados pelas validações de
qualidade (:mod:`causal.quality`) e pelos métodos (:mod:`causal.methods`).
"""

from .prep import build_panel, build_snapshot, infer_windows
from .spec import CausalSpec, MetricSpec, Windows

__all__ = [
    "CausalSpec",
    "MetricSpec",
    "Windows",
    "infer_windows",
    "build_panel",
    "build_snapshot",
]
