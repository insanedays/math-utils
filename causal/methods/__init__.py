"""causal.methods — estimadores de efeito causal.

Cada método consome as visões preparadas por :mod:`causal.prep` (``snapshot``
ou ``panel``) e devolve um :class:`~causal.methods.result.EffectResult`
padronizado. Rode sempre as validações de :mod:`causal.quality` antes de
confiar no efeito estimado.
"""

from .ab import run_ab
from .bayes import run_bayes_ab
from .did import did_event_study, run_did
from .result import EffectResult
from .uplift import run_uplift

__all__ = [
    "EffectResult",
    "run_ab",
    "run_bayes_ab",
    "run_did",
    "did_event_study",
    "run_uplift",
]
