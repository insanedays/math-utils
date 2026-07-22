"""causal.quality — validações de qualidade de dados para análise causal.

Cada módulo expõe funções ``check_*`` que recebem o ``panel`` ou o
``snapshot`` (montados por :mod:`causal.prep`) e devolvem um resultado
padronizado ``{"name", "passed", "value", "criterion"}``.
"""

from .balance import check_balance
from .distribution import (
    check_normality,
    distribution_stats,
    histogram_data,
    plot_distribution,
)
from .result import CheckResult
from .trends import check_parallel_trends, daily_series, plot_parallel_trends

__all__ = [
    "CheckResult",
    "check_balance",
    "check_normality",
    "distribution_stats",
    "histogram_data",
    "plot_distribution",
    "check_parallel_trends",
    "daily_series",
    "plot_parallel_trends",
]
