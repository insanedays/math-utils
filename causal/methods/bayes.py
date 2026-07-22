"""AB bayesiano.

Estima a distribuição a posteriori da diferença entre variante e controle e
responde perguntas de probabilidade — ``P(variante > controle)``, uplift
esperado, perda esperada e intervalo de credibilidade — em vez de p-valor.

Dois modelos:

- ``conversion`` : taxa de conversão via Beta-Binomial conjugado.
- ``continuous`` : média via posterior Normal (aproximação de amostra grande).

Consome o ``snapshot`` (grão ``id``).
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..naming import snap_den, snap_metric, snap_num
from ..quality.balance import _detect_control
from ..spec import MetricSpec
from .result import EffectResult


def run_bayes_ab(
    snapshot: DataFrame,
    metric: MetricSpec,
    kind: str = "continuous",
    treatment_col: str = "treatment",
    control: object | None = None,
    period: str = "post",
    cred_mass: float = 0.95,
    n_samples: int = 100_000,
    prior: tuple[float, float] = (1.0, 1.0),
    seed: int = 42,
) -> list[EffectResult]:
    """Estima o efeito bayesiano de cada variante contra o controle.

    Args:
        snapshot: DataFrame no grão ``id`` (saída de ``build_snapshot``).
        metric: Especificação da métrica (usada para o rótulo).
        kind: ``"conversion"`` (Beta-Binomial) ou ``"continuous"`` (Normal).
        treatment_col: Coluna de grupo.
        control: Rótulo do controle; ``None`` detecta por heurística.
        period: Período avaliado.
        cred_mass: Massa do intervalo de credibilidade (padrão ``0.95``).
        n_samples: Amostras Monte Carlo da posterior.
        prior: Prior Beta ``(a, b)`` para ``conversion``.
        seed: Semente do gerador aleatório.

    Returns:
        Lista de :class:`EffectResult`. ``p_value`` é ``None``; a evidência fica
        em ``extra`` (``prob_better``, ``expected_loss``).
    """
    if kind == "conversion":
        stats_by_group = _conversion_stats(snapshot, metric, treatment_col, period)
    else:
        stats_by_group = _continuous_stats(snapshot, metric, treatment_col, period)

    groups = list(stats_by_group.keys())
    if len(groups) < 2:
        return []

    ctrl = control if control is not None else _detect_control(groups)
    variants = [g for g in groups if g != ctrl]

    results: list[EffectResult] = []
    for variant in variants:
        results.append(
            _posterior_effect(
                metric, kind, variant, ctrl,
                stats_by_group[variant], stats_by_group[ctrl],
                cred_mass, n_samples, prior, seed,
            )
        )
    return results


def _conversion_stats(
    snapshot: DataFrame, metric: MetricSpec, treatment_col: str, period: str
) -> dict:
    """Soma sucessos e tentativas por grupo (modelo de conversão).

    Args:
        snapshot: DataFrame no grão ``id``.
        metric: Especificação da métrica (define as colunas lidas).
        treatment_col: Coluna de grupo.
        period: Período avaliado.

    Returns:
        Dicionário ``{grupo: {"successes", "trials"}}``.
    """
    num, den = snap_num(metric.name, period), snap_den(metric.name, period)
    rows = (
        snapshot.groupBy(treatment_col)
        .agg(F.sum(num).alias("succ"), F.sum(den).alias("trials"))
        .collect()
    )
    return {r[treatment_col]: {"successes": r["succ"], "trials": r["trials"]} for r in rows}


def _continuous_stats(
    snapshot: DataFrame, metric: MetricSpec, treatment_col: str, period: str
) -> dict:
    """Média, desvio e n do valor por unidade por grupo (modelo contínuo).

    Args:
        snapshot: DataFrame no grão ``id``.
        metric: Especificação da métrica (define as colunas lidas).
        treatment_col: Coluna de grupo.
        period: Período avaliado.

    Returns:
        Dicionário ``{grupo: {"mean", "std", "n"}}``.
    """
    col = snap_metric(metric.name, period)
    rows = (
        snapshot.groupBy(treatment_col)
        .agg(
            F.avg(col).alias("mean"),
            F.stddev_samp(col).alias("std"),
            F.count(F.lit(1)).alias("n"),
        )
        .collect()
    )
    return {
        r[treatment_col]: {"mean": r["mean"], "std": r["std"] or 0.0, "n": r["n"]}
        for r in rows
    }


def _posterior_effect(
    metric: MetricSpec, kind: str, variant, ctrl, t: dict, c: dict,
    cred_mass: float, n_samples: int, prior: tuple[float, float], seed: int,
) -> EffectResult:
    """Amostra a posterior e monta o :class:`EffectResult` bayesiano."""
    import numpy as np

    rng = np.random.default_rng(seed)

    if kind == "conversion":
        a, b = prior
        st = rng.beta(a + t["successes"], b + t["trials"] - t["successes"], n_samples)
        sc = rng.beta(a + c["successes"], b + c["trials"] - c["successes"], n_samples)
    else:
        # posterior Normal da média: mean ~ N(xbar, s^2 / n)
        st = rng.normal(t["mean"], t["std"] / np.sqrt(t["n"]), n_samples)
        sc = rng.normal(c["mean"], c["std"] / np.sqrt(c["n"]), n_samples)

    diff = st - sc
    prob_better = float(np.mean(diff > 0))
    lo = (1 - cred_mass) / 2
    ci_low, ci_high = np.quantile(diff, [lo, 1 - lo])
    # perda esperada ao escolher a variante (massa em que ela é pior)
    expected_loss = float(np.mean(np.maximum(sc - st, 0.0)))
    mean_c = float(np.mean(sc))
    effect = float(np.mean(diff))

    return EffectResult(
        method=f"bayes_{kind}",
        metric=metric.name,
        effect=effect,
        relative=(effect / mean_c) if mean_c else None,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=None,
        significant=None,
        extra={
            "variant": variant, "control": ctrl,
            "prob_better": prob_better,
            "expected_loss": expected_loss,
            "cred_mass": cred_mass,
        },
    )
