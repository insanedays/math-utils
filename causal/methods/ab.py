"""AB clássico (frequentista).

Compara controle e variante(s) na métrica do período pós. Trata:

- métrica simples: teste t de Welch sobre o valor por unidade;
- métrica de razão: delta method por grupo + teste z da diferença.

Consome o ``snapshot`` (grão ``id``) produzido por ``build_snapshot``.
"""

from __future__ import annotations

import math

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..naming import snap_den, snap_metric, snap_num
from ..quality.balance import _detect_control
from ..quality.stat_tests import delta_method_ratio, two_sample_z
from ..spec import MetricSpec
from .result import EffectResult


def run_ab(
    snapshot: DataFrame,
    metric: MetricSpec,
    treatment_col: str = "treatment",
    control: object | None = None,
    alpha: float = 0.05,
    period: str = "post",
) -> list[EffectResult]:
    """Estima o efeito de cada variante contra o controle (AB clássico).

    Args:
        snapshot: DataFrame no grão ``id`` (saída de ``build_snapshot``).
        metric: Especificação da métrica (define razão vs simples).
        treatment_col: Coluna de grupo.
        control: Rótulo do controle; ``None`` detecta por heurística.
        alpha: Nível de significância.
        period: Período avaliado (``post`` por padrão).

    Returns:
        Lista de :class:`EffectResult`, uma por variante.
    """
    if metric.is_ratio:
        stats_by_group = _ratio_group_stats(snapshot, metric, treatment_col, period)
    else:
        stats_by_group = _simple_group_stats(snapshot, metric, treatment_col, period)

    groups = list(stats_by_group.keys())
    if len(groups) < 2:
        return []

    ctrl = control if control is not None else _detect_control(groups)
    variants = [g for g in groups if g != ctrl]

    results: list[EffectResult] = []
    for variant in variants:
        if metric.is_ratio:
            results.append(
                _ratio_effect(
                    metric, variant, ctrl,
                    stats_by_group[variant], stats_by_group[ctrl], alpha,
                )
            )
        else:
            results.append(
                _simple_effect(
                    metric, variant, ctrl,
                    stats_by_group[variant], stats_by_group[ctrl], alpha,
                )
            )
    return results


def _simple_group_stats(
    snapshot: DataFrame, metric: MetricSpec, treatment_col: str, period: str
) -> dict:
    """Agrega média, desvio e n do valor por unidade por grupo.

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


def _ratio_group_stats(
    snapshot: DataFrame, metric: MetricSpec, treatment_col: str, period: str
) -> dict:
    """Agrega os momentos de numerador/denominador por grupo (delta method).

    Args:
        snapshot: DataFrame no grão ``id``.
        metric: Especificação da métrica (define as colunas lidas).
        treatment_col: Coluna de grupo.
        period: Período avaliado.

    Returns:
        Dicionário ``{grupo: {"mean_num", "mean_den", "var_num", "var_den",
        "cov", "n"}}``.
    """
    num, den = snap_num(metric.name, period), snap_den(metric.name, period)
    rows = (
        snapshot.groupBy(treatment_col)
        .agg(
            F.avg(num).alias("mean_num"),
            F.avg(den).alias("mean_den"),
            F.var_samp(num).alias("var_num"),
            F.var_samp(den).alias("var_den"),
            F.covar_samp(num, den).alias("cov"),
            F.count(F.lit(1)).alias("n"),
        )
        .collect()
    )
    return {
        r[treatment_col]: {
            "mean_num": r["mean_num"],
            "mean_den": r["mean_den"],
            "var_num": r["var_num"] or 0.0,
            "var_den": r["var_den"] or 0.0,
            "cov": r["cov"] or 0.0,
            "n": r["n"],
        }
        for r in rows
    }


def _simple_effect(
    metric: MetricSpec, variant, ctrl, t: dict, c: dict, alpha: float
) -> EffectResult:
    """Efeito de métrica simples via Welch (diferença de médias + CI)."""
    from scipy import stats

    diff = t["mean"] - c["mean"]
    se = math.sqrt(t["std"] ** 2 / t["n"] + c["std"] ** 2 / c["n"]) if t["n"] and c["n"] else 0.0

    if se == 0:
        p, ci_low, ci_high = float("nan"), diff, diff
    else:
        # graus de liberdade de Welch–Satterthwaite
        num_df = (t["std"] ** 2 / t["n"] + c["std"] ** 2 / c["n"]) ** 2
        den_df = (
            (t["std"] ** 2 / t["n"]) ** 2 / (t["n"] - 1)
            + (c["std"] ** 2 / c["n"]) ** 2 / (c["n"] - 1)
        )
        dof = num_df / den_df if den_df else min(t["n"], c["n"]) - 1
        tstat = diff / se
        p = 2 * (1 - stats.t.cdf(abs(tstat), dof))
        crit = stats.t.ppf(1 - alpha / 2, dof)
        ci_low, ci_high = diff - crit * se, diff + crit * se

    relative = diff / c["mean"] if c["mean"] else None
    return EffectResult(
        method="ab",
        metric=metric.name,
        effect=float(diff),
        relative=relative,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p),
        significant=bool(p < alpha) if p == p else None,  # NaN-safe
        extra={
            "variant": variant, "control": ctrl,
            "mean_variant": t["mean"], "mean_control": c["mean"],
            "n_variant": t["n"], "n_control": c["n"], "std_error": se,
        },
    )


def _ratio_effect(
    metric: MetricSpec, variant, ctrl, t: dict, c: dict, alpha: float
) -> EffectResult:
    """Efeito de métrica de razão via delta method + teste z."""
    r_t, var_t = delta_method_ratio(
        t["mean_num"], t["mean_den"], t["var_num"], t["var_den"], t["cov"], t["n"]
    )
    r_c, var_c = delta_method_ratio(
        c["mean_num"], c["mean_den"], c["var_num"], c["var_den"], c["cov"], c["n"]
    )
    diff, p, ci_low, ci_high = two_sample_z(r_t, r_c, var_t, var_c, alpha)
    relative = diff / r_c if r_c else None
    return EffectResult(
        method="ab",
        metric=metric.name,
        effect=float(diff),
        relative=relative,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p),
        significant=bool(p < alpha) if p == p else None,
        extra={
            "variant": variant, "control": ctrl,
            "ratio_variant": r_t, "ratio_control": r_c,
            "n_variant": t["n"], "n_control": c["n"],
        },
    )
