"""Validação de balanceamento de covariáveis entre grupos.

Avalia se controle e variante(s) partem equilibrados nas covariáveis medidas na
janela pré (AA test / balanceamento de PSM). Desequilíbrio indica que uma
diferença observada no pós pode ser confundida por características prévias.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..naming import snap_metric
from ..spec import MetricSpec
from .result import CheckResult
from .stat_tests import smd, welch_ttest

# Rótulos comumente usados para o grupo de controle.
_CONTROL_LABELS = {"control", "controle", "c", "0", "a"}


def _detect_control(groups: list) -> object:
    """Identifica o rótulo do grupo de controle entre os grupos presentes.

    Args:
        groups: Valores distintos da coluna de tratamento.

    Returns:
        O rótulo tratado como controle (heurística por nome/valor; fallback no
        menor valor ordenado).
    """
    for g in groups:
        if str(g).strip().lower() in _CONTROL_LABELS:
            return g
    return sorted(groups, key=str)[0]


def check_balance(
    snapshot: DataFrame,
    treatment_col: str = "treatment",
    covariates: list[str] | None = None,
    metric: MetricSpec | None = None,
    control: object | None = None,
    threshold: float = 0.1,
    with_pvalue: bool = False,
) -> list[CheckResult]:
    """Avalia o balanceamento de covariáveis entre grupos via SMD.

    Agrega média, desvio e contagem por grupo (uma passada no Spark), coleta o
    resultado pequeno e calcula a Standardized Mean Difference de cada
    covariável de cada variante contra o controle.

    Args:
        snapshot: DataFrame no grão ``id`` (saída de ``build_snapshot``) com a
            coluna de grupo e as covariáveis.
        treatment_col: Nome da coluna de grupo.
        covariates: Covariáveis a avaliar. ``None`` detecta as colunas com
            prefixo ``cov_``.
        metric: Se informada, inclui também o balanceamento do valor pré da
            métrica (``<metric>__pre_metric``), um forte sinal de equilíbrio.
        control: Rótulo do grupo de controle. ``None`` detecta por heurística.
        threshold: Limite de aprovação para ``|SMD|`` (padrão ``0.1``).
        with_pvalue: Se ``True``, inclui o p-valor do teste t de Welch em
            ``extra`` (diagnóstico complementar; o critério segue o SMD).

    Returns:
        Lista de :class:`CheckResult`, uma por covariável × variante.
    """
    if covariates is None:
        covariates = [c for c in snapshot.columns if c.startswith("cov_")]
        if metric is not None:
            covariates.append(snap_metric(metric.name, "pre"))

    if not covariates:
        return []

    stats_by_group = _group_stats(snapshot, treatment_col, covariates)
    groups = list(stats_by_group.keys())

    if len(groups) < 2:
        return [
            CheckResult(
                name="balance",
                passed=False,
                value=float("nan"),
                criterion=f"|SMD| < {threshold}",
                extra={"error": "menos de 2 grupos encontrados", "groups": groups},
            )
        ]

    ctrl = control if control is not None else _detect_control(groups)
    variants = [g for g in groups if g != ctrl]

    results: list[CheckResult] = []
    for variant in variants:
        for cov in covariates:
            t = stats_by_group[variant][cov]
            c = stats_by_group[ctrl][cov]
            value = smd(t["mean"], c["mean"], t["std"], c["std"])
            extra = {
                "covariate": cov,
                "variant": variant,
                "control": ctrl,
                "mean_variant": t["mean"],
                "mean_control": c["mean"],
                "n_variant": t["n"],
                "n_control": c["n"],
            }
            if with_pvalue:
                _, p = welch_ttest(
                    t["mean"], c["mean"], t["std"], c["std"], t["n"], c["n"]
                )
                extra["p_value"] = p

            results.append(
                CheckResult(
                    name=f"balance[{cov}] {variant} vs {ctrl}",
                    passed=abs(value) < threshold,
                    value=float(value),
                    criterion=f"|SMD| < {threshold}",
                    extra=extra,
                )
            )
    return results


def _group_stats(
    snapshot: DataFrame,
    treatment_col: str,
    covariates: list[str],
) -> dict:
    """Agrega média, desvio e contagem por grupo para cada covariável.

    Args:
        snapshot: DataFrame no grão ``id``.
        treatment_col: Nome da coluna de grupo.
        covariates: Covariáveis a agregar.

    Returns:
        Dicionário ``{grupo: {covariável: {"mean", "std", "n"}}}`` já coletado
        no driver.
    """
    aggs = []
    for c in covariates:
        aggs.append(F.avg(F.col(c)).alias(f"{c}__mean"))
        aggs.append(F.stddev_samp(F.col(c)).alias(f"{c}__std"))
    aggs.append(F.count(F.lit(1)).alias("__n"))

    rows = snapshot.groupBy(treatment_col).agg(*aggs).collect()

    out: dict = {}
    for row in rows:
        group = row[treatment_col]
        out[group] = {
            c: {
                "mean": row[f"{c}__mean"],
                "std": row[f"{c}__std"] or 0.0,
                "n": row["__n"],
            }
            for c in covariates
        }
    return out
