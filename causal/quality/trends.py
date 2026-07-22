"""Tendências paralelas: dado, validação e gráfico.

Pressuposto central do Diff-in-Diff: antes do tratamento, tratamento e controle
evoluem em paralelo. Aqui há três peças reaproveitáveis:

- ``daily_series``          : agrega o painel para série diária por grupo (Spark → pandas).
- ``check_parallel_trends`` : testa se a diferença variante−controle tem tendência no pré.
- ``plot_parallel_trends``  : gráfico das séries com marcação do início do teste.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..naming import panel_den, panel_metric, panel_num
from .balance import _detect_control
from .result import CheckResult

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure


def _daily_value_expr(metric_name: str):
    """Expressão do valor diário por grupo, robusta a métrica simples/razão.

    Usa ``sum(num)/sum(den)`` quando há denominador (métrica de razão) e a média
    do valor por unidade caso contrário.

    Args:
        metric_name: Nome da métrica cujas colunas devem ser lidas.

    Returns:
        Coluna Spark com o valor agregado do dia.
    """
    den = F.sum(panel_den(metric_name))
    return F.when(
        den.isNotNull() & (den != 0), F.sum(panel_num(metric_name)) / den
    ).otherwise(F.avg(panel_metric(metric_name)))


def daily_series(
    panel: DataFrame,
    metric_name: str,
    group_col: str = "treatment",
    periods: tuple[str, ...] = ("pre", "post"),
) -> "pd.DataFrame":
    """Agrega o painel para uma série diária por grupo.

    Args:
        panel: Painel ``id × data`` (saída de ``build_panel``).
        metric_name: Nome da métrica a plotar (``MetricSpec.name``).
        group_col: Coluna de grupo; ignorada se ausente (série única).
        periods: Períodos a incluir (``pre``, ``post``, ``out``).

    Returns:
        DataFrame pandas com colunas ``date``, ``value`` e (se houver) ``group``,
        ordenado por data.
    """
    df = panel.filter(F.col("period").isin(list(periods)))
    group_by = ["date"]
    has_group = group_col in panel.columns
    if has_group:
        df = df.withColumn("group", F.col(group_col))
        group_by.append("group")

    agg = df.groupBy(*group_by).agg(_daily_value_expr(metric_name).alias("value"))
    pdf = agg.toPandas().sort_values("date").reset_index(drop=True)
    return pdf


def check_parallel_trends(
    panel: DataFrame,
    metric_name: str,
    group_col: str = "treatment",
    control: object | None = None,
    alpha: float = 0.05,
) -> list[CheckResult]:
    """Testa tendências paralelas no período pré.

    Para cada variante, calcula a série diária da diferença ``variante −
    controle`` no pré e ajusta uma regressão linear no tempo. Se a inclinação
    não é significativa, as tendências são consideradas paralelas.

    Args:
        panel: Painel ``id × data`` (saída de ``build_panel``).
        metric_name: Nome da métrica avaliada (``MetricSpec.name``).
        group_col: Coluna de grupo.
        control: Rótulo do controle; ``None`` detecta por heurística.
        alpha: Nível de significância (padrão ``0.05``).

    Returns:
        Lista de :class:`CheckResult`, uma por variante. ``passed=True`` quando
        ``p > alpha`` (sem divergência de tendência no pré).
    """
    from scipy import stats

    if group_col not in panel.columns:
        return [
            CheckResult(
                name="parallel_trends",
                passed=False,
                value=float("nan"),
                criterion=f"p > {alpha}",
                extra={"error": "sem coluna de grupo — teste não aplicável"},
            )
        ]

    pdf = daily_series(panel, metric_name, group_col=group_col, periods=("pre",))
    wide = pdf.pivot(index="date", columns="group", values="value").sort_index()
    groups = list(wide.columns)

    if len(groups) < 2:
        return [
            CheckResult(
                name="parallel_trends",
                passed=False,
                value=float("nan"),
                criterion=f"p > {alpha}",
                extra={"error": "menos de 2 grupos no pré", "groups": groups},
            )
        ]

    ctrl = control if control is not None else _detect_control(groups)
    variants = [g for g in groups if g != ctrl]

    results: list[CheckResult] = []
    for variant in variants:
        pair = wide[[variant, ctrl]].dropna()
        diff = (pair[variant] - pair[ctrl]).to_numpy()
        t = range(len(diff))
        reg = stats.linregress(list(t), diff)
        results.append(
            CheckResult(
                name=f"parallel_trends[{variant} vs {ctrl}]",
                passed=bool(reg.pvalue > alpha),
                value=float(reg.pvalue),
                criterion=f"p > {alpha} (inclinação da diferença ≈ 0 no pré)",
                extra={
                    "variant": variant,
                    "control": ctrl,
                    "slope": float(reg.slope),
                    "r_squared": float(reg.rvalue) ** 2,
                    "n_days": len(diff),
                },
            )
        )
    return results


def plot_parallel_trends(
    panel: DataFrame,
    metric_name: str,
    group_col: str = "treatment",
    event_date=None,
    title: str = "Tendências por grupo",
) -> "Figure":
    """Plota as séries diárias por grupo com marcação do início do teste.

    Args:
        panel: Painel ``id × data`` (saída de ``build_panel``).
        metric_name: Nome da métrica a plotar (``MetricSpec.name``).
        group_col: Coluna de grupo.
        event_date: Data de início do teste para a linha vertical (opcional).
        title: Título do gráfico.

    Returns:
        Figura matplotlib pronta para exibir ou salvar.
    """
    import matplotlib.pyplot as plt

    pdf = daily_series(panel, metric_name, group_col=group_col, periods=("pre", "post"))
    fig, ax = plt.subplots(figsize=(10, 5))

    if "group" in pdf.columns:
        for group, part in pdf.groupby("group"):
            ax.plot(part["date"], part["value"], marker=".", label=str(group))
        ax.legend(title=group_col)
    else:
        ax.plot(pdf["date"], pdf["value"], marker=".")

    if event_date is not None:
        ax.axvline(event_date, color="red", linestyle="--", label="início do teste")

    ax.set_title(title)
    ax.set_xlabel("data")
    ax.set_ylabel("métrica")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
