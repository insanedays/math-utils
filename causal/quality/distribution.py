"""Distribuição e normalidade: dado, validação e gráfico.

Testes de normalidade clássicos (Shapiro-Wilk) não escalam no Spark, então a
avaliação usa assimetria (skewness) e curtose calculadas nativamente sobre toda
a base. As três peças:

- ``distribution_stats`` : média, desvio, skew, curtose e quantis por grupo (Spark → pandas).
- ``check_normality``    : aprova/reprova por limiares de skew e curtose.
- ``plot_distribution``  : histograma por grupo com curva normal de referência.

Observação: com ``n`` grande, qualquer desvio minúsculo "reprova" normalidade —
priorize o gráfico e métodos robustos ao interpretar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .result import CheckResult

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure


def distribution_stats(
    df: DataFrame,
    col: str,
    group_col: str | None = None,
) -> "pd.DataFrame":
    """Calcula estatísticas de distribuição por grupo.

    Nota: ``F.kurtosis`` do Spark já retorna curtose em excesso (normal ≈ 0).

    Args:
        df: DataFrame no grão de análise (ex: ``snapshot``).
        col: Coluna numérica a avaliar.
        group_col: Coluna de grupo; ``None`` agrega tudo.

    Returns:
        DataFrame pandas com ``mean``, ``std``, ``skewness``, ``kurtosis``,
        ``p25``, ``p50``, ``p75``, ``n`` por grupo.
    """
    aggs = [
        F.avg(col).alias("mean"),
        F.stddev_samp(col).alias("std"),
        F.skewness(col).alias("skewness"),
        F.kurtosis(col).alias("kurtosis"),
        F.expr(f"percentile_approx({col}, 0.25)").alias("p25"),
        F.expr(f"percentile_approx({col}, 0.50)").alias("p50"),
        F.expr(f"percentile_approx({col}, 0.75)").alias("p75"),
        F.count(F.lit(1)).alias("n"),
    ]
    grouped = df.groupBy(group_col).agg(*aggs) if group_col else df.agg(*aggs)
    return grouped.toPandas()


def check_normality(
    df: DataFrame,
    col: str,
    group_col: str | None = None,
    skew_thr: float = 0.5,
    kurt_thr: float = 1.0,
) -> list[CheckResult]:
    """Avalia normalidade aproximada via assimetria e curtose.

    Args:
        df: DataFrame no grão de análise.
        col: Coluna numérica a avaliar.
        group_col: Coluna de grupo; ``None`` avalia a distribuição inteira.
        skew_thr: Limite para ``|skewness|`` (padrão ``0.5``).
        kurt_thr: Limite para ``|kurtose em excesso|`` (padrão ``1.0``).

    Returns:
        Lista de :class:`CheckResult`, uma por grupo. ``passed=True`` quando
        ``|skew| < skew_thr`` e ``|kurt| < kurt_thr``.
    """
    pdf = distribution_stats(df, col, group_col)
    results: list[CheckResult] = []
    for _, row in pdf.iterrows():
        group = row[group_col] if group_col else "all"
        skew = float(row["skewness"]) if row["skewness"] is not None else float("nan")
        kurt = float(row["kurtosis"]) if row["kurtosis"] is not None else float("nan")
        passed = abs(skew) < skew_thr and abs(kurt) < kurt_thr
        results.append(
            CheckResult(
                name=f"normality[{col}] {group}",
                passed=bool(passed),
                value=skew,
                criterion=f"|skew| < {skew_thr} e |kurt| < {kurt_thr}",
                extra={
                    "group": group,
                    "skewness": skew,
                    "kurtosis": kurt,
                    "mean": float(row["mean"]) if row["mean"] is not None else None,
                    "std": float(row["std"]) if row["std"] is not None else None,
                    "n": int(row["n"]),
                },
            )
        )
    return results


def histogram_data(
    df: DataFrame,
    col: str,
    bins: int = 30,
    group_col: str | None = None,
) -> "pd.DataFrame":
    """Calcula o histograma no Spark e coleta as contagens.

    Args:
        df: DataFrame no grão de análise.
        col: Coluna numérica.
        bins: Número de faixas.
        group_col: Coluna de grupo; ``None`` histograma único.

    Returns:
        DataFrame pandas com ``bin_left``, ``bin_right``, ``count`` e (se houver)
        ``group``.
    """
    bounds = df.select(F.min(col).alias("lo"), F.max(col).alias("hi")).first()
    lo, hi = bounds["lo"], bounds["hi"]
    width = (hi - lo) / bins if hi > lo else 1.0

    bucket = F.least(
        F.floor((F.col(col) - F.lit(lo)) / F.lit(width)).cast("int"),
        F.lit(bins - 1),
    )
    dfb = df.filter(F.col(col).isNotNull()).withColumn("_bucket", bucket)

    group_by = ["_bucket"]
    if group_col:
        dfb = dfb.withColumn("group", F.col(group_col))
        group_by.append("group")

    counts = dfb.groupBy(*group_by).agg(F.count(F.lit(1)).alias("count")).toPandas()
    counts["bin_left"] = lo + counts["_bucket"] * width
    counts["bin_right"] = counts["bin_left"] + width
    return counts.drop(columns="_bucket").sort_values("bin_left").reset_index(drop=True)


def plot_distribution(
    df: DataFrame,
    col: str,
    group_col: str | None = None,
    bins: int = 30,
    normal_overlay: bool = True,
) -> "Figure":
    """Plota o histograma (densidade) com curva normal de referência.

    Args:
        df: DataFrame no grão de análise.
        col: Coluna numérica a plotar.
        group_col: Coluna de grupo; ``None`` distribuição única.
        bins: Número de faixas do histograma.
        normal_overlay: Se ``True``, desenha a normal de mesma média/desvio.

    Returns:
        Figura matplotlib pronta para exibir ou salvar.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    hist = histogram_data(df, col, bins=bins, group_col=group_col)
    stats_pdf = distribution_stats(df, col, group_col)
    fig, ax = plt.subplots(figsize=(10, 5))

    groups = hist["group"].unique() if "group" in hist.columns else [None]
    for group in groups:
        part = hist[hist["group"] == group] if group is not None else hist
        width = (part["bin_right"] - part["bin_left"]).iloc[0]
        total = part["count"].sum()
        density = part["count"] / (total * width)
        label = str(group) if group is not None else col
        ax.bar(part["bin_left"], density, width=width, align="edge", alpha=0.5, label=label)

        if normal_overlay:
            if group is not None:
                srow = stats_pdf[stats_pdf[group_col] == group].iloc[0]
            else:
                srow = stats_pdf.iloc[0]
            mu, sigma = srow["mean"], srow["std"]
            if sigma and sigma > 0:
                x = np.linspace(part["bin_left"].min(), part["bin_right"].max(), 200)
                pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
                ax.plot(x, pdf, linestyle="--")

    ax.set_title(f"Distribuição de {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("densidade")
    ax.legend()
    fig.tight_layout()
    return fig
