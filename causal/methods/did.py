"""Diff-in-Diff (diferença em diferenças).

Efeito causal = (pós − pré no tratamento) − (pós − pré no controle). Estimado
por regressão ``metric ~ treated * post`` com erro-padrão clusterizado por
``id``; o coeficiente da interação ``treated:post`` é o efeito.

Pressuposto crítico: tendências paralelas no pré — valide com
``causal.quality.check_parallel_trends`` antes.

Requer o extra ``models`` (``pip install -e ".[models]"``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..naming import snap_metric
from ..quality.balance import _detect_control
from ..quality.trends import daily_series
from ..spec import MetricSpec
from .result import EffectResult

if TYPE_CHECKING:
    import pandas as pd


def run_did(
    snapshot: DataFrame,
    metric: MetricSpec,
    treatment_col: str = "treatment",
    control: object | None = None,
    alpha: float = 0.05,
) -> list[EffectResult]:
    """Estima o efeito Diff-in-Diff de cada variante contra o controle.

    Reorganiza o ``snapshot`` para o formato longo (duas linhas por ``id``:
    pré e pós) e ajusta ``metric ~ treated * post`` com SE clusterizado por
    ``id``.

    Args:
        snapshot: DataFrame no grão ``id`` com ``pre_metric``/``post_metric``.
        metric: Especificação da métrica (rótulo).
        treatment_col: Coluna de grupo.
        control: Rótulo do controle; ``None`` detecta por heurística.
        alpha: Nível de significância.

    Returns:
        Lista de :class:`EffectResult`, uma por variante.
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    pdf = snapshot.select(
        F.col("id").alias("id"),
        F.col(treatment_col).alias("group"),
        F.col(snap_metric(metric.name, "pre")).alias("pre"),
        F.col(snap_metric(metric.name, "post")).alias("post"),
    ).toPandas()

    groups = pdf["group"].dropna().unique().tolist()
    if len(groups) < 2:
        return []

    ctrl = control if control is not None else _detect_control(groups)
    variants = [g for g in groups if g != ctrl]

    results: list[EffectResult] = []
    for variant in variants:
        pair = pdf[pdf["group"].isin([variant, ctrl])].copy()
        long = pair.melt(
            id_vars=["id", "group"],
            value_vars=["pre", "post"],
            var_name="period",
            value_name="metric",
        ).dropna(subset=["metric"])
        long["treated"] = (long["group"] == variant).astype(int)
        long["post"] = (long["period"] == "post").astype(int)

        model = smf.ols("metric ~ treated * post", data=long).fit(
            cov_type="cluster", cov_kwds={"groups": long["id"]}
        )
        coef = model.params["treated:post"]
        p = model.pvalues["treated:post"]
        ci = model.conf_int(alpha=alpha).loc["treated:post"]

        base_c = pair[pair["group"] == ctrl]["post"].mean()
        results.append(
            EffectResult(
                method="did",
                metric=metric.name,
                effect=float(coef),
                relative=float(coef / base_c) if base_c else None,
                ci_low=float(ci[0]),
                ci_high=float(ci[1]),
                p_value=float(p),
                significant=bool(p < alpha),
                extra={"variant": variant, "control": ctrl, "n_obs": int(len(long))},
            )
        )
    return results


def did_event_study(
    panel: DataFrame,
    metric_name: str,
    group_col: str = "treatment",
    control: object | None = None,
) -> "pd.DataFrame":
    """Série descritiva da diferença variante−controle por dia (event study).

    Útil para inspecionar visualmente a dinâmica do efeito: no pré deve oscilar
    em torno de zero (paralelismo); no pós, afastar-se indica efeito.

    Args:
        panel: Painel ``id × data`` (saída de ``build_panel``).
        metric_name: Nome da métrica avaliada (``MetricSpec.name``).
        group_col: Coluna de grupo.
        control: Rótulo do controle; ``None`` detecta por heurística.

    Returns:
        DataFrame pandas com ``date``, ``diff`` (variante − controle) por
        variante em ``variant``.
    """
    pdf = daily_series(panel, metric_name, group_col=group_col, periods=("pre", "post"))
    wide = pdf.pivot(index="date", columns="group", values="value").sort_index()
    groups = list(wide.columns)
    ctrl = control if control is not None else _detect_control(groups)

    frames = []
    for variant in [g for g in groups if g != ctrl]:
        d = (wide[variant] - wide[ctrl]).rename("diff").reset_index()
        d["variant"] = variant
        frames.append(d)

    import pandas as pd

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
