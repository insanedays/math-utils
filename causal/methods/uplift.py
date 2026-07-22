"""Uplift modeling (efeito heterogêneo / CATE).

Estima o efeito individual do tratamento com um T-learner: um modelo ajustado
no controle e outro no tratado; o uplift de cada unidade é a diferença das
predições. Responde "para quem o tratamento funciona mais", não só o efeito
médio.

Consome o ``snapshot`` (grão ``id``) com covariáveis (``cov_*``), grupo e o
outcome do pós (``post_metric``). Requer o extra ``ml``
(``pip install -e ".[ml]"``).
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..naming import snap_metric
from ..quality.balance import _detect_control
from ..spec import MetricSpec
from .result import EffectResult


def run_uplift(
    snapshot: DataFrame,
    metric: MetricSpec,
    treatment_col: str = "treatment",
    covariates: list[str] | None = None,
    control: object | None = None,
    n_deciles: int = 10,
    return_scores: bool = False,
    seed: int = 42,
) -> EffectResult:
    """Estima o uplift por unidade via T-learner e resume por decil.

    Args:
        snapshot: DataFrame no grão ``id`` (saída de ``build_snapshot``).
        metric: Especificação da métrica (rótulo).
        treatment_col: Coluna de grupo (binarizada em tratado vs controle).
        covariates: Preditores. ``None`` usa as colunas ``cov_*``.
        control: Rótulo do controle; ``None`` detecta por heurística.
        n_deciles: Número de faixas para a tabela de uplift.
        return_scores: Se ``True``, inclui o DataFrame de scores por ``id`` em
            ``extra["scores"]``.
        seed: Semente dos modelos.

    Returns:
        :class:`EffectResult` com o uplift médio (ATE) em ``effect`` e a tabela
        de uplift previsto/observado por decil em ``extra["uplift_by_decile"]``.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    if covariates is None:
        covariates = [c for c in snapshot.columns if c.startswith("cov_")]
    if not covariates:
        return EffectResult(
            method="uplift", metric=metric.name, effect=float("nan"),
            extra={"error": "sem covariáveis (cov_*) para modelar uplift"},
        )

    outcome = snap_metric(metric.name, "post")
    cols = ["id", treatment_col, outcome, *covariates]
    pdf = snapshot.select(*[F.col(c) for c in cols]).dropna().toPandas()
    pdf = pdf.rename(columns={outcome: "post_metric"})

    groups = pdf[treatment_col].unique().tolist()
    if len(groups) < 2:
        return EffectResult(
            method="uplift", metric=metric.name, effect=float("nan"),
            extra={"error": "menos de 2 grupos"},
        )

    ctrl = control if control is not None else _detect_control(groups)
    pdf["_treated"] = (pdf[treatment_col] != ctrl).astype(int)

    X = pdf[covariates].to_numpy()
    y = pdf["post_metric"].to_numpy()
    w = pdf["_treated"].to_numpy()

    m0 = RandomForestRegressor(n_estimators=200, random_state=seed).fit(X[w == 0], y[w == 0])
    m1 = RandomForestRegressor(n_estimators=200, random_state=seed).fit(X[w == 1], y[w == 1])
    uplift = m1.predict(X) - m0.predict(X)
    pdf["_uplift"] = uplift

    ate = float(np.mean(uplift))
    decile_table = _uplift_by_decile(pdf, n_deciles)

    extra = {
        "control": ctrl,
        "ate": ate,
        "covariates": covariates,
        "uplift_by_decile": decile_table,
        "n": int(len(pdf)),
    }
    if return_scores:
        extra["scores"] = pdf[["id", "_uplift"]].rename(columns={"_uplift": "uplift"})

    return EffectResult(
        method="uplift",
        metric=metric.name,
        effect=ate,
        relative=None,
        extra=extra,
    )


def _uplift_by_decile(pdf, n_deciles: int) -> list[dict]:
    """Tabela de uplift previsto vs observado por decil de uplift previsto.

    Ordena as unidades pelo uplift previsto (desc), forma decis e compara, em
    cada faixa, o uplift previsto médio com o observado (média do outcome no
    tratado menos no controle). Faixas com efeito observado alto no topo
    indicam bom poder de segmentação.

    Args:
        pdf: DataFrame pandas com ``_uplift``, ``_treated`` e ``post_metric``.
        n_deciles: Número de faixas.

    Returns:
        Lista de dicionários por decil com uplift previsto e observado.
    """
    import pandas as pd

    ranked = pdf.sort_values("_uplift", ascending=False).reset_index(drop=True)
    ranked["_decile"] = pd.qcut(
        ranked["_uplift"].rank(method="first", ascending=False),
        q=n_deciles,
        labels=False,
    )

    table = []
    for decile, part in ranked.groupby("_decile"):
        treated = part[part["_treated"] == 1]["post_metric"]
        control = part[part["_treated"] == 0]["post_metric"]
        observed = (
            float(treated.mean() - control.mean())
            if len(treated) and len(control)
            else float("nan")
        )
        table.append(
            {
                "decile": int(decile) + 1,
                "predicted_uplift": float(part["_uplift"].mean()),
                "observed_uplift": observed,
                "n": int(len(part)),
            }
        )
    return table
