"""Preparação de dados para análises de causalidade.

Transforma o DataFrame de entrada (event-level ou ``id × data``) nas duas
visões reaproveitadas por todos os métodos:

- ``panel``   : grão ``id × data`` — série temporal (tendências, DiD, bayesiano).
- ``snapshot``: grão ``id × período`` — métrica agregada por unidade em pré/pós
  mais covariáveis medidas na janela pré (balanceamento, AB, uplift).

Como agregar um DataFrame já em ``id × data`` por ``id × data`` é idempotente,
a preparação sempre agrega para ``id × data``, funcionando de forma idêntica
para os dois grãos de entrada.
"""

from __future__ import annotations

from datetime import timedelta

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from .naming import (
    cov,
    panel_den,
    panel_metric,
    panel_num,
    snap_delta,
    snap_den,
    snap_metric,
    snap_num,
)
from .spec import CausalSpec, MetricSpec, Windows


def _max_date(spec: CausalSpec):
    """Obtém a última data disponível no ``df``.

    Args:
        spec: Configuração da análise.

    Returns:
        Maior valor de ``date_col`` como :class:`datetime.date`.
    """
    return spec.df.select(F.max(F.to_date(F.col(spec.date_col)))).first()[0]


def infer_windows(spec: CausalSpec) -> Windows:
    """Deriva as janelas pré e pós a partir da data de início do teste.

    Basta informar ``test_start_date``. A janela pós vai do início do teste até
    a última data disponível no dado (quando ``post_days`` é ``None``) e a pré
    espelha esse tamanho para trás (quando ``pre_days`` é ``None``). Aplica
    ``gap_days`` como washout em torno do início. Intervalos são semiabertos
    ``[start, end)``, então a última data do dado fica incluída na pós.

    Args:
        spec: Configuração da análise.

    Returns:
        Janelas :class:`~causal.spec.Windows` calculadas.
    """
    start = spec.start
    gap = timedelta(days=spec.gap_days)
    post_start = start + gap
    pre_end = start - gap

    if spec.post_days is None:
        last_date = _max_date(spec)
        # +1 dia para incluir a última data no intervalo semiaberto.
        post_end = last_date + timedelta(days=1)
    else:
        post_end = post_start + timedelta(days=spec.post_days)

    post_len = (post_end - post_start).days
    pre_len = spec.pre_days if spec.pre_days is not None else post_len
    pre_start = pre_end - timedelta(days=pre_len)

    return Windows(
        pre_start=pre_start,
        pre_end=pre_end,
        post_start=post_start,
        post_end=post_end,
    )


def _period_col(date_col: str, w: Windows) -> Column:
    """Cria a coluna de rótulo de período (``pre``/``post``/``out``).

    Args:
        date_col: Nome da coluna de data.
        w: Janelas de referência.

    Returns:
        Coluna Spark com o período de cada linha.
    """
    d = F.to_date(F.col(date_col))
    return (
        F.when((d >= F.lit(w.pre_start)) & (d < F.lit(w.pre_end)), F.lit("pre"))
        .when((d >= F.lit(w.post_start)) & (d < F.lit(w.post_end)), F.lit("post"))
        .otherwise(F.lit("out"))
    )


def _metric_components(metric: MetricSpec) -> tuple[Column, Column]:
    """Monta as colunas agregadas de numerador e denominador.

    Para métrica de razão, agrega numerador e denominador por ``sum`` (o valor
    final é ``sum(num) / sum(den)``, computado depois). Para métrica simples,
    aplica a agregação configurada no numerador e deixa o denominador nulo.

    Args:
        metric: Especificação da métrica.

    Returns:
        Tupla ``(numerator_agg, denominator_agg)`` de colunas Spark.
    """
    if metric.is_ratio:
        num = F.sum(F.expr(metric.numerator))
        den = F.sum(F.expr(metric.denominator))
        return num, den

    if metric.agg == "count":
        num = F.count(F.lit(1))
    elif metric.agg == "avg":
        num = F.avg(F.expr(metric.numerator))
    else:
        num = F.sum(F.expr(metric.numerator))
    return num, F.lit(None).cast("double")


def _metric_value(num_col: str, den_col: str, metric: MetricSpec) -> Column:
    """Combina numerador e denominador no valor final da métrica.

    Args:
        num_col: Nome da coluna de numerador agregado.
        den_col: Nome da coluna de denominador agregado.
        metric: Especificação da métrica.

    Returns:
        Coluna Spark com o valor da métrica (protege divisão por zero).
    """
    if metric.is_ratio:
        den = F.col(den_col)
        return F.when(den != 0, F.col(num_col) / den).otherwise(F.lit(None))
    return F.col(num_col)


def build_panel(spec: CausalSpec) -> DataFrame:
    """Constrói o painel ``id × data`` com todas as métricas e o período.

    Agrega o DataFrame de entrada para o grão ``id × data`` e calcula, na mesma
    passada, os componentes de cada métrica de ``spec.metric_list``. Cada
    métrica gera colunas prefixadas pelo nome (ver :mod:`causal.naming`).

    Args:
        spec: Configuração da análise.

    Returns:
        DataFrame com ``id``, ``date``, ``period``, ``<m>__num``, ``<m>__den``,
        ``<m>__metric`` para cada métrica e (opcional) ``treatment``.
    """
    w = infer_windows(spec)
    group_cols = [
        F.col(spec.id_col).alias("id"),
        F.to_date(F.col(spec.date_col)).alias("date"),
    ]

    agg_exprs = []
    for m in spec.metric_list:
        num_agg, den_agg = _metric_components(m)
        agg_exprs.append(num_agg.alias(panel_num(m.name)))
        agg_exprs.append(den_agg.alias(panel_den(m.name)))
    if spec.has_groups:
        agg_exprs.append(F.max(F.col(spec.treatment_col)).alias("treatment"))

    panel = spec.df.groupBy(*group_cols).agg(*agg_exprs).withColumn(
        "period", _period_col("date", w)
    )
    for m in spec.metric_list:
        panel = panel.withColumn(
            panel_metric(m.name),
            _metric_value(panel_num(m.name), panel_den(m.name), m),
        )
    return panel


def build_snapshot(spec: CausalSpec) -> DataFrame:
    """Constrói o snapshot ``id × período`` para balanceamento e efeito.

    Colapsa o painel para uma linha por ``id`` usando agregação condicional por
    período — assim todas as métricas e ambos os períodos (``pre``/``post``)
    saem de uma única passada. Anexa as covariáveis medidas só na janela pré.

    Args:
        spec: Configuração da análise.

    Returns:
        DataFrame com uma linha por ``id`` contendo, para cada métrica,
        ``<m>__{pre,post}_num``, ``<m>__{pre,post}_den``,
        ``<m>__{pre,post}_metric`` e ``<m>__delta``; além do ``treatment`` (se
        houver) e das covariáveis pré (prefixadas com ``cov_``).
    """
    w = infer_windows(spec)
    panel = build_panel(spec).filter(F.col("period").isin("pre", "post"))

    aggs = []
    for m in spec.metric_list:
        for period in ("pre", "post"):
            cond = F.col("period") == period
            aggs.append(
                F.sum(F.when(cond, F.col(panel_num(m.name)))).alias(snap_num(m.name, period))
            )
            aggs.append(
                F.sum(F.when(cond, F.col(panel_den(m.name)))).alias(snap_den(m.name, period))
            )
    if spec.has_groups:
        aggs.append(F.max("treatment").alias("treatment"))

    snapshot = panel.groupBy("id").agg(*aggs)

    for m in spec.metric_list:
        for period in ("pre", "post"):
            snapshot = snapshot.withColumn(
                snap_metric(m.name, period),
                _metric_value(snap_num(m.name, period), snap_den(m.name, period), m),
            )
        snapshot = snapshot.withColumn(
            snap_delta(m.name),
            F.col(snap_metric(m.name, "post")) - F.col(snap_metric(m.name, "pre")),
        )

    if spec.covariates:
        snapshot = snapshot.join(_pre_covariates(spec, w), on="id", how="left")

    return snapshot


def _pre_covariates(spec: CausalSpec, w: Windows) -> DataFrame:
    """Agrega covariáveis por ``id`` usando apenas a janela pré.

    Args:
        spec: Configuração da análise.
        w: Janelas de referência.

    Returns:
        DataFrame ``id`` × covariáveis (média na janela pré, prefixo ``cov_``).
    """
    d = F.to_date(F.col(spec.date_col))
    pre = spec.df.filter((d >= F.lit(w.pre_start)) & (d < F.lit(w.pre_end)))
    aggs = [F.avg(F.expr(c)).alias(cov(c)) for c in spec.covariates]
    return pre.groupBy(F.col(spec.id_col).alias("id")).agg(*aggs)
