"""Contratos de entrada para análises de causalidade.

Define as estruturas que descrevem *o que* será analisado (métrica, janelas,
grupos, covariáveis). Todo método causal (AB, DiD, uplift, bayesiano) consome
o mesmo ``CausalSpec``, garantindo que a etapa de preparação e as validações de
qualidade sejam reaproveitadas sem duplicação.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

from pyspark.sql import DataFrame

Agg = Literal["sum", "avg", "count"]
Grain = Literal["event", "id_date", "auto"]


@dataclass
class MetricSpec:
    """Descreve como calcular a métrica a partir de colunas do DataFrame.

    Suporta métricas simples (uma expressão agregada) e métricas de razão
    (numerador/denominador), estas últimas necessárias para tratar corretamente
    a variância de indicadores como ticket médio e conversão.

    Args:
        numerator: Expressão SQL do numerador, ex: ``"gmv"`` ou ``"case when
            converted then 1 else 0 end"``.
        denominator: Expressão SQL do denominador para métricas de razão, ex:
            ``"orders"``. ``None`` indica métrica simples.
        agg: Agregação aplicada em métricas simples (``sum``, ``avg`` ou
            ``count``). Ignorada quando ``denominator`` é definido (razão usa
            sempre ``sum`` no numerador e no denominador).
        name: Rótulo da métrica usado nas saídas.

    Returns:
        Instância imutável de configuração da métrica.
    """

    numerator: str
    denominator: str | None = None
    agg: Agg = "sum"
    name: str = "metric"

    @property
    def is_ratio(self) -> bool:
        """Indica se a métrica é de razão (tem denominador).

        Returns:
            ``True`` quando há denominador definido.
        """
        return self.denominator is not None


@dataclass
class Windows:
    """Janelas temporais pré e pós início do teste.

    Convenção de intervalos semiabertos: ``[start, end)`` (início inclusivo,
    fim exclusivo).

    Args:
        pre_start: Início da janela pré-tratamento.
        pre_end: Fim (exclusivo) da janela pré-tratamento.
        post_start: Início da janela pós-tratamento.
        post_end: Fim (exclusivo) da janela pós-tratamento.

    Returns:
        Estrutura com as quatro fronteiras de data.
    """

    pre_start: date
    pre_end: date
    post_start: date
    post_end: date

    @property
    def pre_days(self) -> int:
        """Quantidade de dias na janela pré.

        Returns:
            Número de dias do intervalo pré.
        """
        return (self.pre_end - self.pre_start).days

    @property
    def post_days(self) -> int:
        """Quantidade de dias na janela pós.

        Returns:
            Número de dias do intervalo pós.
        """
        return (self.post_end - self.post_start).days


@dataclass
class CausalSpec:
    """Configuração completa de uma análise causal.

    Reúne o DataFrame de entrada, a métrica, o recorte temporal e, quando
    aplicável, o grupo de tratamento e as covariáveis a balancear. É o objeto
    único consumido por ``prep`` e por todas as validações de qualidade.

    Args:
        df: DataFrame Spark de entrada (event-level ou já em ``id × data``).
        id_col: Coluna identificadora da unidade (usuário, restaurante, etc.).
        date_col: Coluna de data.
        metric: Uma :class:`MetricSpec` ou uma lista delas. Com lista, todas são
            calculadas na mesma passada do Spark; a primeira é a primária. Cada
            métrica precisa de um ``name`` único.
        test_start_date: Data de início do teste (``YYYY-MM-DD``), usada para
            inferir as janelas pré e pós.
        post_days: Tamanho da janela pós em dias. ``None`` infere do dado: do
            início do teste até a última data disponível.
        pre_days: Tamanho da janela pré em dias. ``None`` espelha a janela pós
            (mesma quantidade de dias para trás).
        gap_days: Dias de washout descartados ao redor do início do teste.
        treatment_col: Coluna de atribuição de grupo (ex: ``group`` com
            controle/variante). A presença dessa coluna no ``df`` indica que a
            análise tem grupos; ``None`` ou coluna ausente = método sem grupo.
        covariates: Colunas a balancear, sempre medidas na janela pré.
        grain: Grão do ``df`` de entrada (``event``, ``id_date`` ou ``auto``).

    Returns:
        Instância de configuração pronta para ``prep`` e validações.
    """

    df: DataFrame
    id_col: str
    date_col: str
    metric: MetricSpec | list[MetricSpec]
    test_start_date: str
    post_days: int | None = None
    pre_days: int | None = None
    gap_days: int = 0
    treatment_col: str | None = "group"
    covariates: list[str] = field(default_factory=list)
    grain: Grain = "auto"

    @property
    def start(self) -> date:
        """Converte ``test_start_date`` para :class:`datetime.date`.

        Returns:
            Data de início do teste.
        """
        return datetime.strptime(self.test_start_date, "%Y-%m-%d").date()

    @property
    def metric_list(self) -> list[MetricSpec]:
        """Normaliza ``metric`` numa lista de métricas.

        Returns:
            Lista de :class:`MetricSpec` (uma ou várias).
        """
        return self.metric if isinstance(self.metric, list) else [self.metric]

    @property
    def primary_metric(self) -> MetricSpec:
        """Métrica primária (primeira da lista).

        Returns:
            A :class:`MetricSpec` primária.
        """
        return self.metric_list[0]

    @property
    def has_groups(self) -> bool:
        """Indica se a análise possui grupo de tratamento.

        Considera que há grupos quando ``treatment_col`` está definido e a
        coluna existe de fato no ``df``.

        Returns:
            ``True`` quando a coluna de grupo está presente no ``df``.
        """
        return self.treatment_col is not None and self.treatment_col in self.df.columns
