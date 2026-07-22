"""Fixtures compartilhadas dos testes.

Fornece uma SparkSession local (sem cluster) e um gerador de dados sintéticos
(mock) com um efeito de tratamento plantado, usado nos testes de fumaça.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

# ── parâmetros do mock ──────────────────────────────────────────────
TEST_START = date(2026, 6, 1)
PRE_DAYS = 28
POST_DAYS = 28
N_IDS = 60
BASE_TICKET = 25.0          # ticket médio base (gmv/orders)
EFFECT_MULT = 1.15          # efeito plantado: +15% no ticket da variante no pós
SEED = 42


@pytest.fixture(scope="session")
def spark():
    """Cria uma SparkSession local para os testes.

    Returns:
        SparkSession em modo ``local[1]`` (encerrada ao fim da sessão).
    """
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[1]")
        .appName("causal-tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def _generate_rows() -> list[dict]:
    """Gera linhas event-level (id × dia) com efeito plantado na variante.

    Metade dos ids é ``control`` e metade ``variant``. No período pós, o ticket
    (``gmv / orders``) da variante é multiplicado por ``EFFECT_MULT``.

    Returns:
        Lista de dicionários com ``restaurant_id``, ``dt``, ``group``, ``gmv``,
        ``orders``.
    """
    rng = random.Random(SEED)
    start = TEST_START - timedelta(days=PRE_DAYS)
    n_days = PRE_DAYS + POST_DAYS
    dates = [start + timedelta(days=d) for d in range(n_days)]

    rows: list[dict] = []
    for i in range(N_IDS):
        group = "control" if i % 2 == 0 else "variant"
        base_ticket = rng.gauss(BASE_TICKET, 3.0)
        for dt in dates:
            orders = max(1, int(rng.gauss(10, 3)))
            is_post = dt >= TEST_START
            mult = EFFECT_MULT if (group == "variant" and is_post) else 1.0
            ticket = base_ticket * mult * (1 + rng.gauss(0, 0.05))
            gmv = orders * ticket
            rows.append(
                {
                    "restaurant_id": i,
                    "dt": dt,
                    "group": group,
                    "gmv": float(gmv),
                    "orders": int(orders),
                }
            )
    return rows


@pytest.fixture(scope="session")
def mock_events(spark):
    """DataFrame Spark event-level com efeito de +15% no ticket da variante.

    Args:
        spark: SparkSession local.

    Returns:
        DataFrame com ``restaurant_id``, ``dt``, ``group``, ``gmv``, ``orders``.
    """
    return spark.createDataFrame(_generate_rows())


@pytest.fixture()
def spec(mock_events):
    """CausalSpec pronto sobre o mock, métrica de razão ticket = gmv/orders.

    Args:
        mock_events: DataFrame de eventos.

    Returns:
        Instância de ``CausalSpec``.
    """
    from causal import CausalSpec, MetricSpec

    return CausalSpec(
        df=mock_events,
        id_col="restaurant_id",
        date_col="dt",
        metric=MetricSpec(numerator="gmv", denominator="orders", name="ticket"),
        test_start_date=TEST_START.isoformat(),
        pre_days=PRE_DAYS,
        post_days=POST_DAYS,
        treatment_col="group",
        covariates=["gmv", "orders"],
    )
