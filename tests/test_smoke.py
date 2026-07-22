"""Testes de fumaça: o fluxo roda ponta a ponta sem quebrar.

Objetivo é pegar erros bobos (import, nome de coluna, sintaxe Spark) e conferir
que o efeito plantado (+15% no ticket da variante) é grosseiramente recuperado.
Não é validação estatística fina.
"""

from __future__ import annotations

import pytest

from causal import build_panel, build_snapshot, infer_windows
from causal.naming import snap_delta, snap_den, snap_metric, snap_num

# ── prep ────────────────────────────────────────────────────────────


def test_infer_windows(spec):
    w = infer_windows(spec)
    assert w.pre_days == 28
    assert w.post_days == 28
    assert w.pre_end == w.post_start  # sem gap


def test_snapshot_schema(spec):
    snap = build_snapshot(spec)
    cols = set(snap.columns)
    assert {"id", "treatment"} <= cols  # grupo padronizado para "treatment"
    assert {
        snap_metric("ticket", "pre"),
        snap_metric("ticket", "post"),
        snap_delta("ticket"),
        snap_num("ticket", "post"),
        snap_den("ticket", "post"),
        snap_num("ticket", "pre"),
        snap_den("ticket", "pre"),
    } <= cols
    assert {"cov_gmv", "cov_orders"} <= cols
    assert snap.count() == 60


def test_panel_periods(spec):
    panel = build_panel(spec)
    periods = {r["period"] for r in panel.select("period").distinct().collect()}
    assert {"pre", "post"} <= periods


def test_multi_metric_single_pass(spec):
    from dataclasses import replace

    from causal import MetricSpec

    multi = replace(
        spec,
        metric=[
            MetricSpec("gmv", "orders", name="ticket"),
            MetricSpec("gmv", name="gmv_total", agg="sum"),
        ],
    )
    snap = build_snapshot(multi)
    cols = set(snap.columns)
    # colunas das duas métricas presentes no mesmo snapshot
    assert {snap_metric("ticket", "post"), snap_metric("gmv_total", "post")} <= cols
    assert snap.count() == 60


# ── quality ─────────────────────────────────────────────────────────


def test_check_balance(spec):
    pytest.importorskip("scipy")
    from causal.quality import check_balance

    results = check_balance(build_snapshot(spec), metric=spec.primary_metric)
    assert results
    assert all(hasattr(r, "passed") for r in results)


def test_check_parallel_trends(spec):
    pytest.importorskip("scipy")
    from causal.quality import check_parallel_trends

    results = check_parallel_trends(build_panel(spec), "ticket")
    assert results
    # sem efeito no pré, tendências devem ser paralelas
    assert results[0].passed


def test_check_normality(spec):
    from causal.quality import check_normality

    results = check_normality(
        build_snapshot(spec), snap_metric("ticket", "pre"), group_col="treatment"
    )
    assert len(results) == 2  # control e variant


# ── methods ─────────────────────────────────────────────────────────


def test_run_ab_recovers_effect(spec):
    pytest.importorskip("scipy")
    from causal.methods import run_ab

    results = run_ab(build_snapshot(spec), spec.metric)
    assert len(results) == 1
    eff = results[0]
    # efeito plantado ≈ +15% no ticket
    assert 0.08 < eff.relative < 0.22
    assert eff.significant is True


def test_run_bayes(spec):
    pytest.importorskip("numpy")
    from causal.methods import run_bayes_ab

    results = run_bayes_ab(build_snapshot(spec), spec.metric)
    assert len(results) == 1
    assert results[0].extra["prob_better"] > 0.9  # variante claramente melhor


def test_run_did(spec):
    pytest.importorskip("statsmodels")
    from causal.methods import run_did

    results = run_did(build_snapshot(spec), spec.metric)
    assert len(results) == 1
    assert results[0].effect > 0


def test_run_uplift(spec):
    pytest.importorskip("sklearn")
    from causal.methods import run_uplift

    result = run_uplift(build_snapshot(spec), spec.metric)
    assert result.effect == result.effect  # não é NaN
    assert len(result.extra["uplift_by_decile"]) == 10
