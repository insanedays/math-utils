"""Nomenclatura das colunas geradas pelo ``prep``.

Com suporte a múltiplas métricas, cada métrica tem suas próprias colunas
prefixadas pelo nome (``<metrica>__...``), evitando colisão quando várias são
calculadas na mesma passada. Centralizar aqui garante que ``prep``, ``quality``
e ``methods`` usem exatamente os mesmos nomes.

Convenção (métrica chamada ``ticket``):

- painel:   ``ticket__num``, ``ticket__den``, ``ticket__metric``
- snapshot: ``ticket__pre_num``, ``ticket__pre_den``, ``ticket__pre_metric``,
            ``ticket__post_num``, ``ticket__post_den``, ``ticket__post_metric``,
            ``ticket__delta``
"""

from __future__ import annotations

SEP = "__"


def panel_num(name: str) -> str:
    """Coluna do numerador agregado por dia no painel."""
    return f"{name}{SEP}num"


def panel_den(name: str) -> str:
    """Coluna do denominador agregado por dia no painel."""
    return f"{name}{SEP}den"


def panel_metric(name: str) -> str:
    """Coluna do valor da métrica por dia no painel."""
    return f"{name}{SEP}metric"


def snap_num(name: str, period: str) -> str:
    """Coluna do numerador somado no período (``pre``/``post``) no snapshot."""
    return f"{name}{SEP}{period}_num"


def snap_den(name: str, period: str) -> str:
    """Coluna do denominador somado no período no snapshot."""
    return f"{name}{SEP}{period}_den"


def snap_metric(name: str, period: str) -> str:
    """Coluna do valor da métrica no período no snapshot."""
    return f"{name}{SEP}{period}_metric"


def snap_delta(name: str) -> str:
    """Coluna da variação pós − pré da métrica no snapshot."""
    return f"{name}{SEP}delta"


def cov(name: str) -> str:
    """Coluna de uma covariável agregada na janela pré."""
    return f"cov_{name}"
