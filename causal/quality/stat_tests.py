"""Primitivas estatísticas puras (sem Spark).

Operam sobre números já agregados (médias, desvios, contagens) que são
calculados no Spark e coletados para o driver. Mantê-las puras deixa a camada
testável sem sessão Spark e reaproveitável por várias validações.

Requer o extra opcional ``stats`` (``pip install -e ".[stats]"``).
"""

from __future__ import annotations

import math


def smd(mean_t: float, mean_c: float, std_t: float, std_c: float) -> float:
    """Standardized Mean Difference entre tratamento e controle.

    Usa o desvio-padrão combinado (pooled) como denominador. É a métrica padrão
    de balanceamento de covariáveis; convenção usual de aprovação: ``|SMD| < 0.1``.

    Args:
        mean_t: Média no grupo tratado.
        mean_c: Média no grupo controle.
        std_t: Desvio-padrão no grupo tratado.
        std_c: Desvio-padrão no grupo controle.

    Returns:
        SMD (float). Retorna ``0.0`` quando o desvio combinado é nulo.
    """
    pooled = math.sqrt(((std_t or 0.0) ** 2 + (std_c or 0.0) ** 2) / 2)
    if pooled == 0:
        return 0.0
    return (mean_t - mean_c) / pooled


def welch_ttest(
    mean_t: float,
    mean_c: float,
    std_t: float,
    std_c: float,
    n_t: int,
    n_c: int,
) -> tuple[float, float]:
    """Teste t de Welch (variâncias desiguais) a partir de estatísticas.

    Args:
        mean_t: Média no grupo tratado.
        mean_c: Média no grupo controle.
        std_t: Desvio-padrão amostral do grupo tratado.
        std_c: Desvio-padrão amostral do grupo controle.
        n_t: Tamanho do grupo tratado.
        n_c: Tamanho do grupo controle.

    Returns:
        Tupla ``(t_stat, p_value)`` bicaudal.
    """
    from scipy import stats

    return stats.ttest_ind_from_stats(
        mean1=mean_t, std1=std_t, nobs1=n_t,
        mean2=mean_c, std2=std_c, nobs2=n_c,
        equal_var=False,
    )


def chi2_srm(
    observed: dict[str, int],
    expected_ratios: dict[str, float] | None = None,
) -> tuple[float, float]:
    """Teste qui-quadrado de aderência para Sample Ratio Mismatch (SRM).

    Compara as contagens observadas por grupo com as esperadas segundo os
    ratios do desenho do teste (default: split uniforme).

    Args:
        observed: Contagem observada por grupo, ex: ``{"control": 5010,
            "variant": 4990}``.
        expected_ratios: Proporção esperada por grupo (soma 1). ``None`` assume
            distribuição uniforme entre os grupos.

    Returns:
        Tupla ``(chi2_stat, p_value)``. Convenção de aprovação: ``p > 0.01``.
    """
    from scipy import stats

    groups = list(observed.keys())
    obs = [observed[g] for g in groups]
    total = sum(obs)

    if expected_ratios is None:
        exp = [total / len(groups)] * len(groups)
    else:
        exp = [expected_ratios[g] * total for g in groups]

    chi2, p = stats.chisquare(f_obs=obs, f_exp=exp)
    return float(chi2), float(p)


def delta_method_ratio(
    mean_num: float,
    mean_den: float,
    var_num: float,
    var_den: float,
    cov: float,
    n: int,
) -> tuple[float, float]:
    """Estimativa e variância de uma métrica de razão via delta method.

    Para ``R = mean(num) / mean(den)`` sobre ``n`` unidades independentes:

        ``Var(R) ≈ (1/n) · (1/mean_den²) · (var_num + R²·var_den − 2R·cov)``.

    Necessário porque, em métricas de razão (ticket médio, conversão por
    sessão), a unidade de aleatorização é o ``id``, não a linha do denominador.

    Args:
        mean_num: Média do numerador por unidade.
        mean_den: Média do denominador por unidade.
        var_num: Variância amostral do numerador.
        var_den: Variância amostral do denominador.
        cov: Covariância amostral entre numerador e denominador.
        n: Número de unidades.

    Returns:
        Tupla ``(R, var_R)`` com a razão estimada e sua variância.
    """
    if mean_den == 0 or n == 0:
        return float("nan"), float("nan")
    r = mean_num / mean_den
    var_r = (var_num + r**2 * var_den - 2 * r * cov) / (mean_den**2 * n)
    return r, var_r


def two_sample_z(
    est_t: float,
    est_c: float,
    var_t: float,
    var_c: float,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """Teste z de duas amostras para a diferença de dois estimadores.

    Útil quando cada grupo já tem estimativa e variância (ex: razões via
    :func:`delta_method_ratio`).

    Args:
        est_t: Estimativa no grupo tratado.
        est_c: Estimativa no grupo controle.
        var_t: Variância da estimativa tratada.
        var_c: Variância da estimativa controle.
        alpha: Nível de significância para o intervalo.

    Returns:
        Tupla ``(diff, p_value, ci_low, ci_high)``.
    """
    from scipy import stats

    diff = est_t - est_c
    se = math.sqrt((var_t or 0.0) + (var_c or 0.0))
    if se == 0:
        return diff, float("nan"), diff, diff
    z = diff / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    crit = stats.norm.ppf(1 - alpha / 2)
    return diff, float(p), diff - crit * se, diff + crit * se
