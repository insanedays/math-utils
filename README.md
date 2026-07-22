# math-utils

Utilitários de análise de dados em **PySpark**, com foco em **inferência causal**:
preparação de dados, validações de qualidade (balanceamento, tendências
paralelas, normalidade) e estimadores de efeito (AB clássico, bayesiano,
Diff-in-Diff, uplift).

## Instalação

Instale direto do repositório:

```bash
pip install "git+https://github.com/insanedays/math-utils.git"
```

A base depende apenas de `pyspark`. Os recursos estatísticos e de modelagem
ficam em **extras opcionais** — instale os que for usar:

```bash
# tudo que precisa para rodar as validações e os métodos:
pip install "math-utils[stats,viz,models,ml] @ git+https://github.com/insanedays/math-utils.git"
```

| Extra | Instala | Necessário para |
|-------|---------|-----------------|
| `stats` | scipy, numpy | testes (t, qui², delta method), balanceamento, paralelismo |
| `viz` | matplotlib, pandas, numpy | gráficos (`plot_*`) |
| `models` | statsmodels, pandas | Diff-in-Diff (`run_did`) |
| `ml` | scikit-learn, pandas, numpy | uplift (`run_uplift`) |
| `dev` | pytest + todos acima | rodar os testes |

> Os imports são **lazy**: o pacote carrega sem nenhum extra; a dependência só
> é exigida quando você chama a função que a usa.

## Estrutura do repositório

```
math-utils/
├── causal/                  # domínio de inferência causal
│   ├── spec.py              # contrato: CausalSpec, MetricSpec, Windows
│   ├── prep.py              # build_panel / build_snapshot / infer_windows
│   ├── naming.py            # nomes das colunas geradas (prefixadas por métrica)
│   ├── quality/             # validações de qualidade (rodar ANTES do método)
│   │   ├── balance.py       # check_balance (SMD)
│   │   ├── trends.py        # check_parallel_trends + gráfico
│   │   ├── distribution.py  # check_normality + histograma
│   │   ├── stat_tests.py    # primitivas puras (scipy)
│   │   ├── result.py        # CheckResult (saída padronizada)
│   │   └── exploration.py   # EDA de entrada
│   └── methods/             # estimadores de efeito
│       ├── ab.py            # run_ab (frequentista)
│       ├── bayes.py         # run_bayes_ab (bayesiano)
│       ├── did.py           # run_did + did_event_study
│       ├── uplift.py        # run_uplift (T-learner / CATE)
│       └── result.py        # EffectResult (saída padronizada)
├── tests/                   # mock + testes de fumaça (pytest)
└── pyproject.toml
```

## Uso rápido

```python
from causal import CausalSpec, MetricSpec, build_snapshot, build_panel
from causal.quality import check_balance, check_parallel_trends
from causal.methods import run_ab

spec = CausalSpec(
    df=eventos, id_col="restaurant_id", date_col="dt",
    metric=MetricSpec("gmv", "orders", name="ticket"),
    test_start_date="2026-06-01", treatment_col="group", covariates=["gmv"],
)

# 1. validar pressupostos
check_balance(build_snapshot(spec), metric=spec.primary_metric)
check_parallel_trends(build_panel(spec), "ticket")

# 2. estimar o efeito
run_ab(build_snapshot(spec), spec.primary_metric)
```

## Documentação por domínio

- **[Inferência causal](causal/README.md)** — DataFrame de entrada esperado,
  cadastro de configurações (`CausalSpec`/`MetricSpec`), janelas, o fluxo
  completo e a referência de saídas.

## Testes

Requer um ambiente com PySpark (roda em modo local, sem cluster):

```bash
pip install -e ".[dev]"
pytest
```
