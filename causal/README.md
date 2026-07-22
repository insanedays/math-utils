# causal — inferência causal em PySpark

Este pacote leva você de um **DataFrame de eventos** até o **efeito causal
estimado**, passando pelas validações que dizem se o método faz sentido.

O fluxo é sempre o mesmo:

```
DataFrame  →  CausalSpec  →  prep  →  quality (valida)  →  methods (estima)
```

---

## 1. DataFrame de entrada esperado

Você fornece **um** DataFrame Spark. Ele pode estar em dois grãos (ambos
funcionam — o `prep` sempre agrega para `id × dia`):

- **event-level**: uma linha por evento (ex: um pedido)
- **`id × data`**: uma linha por unidade por dia (já pré-agregado)

### Colunas necessárias

| Coluna | Obrigatória? | Descrição |
|--------|--------------|-----------|
| **id** | ✅ | Identificador da unidade (restaurante, usuário…). Nome livre → `id_col`. |
| **data** | ✅ | Data do evento/linha. Nome livre → `date_col`. |
| **métrica** | ✅ | Colunas cruas usadas para calcular a métrica (ex: `gmv`, `orders`). Você passa uma **expressão**, não uma coluna pronta. |
| **grupo** | opcional | Atribuição controle/variante. Se existir uma coluna (default `group`), a análise é tratada com grupos; se não, é método sem grupo. |
| **covariáveis** | opcional | Colunas a balancear (medidas só na janela pré). |

Exemplo (event-level):

| restaurant_id | dt | group | gmv | orders |
|---|---|---|---|---|
| 1 | 2026-05-10 | variant | 226.1 | 9 |
| 1 | 2026-06-10 | variant | 315.7 | 11 |
| 2 | 2026-05-10 | control | 201.3 | 8 |

---

## 2. Cadastro de configurações

Toda a análise é descrita por dois objetos.

### `MetricSpec` — como calcular a métrica

```python
from causal import MetricSpec

# métrica simples (soma/média/contagem de uma expressão)
MetricSpec(numerator="gmv", agg="sum", name="gmv_total")

# métrica de razão (numerador / denominador) — trata ticket, conversão etc.
MetricSpec(numerator="gmv", denominator="orders", name="ticket")
```

| Campo | Descrição |
|-------|-----------|
| `numerator` | Expressão SQL do numerador (ex: `"gmv"`, `"case when converteu then 1 else 0 end"`). |
| `denominator` | Expressão do denominador para razão; `None` = métrica simples. |
| `agg` | `sum` / `avg` / `count` (só para métrica simples). |
| `name` | Rótulo **único** — vira prefixo das colunas geradas. |

### `CausalSpec` — a configuração completa

```python
from causal import CausalSpec, MetricSpec

spec = CausalSpec(
    df=eventos,
    id_col="restaurant_id",
    date_col="dt",
    metric=MetricSpec("gmv", "orders", name="ticket"),
    test_start_date="2026-06-01",   # só a data de início já basta
    treatment_col="group",           # default "group"; usado se a coluna existir
    covariates=["gmv", "orders"],    # medidas na janela pré
)
```

| Campo | Default | Descrição |
|-------|---------|-----------|
| `df` | — | DataFrame de entrada. |
| `id_col`, `date_col` | — | Colunas de unidade e data. |
| `metric` | — | Uma `MetricSpec` **ou uma lista** (ver multi-métrica). |
| `test_start_date` | — | Início do teste (`YYYY-MM-DD`). |
| `post_days` | `None` | Tamanho da janela pós; `None` = do início até a última data do dado. |
| `pre_days` | `None` | Tamanho da janela pré; `None` = espelha a pós. |
| `gap_days` | `0` | Washout descartado ao redor do início. |
| `treatment_col` | `"group"` | Coluna de grupo; ausência = sem grupo. |
| `covariates` | `[]` | Covariáveis a balancear (janela pré). |

### Janelas: só a data de início

Basta `test_start_date`. As janelas são inferidas do próprio dado:

- **pós** = do início do teste até a **última data disponível**
- **pré** = a **mesma quantidade de dias** para trás

Precisa controlar manualmente? Passe `pre_days` / `post_days` / `gap_days`.

### Múltiplas métricas (uma passada só)

Passe uma lista em `metric`; todas são calculadas na mesma agregação Spark:

```python
spec = CausalSpec(
    df=eventos, id_col="restaurant_id", date_col="dt",
    metric=[
        MetricSpec("gmv", "orders", name="ticket"),      # primária
        MetricSpec("gmv", name="gmv_total", agg="sum"),
    ],
    test_start_date="2026-06-01", treatment_col="group",
)
# spec.primary_metric  -> ticket
# spec.metric_list     -> [ticket, gmv_total]
```

> ⚠️ Testar várias métricas = múltiplas comparações. Defina **uma primária** e
> trate as outras como secundárias/diagnóstico.

---

## 3. Preparação (`prep`)

```python
from causal import build_panel, build_snapshot

panel = build_panel(spec)      # grão id × dia   → tendências, DiD visual
snap  = build_snapshot(spec)   # grão id         → balanceamento, efeito
```

As colunas geradas são **prefixadas pelo nome da métrica** (ver `naming.py`).
Para a métrica `ticket`:

| DataFrame | Colunas |
|-----------|---------|
| `panel` | `id`, `date`, `period`, `ticket__num`, `ticket__den`, `ticket__metric`, `treatment` |
| `snapshot` | `id`, `ticket__pre_metric`, `ticket__post_metric`, `ticket__delta`, `ticket__{pre,post}_num`, `ticket__{pre,post}_den`, `treatment`, `cov_*` |

> 📌 O `prep` **padroniza** a coluna de grupo (qualquer que seja o
> `treatment_col`, ex: `group`) para o nome fixo **`treatment`** no `panel` e no
> `snapshot`. Por isso os `check_*`/`run_*` usam `treatment_col="treatment"` /
> `group_col="treatment"` por padrão — você não precisa passar nada.

---

## 4. Validações (`quality`) — rodar ANTES do método

```python
from causal.quality import check_balance, check_parallel_trends, check_normality

check_balance(snap, metric=spec.primary_metric)     # SMD < 0.1 por covariável
check_parallel_trends(panel, "ticket")              # p > 0.05 (paralelismo, p/ DiD)
check_normality(snap, "ticket__pre_metric")         # group_col default = "treatment"
```

Cada `check_*` devolve `CheckResult` (`name`, `passed`, `value`, `criterion`,
`extra`). Os gráficos correspondentes: `plot_parallel_trends`,
`plot_distribution`.

---

## 5. Métodos (`methods`)

Todos recebem `snapshot` + a `MetricSpec` e devolvem `EffectResult`
(`effect`, `relative`, `ci_low/high`, `p_value`, `significant`, `extra`).

```python
from causal.methods import run_ab, run_bayes_ab, run_did, run_uplift

m = spec.primary_metric
run_ab(snap, m)                         # frequentista (Welch / delta method)
run_bayes_ab(snap, m, kind="continuous")# bayesiano (prob_better, cred. interval)
run_did(snap, m)                        # Diff-in-Diff (exige paralelismo ok)
run_uplift(snap, m)                     # uplift / CATE por unidade
```

| Método | Quando usar | Requisito principal |
|--------|-------------|---------------------|
| `run_ab` | grupos randomizados | balanceamento (SMD) |
| `run_bayes_ab` | quer probabilidade, não p-valor | idem AB |
| `run_did` | grupos não-random + antes/depois | tendências paralelas |
| `run_uplift` | efeito heterogêneo (para quem funciona) | covariáveis preditivas |

---

## Fluxo completo (exemplo)

```python
from causal import CausalSpec, MetricSpec, build_panel, build_snapshot
from causal.quality import check_balance, check_parallel_trends
from causal.methods import run_did

spec = CausalSpec(
    df=eventos, id_col="restaurant_id", date_col="dt",
    metric=MetricSpec("gmv", "orders", name="ticket"),
    test_start_date="2026-06-01", treatment_col="group", covariates=["gmv"],
)
panel, snap = build_panel(spec), build_snapshot(spec)

# 1. validar
assert all(r.passed for r in check_balance(snap, metric=spec.primary_metric))
assert check_parallel_trends(panel, "ticket")[0].passed

# 2. estimar
for eff in run_did(snap, spec.primary_metric):
    print(eff.to_dict())
```
