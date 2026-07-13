# O Pipeline do Projeto (`src/main.py`)

Este é o documento central da documentação: descreve **todo o fluxo do pipeline**, na ordem exata em que ele executa, referenciando os arquivos responsáveis por cada etapa e os trechos de código mais importantes.

## Visão geral

O arquivo [`src/main.py`](../src/main.py) é o **orquestrador** do projeto: ele não contém lógica de negócio própria — apenas chama, na ordem correta, as funções dos módulos especializados. Rodar o pipeline completo é um único comando, a partir da raiz do projeto:

```bash
python src/main.py
```

O fluxo completo, em ordem:

```
1. Carregamento          (src/data/load.py)
        ↓
2. Tratamento            (src/data/preprocess.py)
        ↓
3. Treino supervisionado (src/models/train.py)
        ↓
4. Seleção de atributos  (src/features/feature_selection.py)
        ↓
5. Clustering            (src/features/clustering.py)
        ↓
6. PCA                   (src/features/dimensionality_reduction.py)
        ↓
7. Autoencoder           (src/features/autoencoder.py)
        ↓
8. Avaliação avançada    (src/models/evaluate.py)
        ↓
9. Explicabilidade SHAP  (src/models/explain.py)
```

Cada etapa **persiste artefatos** (modelos `.pkl`/`.pt`, gráficos `.png`) que serão consumidos depois pela API e pelo relatório — o pipeline treina, a API só carrega.

O esqueleto do orquestrador:

```python
# src/main.py
def main():
    df = carregar_dados(DATA_RAW)          # etapa 1
    df = tratar_dados(df)                  # etapa 2
    df.to_csv(DATA_PROCESSED, index=False)

    modelos = treinar_todos(df)            # etapa 3
    resultado_selecao = selecionar_features(df)   # etapa 4
    df, resumo_clusters = treinar_clustering(df)  # etapa 5
    df, resumo_variancia = treinar_pca(df)        # etapa 6
    df, autoencoder, threshold = treinar_autoencoder(df)  # etapa 7

    metricas = avaliacao_completa(modelos["xgb"], x_test, y_test)  # etapa 8
    explainer, _ = gerar_shap_summary(modelos["xgb"], x_test, feature_names)  # etapa 9
```

---

## Configuração central (`src/config.py`)

Antes das etapas: **nenhum caminho ou parâmetro é escrito "solto" nos módulos**. Tudo vem de [`src/config.py`](../src/config.py):

```python
# src/config.py
RANDOM_STATE = 42        # seed única para todas as etapas estocásticas
TEST_SIZE = 0.2          # proporção do conjunto de teste

DATA_RAW = os.path.join(BASE_DIR, "data", "raw", "cs-training.csv")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "dataset_tratado.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
```

Isso garante **reprodutibilidade** (mesma seed em split, SMOTE, clustering e autoencoder) e facilita manutenção (mudar um caminho ou parâmetro em um único lugar).

---

## Etapa 1 — Carregamento (`src/data/load.py`)

O dataset bruto (`cs-training.csv`, 150.000 linhas) é carregado e as colunas são renomeadas do inglês para o português:

```python
# src/data/load.py
def carregar_dados(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho, index_col=0)
    df.columns = COLUNAS_PT   # definidas em config.py
    return df
```

As 11 colunas resultantes: `inadimplente` (alvo), `utilizacao_credito_rotativo`, `idade`, `atrasos_30_59_dias`, `debt_ratio`, `renda_mensal`, `linhas_credito_abertas`, `atrasos_90_dias`, `emprestimos_imobiliarios`, `atrasos_60_89_dias`, `dependentes`.

---

## Etapa 2 — Tratamento (`src/data/preprocess.py`)

A função `tratar_dados()` executa, em sequência:

**1. Remoção de duplicatas** — `df.drop_duplicates()`.

**2. Missing values** — só duas colunas têm ausências:

```python
df["renda_mensal"] = df["renda_mensal"].fillna(df["renda_mensal"].median())  # mediana: robusta a assimetria
df["dependentes"] = df["dependentes"].fillna(df["dependentes"].mode()[0])    # moda: valor mais frequente
```

**3. Outliers por limite de domínio** — em dados financeiros muito assimétricos, o método IQR puro gera limites irreais; usamos limites baseados em conhecimento do domínio:

```python
df["utilizacao_credito_rotativo"] = df["utilizacao_credito_rotativo"].clip(upper=1, lower=0)  # é percentual
df["debt_ratio"] = df["debt_ratio"].clip(upper=1, lower=0)                                    # é percentual
df["idade"] = df["idade"].clip(upper=150, lower=18)
df["renda_mensal"] = df["renda_mensal"].clip(upper=100_000, lower=limite_inferior_iqr)
```

**4. Remoção de registros corrompidos** — o achado mais importante do projeto. O dataset tem ~269 linhas onde as 3 colunas de atraso carregam valores-sentinela (96/98 — códigos de erro da coleta, não contagens reais) e, simultaneamente, `linhas_credito_abertas = 0` e `emprestimos_imobiliarios = 0` com 100% de uso do rotativo — combinação logicamente impossível. Essas linhas são **removidas** (não "clipadas"):

```python
mask_sentinela = (
    (df["atrasos_30_59_dias"] >= 96)
    | (df["atrasos_60_89_dias"] >= 96)
    | (df["atrasos_90_dias"] >= 96)
)
df = df.loc[~mask_sentinela].copy()
```

> A história de como esse problema foi descoberto (via PCA) está em [modelagem-nao-supervisionada.md](modelagem-nao-supervisionada.md#como-o-pca-revelou-dados-corrompidos).

Ao final, o dataset tratado é salvo em `data/processed/dataset_tratado.csv`.

---

## Etapa 3 — Treino supervisionado (`src/models/train.py`)

### Preparação anti-vazamento

A função `preparar_dados()` implementa a ordem correta das operações — o ponto metodológico mais importante do treino:

```python
# src/models/train.py
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)  # fit SÓ no treino
x_test_scaled = scaler.transform(x_test)        # teste só é transformado

smote = SMOTE(random_state=RANDOM_STATE)
x_train_bal, y_train_bal = smote.fit_resample(x_train_scaled, y_train)  # SMOTE SÓ no treino
```

Por que essa ordem importa:
- **`stratify=y`** mantém a proporção de inadimplentes (~6,6%) igual em treino e teste;
- **normalização ajustada só no treino**: se o scaler "visse" o teste, estatísticas do teste vazariam para o treino (data leakage);
- **SMOTE só no treino**: o balanceamento cria exemplos sintéticos da classe minoritária — se aplicado antes do split, exemplos sintéticos derivados do teste contaminariam o treino.

### Os 4 modelos

`treinar_todos()` treina e avalia, em sequência: **Regressão Logística** (baseline), **Random Forest** (bagging), **XGBoost** (boosting) e **MLP** (rede neural). O XGBoost é o campeão (AUC-ROC 0,850) e é persistido:

```python
joblib.dump(modelo_xgb, f"{MODELS_DIR}/modelo_xgb_campeao.pkl")
joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
joblib.dump(colunas, f"{MODELS_DIR}/colunas_features.pkl")
```

---

## Etapa 4 — Seleção automática de atributos (`src/features/feature_selection.py`)

Experimento de Feature Engineering Avançado: `SelectFromModel` mantém apenas as features com importância (no XGBoost) acima da mediana, e um modelo reduzido é retreinado para comparação:

```python
selector = SelectFromModel(modelo_completo, threshold="median", prefit=True)
x_train_sel = selector.transform(x_train)
```

**Resultado:** 10 → 5 features, mas AUC caiu de 0,8497 para 0,8261 (-0,024). **Decisão: o modelo de produção mantém as 10 features** — a seleção fica documentada como experimento. Detalhe curioso: `debt_ratio` foi descartada pela seleção, convergindo com o que o SHAP já apontava (feature de menor impacto).

---

## Etapas 5, 6 e 7 — Trilha Não Supervisionada

As três técnicas da trilha escolhida no NAP2. Documentação completa em [modelagem-nao-supervisionada.md](modelagem-nao-supervisionada.md); resumo do papel de cada uma no pipeline:

### Etapa 5 — Clustering hierárquico (`src/features/clustering.py`)

Descobre 4 perfis de risco. Como o algoritmo é O(n²) em memória, roda numa **amostra de 5.000 clientes**; um **KNN surrogate** aprende o mapeamento features → cluster e aplica a todo o dataset (e, depois, a clientes novos na API):

```python
Z = linkage(x_amostra_scaled, method="ward")
labels_amostra = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_amostra_scaled, labels_amostra)
df["cluster_id"] = knn.predict(x_full_scaled)   # aplica ao dataset inteiro
```

Os clusters são nomeados (`baixo risco` → `risco muito alto`) **depois** do agrupamento, pela taxa real de inadimplência de cada grupo.

### Etapa 6 — PCA (`src/features/dimensionality_reduction.py`)

Comprime as 10 variáveis comportamentais em 2 componentes principais (36,1% da variância preservada) e gera o gráfico `reports/pca_scatter.png` — os clientes no plano 2D, coloridos por perfil de risco.

### Etapa 7 — Autoencoder (`src/features/autoencoder.py`)

Rede neural (PyTorch) treinada **apenas com adimplentes** — aprende o padrão "normal". O erro de reconstrução vira o **score de anomalia**:

```python
mask_adimplente = (df["inadimplente"] == 0).values
x_treino = x_scaled_full[mask_adimplente]     # treino só com bons pagadores
...
erro_reconstrucao = torch.mean((x_full_tensor - reconstrucao_full) ** 2, dim=1)
df["score_anomalia"] = erro_reconstrucao.numpy()
```

Threshold: percentil 95 do erro entre os adimplentes. Resultado: a taxa de inadimplência dentro das anomalias (18,8%) é ~3x a geral (6,6%).

---

## Etapa 8 — Avaliação avançada (`src/models/evaluate.py`)

Gera as **curvas ROC e Precision-Recall** (salvas em `reports/`) e roda a **análise de custos**: varre thresholds de decisão de 0,05 a 0,95 assumindo que um falso negativo (aprovar quem não paga) custa 5x mais que um falso positivo (negar quem pagaria):

```python
custo_total = fn * CUSTO_FALSO_NEGATIVO + fp * CUSTO_FALSO_POSITIVO  # FN=5, FP=1
melhor = min(resultados, key=lambda r: r["custo_total"])
joblib.dump(melhor_threshold["threshold"], f"{MODELS_DIR}/threshold_decisao.pkl")
```

**O threshold ótimo encontrado (0,25) é persistido e é o que a API usa em produção** — não o 0,5 padrão. Detalhes em [avaliacao-e-explicabilidade.md](avaliacao-e-explicabilidade.md).

---

## Etapa 9 — Explicabilidade (`src/models/explain.py`)

O SHAP (`TreeExplainer`) gera a explicação **global** (summary plot: quais features mais pesam nas decisões do modelo, salvo em `reports/shap_summary.png`) e a **local** (por cliente — a mesma usada pela API para explicar cada análise individual):

```python
explainer = shap.TreeExplainer(modelo)
shap_values = explainer.shap_values(x_test)
joblib.dump(explainer, f"{MODELS_DIR}/shap_explainer.pkl")  # a API carrega este artefato
```

---

## Artefatos produzidos pelo pipeline

Ao final da execução, a pasta `models/` contém tudo o que a API precisa (ela **não treina nada** — só carrega):

| Artefato | Produzido pela etapa | Consumido para |
|---|---|---|
| `modelo_xgb_campeao.pkl` | 3 (treino) | Probabilidade de inadimplência |
| `scaler.pkl`, `colunas_features.pkl` | 3 (treino) | Normalizar a entrada do modelo |
| `threshold_decisao.pkl` | 8 (avaliação) | Corte adimplente/inadimplente |
| `scaler_clustering.pkl`, `knn_surrogate_cluster.pkl`, `mapa_perfis_risco.pkl`, `taxa_inadimplencia_por_perfil.pkl` | 5 (clustering) | Perfil de risco + contexto |
| `scaler_autoencoder.pkl`, `autoencoder.pt`, `threshold_anomalia.pkl`, `autoencoder_input_dim.pkl` | 7 (autoencoder) | Score de anomalia |
| `shap_explainer.pkl` | 9 (SHAP) | Explicação da decisão |

E a pasta `reports/` contém os gráficos: `curva_roc.png`, `curva_precision_recall.png`, `shap_summary.png`, `pca_scatter.png`.

---

## Fora do pipeline principal

Dois fluxos rodam separadamente do `main.py`, por design:

- **Monitoramento** (`src/monitoring/executar_monitoramento.py`): em produção rodaria periodicamente, não a cada treino — ver [monitoramento.md](monitoramento.md);
- **API** (`api/main.py`): serviço de longa duração que carrega os artefatos na subida — ver [api.md](api.md).
