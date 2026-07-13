# Modelagem Não Supervisionada

A trilha de modelagem avançada escolhida no NAP2, composta por três técnicas complementares: **clustering hierárquico** (segmentar), **PCA** (visualizar/comprimir) e **autoencoder** (detectar anomalias). Todas usam as mesmas 10 variáveis comportamentais e nenhuma usa o rótulo `inadimplente` durante o treino — ele só entra depois, para interpretação e validação.

---

## 1. Clustering hierárquico — perfis de risco

**Arquivo:** [`src/features/clustering.py`](../src/features/clustering.py)

### O problema de escala e a solução

O clustering hierárquico aglomerativo precisa da matriz de distâncias completa entre todos os pares de pontos — custo **O(n²)** em memória. Para 150.000 clientes seriam ~22 bilhões de pares: inviável.

A solução adotada (padrão na prática da indústria):

```python
# 1. Amostra representativa de 5.000 clientes
amostra = df.sample(n=5000, random_state=RANDOM_STATE)

# 2. Clustering hierárquico de verdade, só na amostra
Z = linkage(x_amostra_scaled, method="ward")     # Ward: minimiza variância intra-cluster
labels_amostra = fcluster(Z, t=4, criterion="maxclust")

# 3. KNN "surrogate": aprende o mapeamento features -> cluster
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_amostra_scaled, labels_amostra)

# 4. O KNN generaliza a atribuição para TODO o dataset
df["cluster_id"] = knn.predict(x_full_scaled)
```

O mesmo KNN é persistido (`knn_surrogate_cluster.pkl`) e usado pela API para atribuir perfil a **clientes novos** — sem nunca precisar re-rodar o algoritmo hierárquico.

### Nomeando os clusters

Os grupos nascem sem significado ("cluster 1, 2, 3, 4"). A interpretação vem **depois**, ordenando-os pela taxa real de inadimplência:

```python
taxa_por_cluster = df.groupby("cluster_id")["inadimplente"].mean().sort_values()
ordem_risco = ["baixo risco", "risco moderado", "risco alto", "risco muito alto"]
mapa_nomes = {cluster_id: ordem_risco[i] for i, cluster_id in enumerate(taxa_por_cluster.index)}
```

Importante: `inadimplente` **não participa do agrupamento** — só da rotulagem posterior. Resultado final:

| Perfil | Clientes | Taxa de inadimplência |
|---|---|---|
| Baixo risco | 29.893 | 3,2% |
| Risco moderado | 78.328 | 4,6% |
| Risco alto | 29.750 | 5,3% |
| Risco muito alto | 11.195 | 33,3% |

A taxa cresce de forma monotônica — evidência de que os grupos capturam risco real.

---

## 2. PCA — redução de dimensionalidade

**Arquivo:** [`src/features/dimensionality_reduction.py`](../src/features/dimensionality_reduction.py)

O PCA comprime as 10 variáveis (padronizadas com `StandardScaler`) em 2 componentes principais:

```python
pca = PCA(n_components=2, random_state=42)
componentes = pca.fit_transform(x_scaled)
df["pca_1"], df["pca_2"] = componentes[:, 0], componentes[:, 1]
```

**Variância preservada: 36,1%** (PC1: 19,3% + PC2: 16,8%) — valor modesto, típico de dados financeiros com relações não-lineares; documentado como limitação. O gráfico gerado (`reports/pca_scatter.png`) plota todos os clientes no plano 2D, coloridos pelo perfil de risco do clustering.

### Como o PCA revelou dados corrompidos

Este foi o achado metodológico mais importante do projeto. Na primeira execução, o gráfico do PCA mostrou o cluster "risco muito alto" (225 clientes) **isolado a uma distância impossível** do resto da população (PC1 ≈ 43, contra -5 a 2 de todo o resto).

A investigação, em etapas:

1. **Hipótese inicial:** valores-sentinela 96/98 nas colunas de atraso (problema documentado do dataset). Confirmado: 269 registros. Primeira correção: `clip(upper=17)`.
2. **A correção falhou:** o cluster isolado persistiu com estatísticas **idênticas** — o clip trocou 96/98 por 17, mas as linhas continuavam extremas e empatadas.
3. **Investigação multivariada:** um escore de extremidade (soma dos z-scores²) por cliente revelou a assinatura completa dos registros: as 3 colunas de atraso no teto **e simultaneamente** 0 linhas de crédito abertas, 0 empréstimos imobiliários e 100% de uso do rotativo — logicamente impossível (não existem 17+ atrasos em linhas de crédito que não existem).
4. **Correção definitiva:** remoção das linhas (~0,2% do dataset) no `preprocess.py`.

Após a correção, o cluster fantasma desapareceu e o perfil "risco muito alto" passou de um artefato de 225 linhas corrompidas para um segmento real de 11.195 clientes.

**Lição:** a técnica de visualização funcionou como **auditoria de qualidade de dados** — expôs um problema invisível às etapas anteriores.

---

## 3. Autoencoder — detecção de anomalias

**Arquivo:** [`src/features/autoencoder.py`](../src/features/autoencoder.py)

### A ideia

Um autoencoder é uma rede neural que aprende a **reconstruir a própria entrada**, passando por um gargalo (bottleneck) que a força a aprender só o essencial. O truque para detecção de anomalias: treinar **apenas com clientes adimplentes**. A rede aprende o que é comportamento financeiro "normal"; quando um cliente atípico passa por ela, a reconstrução sai ruim — e o **erro de reconstrução** vira um score de anomalia.

### Arquitetura e treino

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        self.encoder = nn.Sequential(nn.Linear(input_dim, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, input_dim))
```

- Encoder 10 → 8 → 4, decoder 4 → 8 → 10; loss MSE; Adam (lr=1e-3); 50 épocas
- Implementado em **PyTorch** (compatibilidade com Python 3.13, melhor que TensorFlow na época do projeto)

```python
mask_adimplente = (df["inadimplente"] == 0).values
x_treino = x_scaled_full[mask_adimplente]        # SÓ adimplentes no treino
```

### Score e threshold

O score de anomalia é calculado para **todos** os clientes; o corte é o **percentil 95** do erro entre os adimplentes:

```python
erro = torch.mean((x_full_tensor - reconstrucao_full) ** 2, dim=1)
threshold = np.percentile(df.loc[mask_adimplente, "score_anomalia"], 95)
df["anomalia_detectada"] = (df["score_anomalia"] > threshold).astype(int)
```

### Validação do sinal

| | Taxa de inadimplência |
|---|---|
| Clientes sinalizados como anômalos (5,8% da base) | **18,8%** |
| Base geral | 6,6% |

A taxa entre as anomalias é ~3x a geral — o autoencoder captura um sinal real de risco **sem nunca ter visto o rótulo durante o treino**. Este é o argumento central da trilha não supervisionada, e o motivo do score de anomalia ser exposto pela API como leitura complementar à probabilidade do XGBoost.
