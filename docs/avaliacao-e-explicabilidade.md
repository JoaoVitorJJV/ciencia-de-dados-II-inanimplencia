# Avaliação Avançada e Explicabilidade

Como o modelo campeão é avaliado além da acurácia — e como cada decisão dele é explicada.

**Arquivos:** [`src/models/evaluate.py`](../src/models/evaluate.py) e [`src/models/explain.py`](../src/models/explain.py)

---

## Por que acurácia não serve aqui

O dataset é fortemente desbalanceado: só **~6,6% dos clientes são inadimplentes**. Um modelo inútil que respondesse "adimplente" para todo mundo teria ~93% de acurácia. Por isso a avaliação usa métricas que enxergam a classe minoritária: AUC-ROC, curva Precision-Recall e, principalmente, **custo de negócio**.

---

## Curva ROC

Mede a capacidade do modelo de **ordenar** o risco: dado um inadimplente e um adimplente aleatórios, com que frequência o modelo atribui probabilidade maior ao inadimplente?

```python
# src/models/evaluate.py
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = auc(fpr, tpr)
```

**Resultado do XGBoost campeão: AUC = 0,850.** Gráfico em `reports/curva_roc.png`.

Comparativo entre os 4 modelos (dados limpos):

| Modelo | AUC-ROC |
|---|---|
| Regressão Logística (baseline) | 0,848 |
| Random Forest | 0,825 |
| **XGBoost** | **0,850** |
| MLP | 0,839 |

---

## Curva Precision-Recall

Mais informativa que a ROC em bases desbalanceadas: mostra o trade-off entre *precision* (dos que o modelo apontou como risco, quantos eram de fato?) e *recall* (dos inadimplentes reais, quantos o modelo pegou?).

No nosso caso a precision cai rápido conforme o recall sobe — consequência direta dos 6,6% de positivos. Gráfico em `reports/curva_precision_recall.png`. É essa curva que justifica a próxima seção: escolher o ponto de operação por **custo**, não por métrica genérica.

---

## Análise de custos e o threshold de decisão

### A premissa de negócio

Em crédito, os dois tipos de erro têm custos muito diferentes:

- **Falso negativo** (aprovar quem não paga): perde-se o valor emprestado — caro;
- **Falso positivo** (negar quem pagaria): perde-se a receita daquele cliente — custo de oportunidade, bem menor.

Adotamos pesos didáticos: **FN = 5x, FP = 1x** (constantes ajustáveis no topo do `evaluate.py`).

### O método

Varremos thresholds de decisão de 0,05 a 0,95 e calculamos o custo total em cada corte:

```python
for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    custo_total = fn * CUSTO_FALSO_NEGATIVO + fp * CUSTO_FALSO_POSITIVO
melhor = min(resultados, key=lambda r: r["custo_total"])
```

### O resultado — e onde ele é usado

**Threshold ótimo: 0,25** — bem abaixo do 0,5 padrão do `predict()`. Ou seja: dado o custo assimétrico, vale a pena sinalizar como risco qualquer cliente com probabilidade ≥ 25%.

O ponto mais importante: esse valor **é persistido como artefato e usado pela API em produção**:

```python
# evaluate.py — persiste
joblib.dump(melhor_threshold["threshold"], f"{MODELS_DIR}/threshold_decisao.pkl")

# api/main.py — consome
classificacao = "inadimplente" if probabilidade >= threshold_decisao else "adimplente"
```

A análise de custo não ficou teórica: ela define o comportamento real do sistema.

---

## Explicabilidade (SHAP)

**Arquivo:** [`src/models/explain.py`](../src/models/explain.py)

O SHAP atribui a cada feature, em cada previsão, um valor que quantifica **quanto ela empurrou a decisão** para cima (mais risco) ou para baixo (menos risco).

### Explicação global

O summary plot (`reports/shap_summary.png`) mostra o impacto de cada feature em todo o conjunto de teste:

```python
explainer = shap.TreeExplainer(modelo)     # otimizado para modelos de árvore
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values, x_test, feature_names=feature_names)
```

**Principais achados:**
- Features mais influentes: `utilizacao_credito_rotativo`, `linhas_credito_abertas`, e os atrasos (`atrasos_90_dias`, `atrasos_30_59_dias`) — coerente com o senso de negócio;
- `idade` tem relação inversa (mais jovem = mais risco);
- `debt_ratio` é a feature de **menor** impacto — achado que depois convergiu com a seleção automática de atributos (que a descartou), exemplo de SHAP auditando a qualidade das features.

### Explicação local (por cliente)

A mesma técnica explica **uma previsão individual** — e é exatamente o que a API expõe:

```python
# api/main.py
shap_values = shap_explainer.shap_values(x_modelo_scaled)
contribuicoes = sorted(zip(colunas_modelo, shap_values[0]), key=lambda par: abs(par[1]), reverse=True)
```

O front-end traduz essas contribuições em um "demonstrativo de fatores": barras divergentes (reduz risco ← → aumenta risco), com nomes amigáveis. Além da transparência para o analista, isso conecta com a LGPD (Art. 20 — direito à explicação de decisões automatizadas).
