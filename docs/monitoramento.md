# Monitoramento de Drift

Como o projeto detecta se os dados (ou as previsões do modelo) mudaram de comportamento ao longo do tempo.

**Arquivos:** [`src/monitoring/drift.py`](../src/monitoring/drift.py) (cálculo) e [`src/monitoring/executar_monitoramento.py`](../src/monitoring/executar_monitoramento.py) (script executável)

---

## Por que monitorar

Um modelo é treinado com uma "fotografia" dos dados de uma época. Se o mundo muda — crise econômica, mudança no perfil dos solicitantes, nova política de crédito — os dados de produção se afastam dos dados de treino (*drift*), e a performance do modelo degrada silenciosamente. O monitoramento existe para detectar isso **antes** do prejuízo.

## A métrica: PSI (Population Stability Index)

O PSI é o indicador padrão da indústria de crédito para comparar a distribuição de uma variável entre dois períodos. A ideia: dividir a faixa de valores em buckets (decis do período de referência), e comparar a proporção de registros em cada bucket entre referência e atual.

```python
# src/monitoring/drift.py
limites = np.unique(np.percentile(referencia, np.linspace(0, 100, buckets + 1)))
freq_referencia, _ = np.histogram(referencia, bins=limites)
freq_atual, _ = np.histogram(atual, bins=limites)

prop_ref = np.clip(freq_referencia / len(referencia), 1e-4, None)
prop_atu = np.clip(freq_atual / len(atual), 1e-4, None)

psi = np.sum((prop_atu - prop_ref) * np.log(prop_atu / prop_ref))
```

**Convenção de leitura:**

| PSI | Interpretação |
|---|---|
| < 0,10 | Estável — sem mudança relevante |
| 0,10 – 0,25 | Mudança moderada — atenção |
| ≥ 0,25 | Mudança significativa — investigar / considerar retreino |

## O que o módulo monitora

**1. Drift de features** — o PSI de cada uma das 10 variáveis comportamentais, comparando referência vs. lote atual (`relatorio_drift_features`).

**2. Drift das previsões** — o PSI da distribuição das **probabilidades previstas** pelo modelo (`monitorar_distribuicao_predicoes`). Esse segundo monitoramento é especialmente importante em crédito, por um motivo prático: **o rótulo real demora**. Só se sabe se o cliente pagou ou não meses após a decisão — então métricas como AUC não podem ser recalculadas em tempo real. A distribuição das previsões funciona como *proxy*: se o modelo passa a prever probabilidades muito diferentes do padrão histórico, é sinal de alerta mesmo sem os rótulos.

```python
# executar_monitoramento.py — trecho central
prob_referencia = modelo.predict_proba(x_referencia)[:, 1]
prob_atual = modelo.predict_proba(x_atual)[:, 1]
resultado = monitorar_distribuicao_predicoes(prob_referencia, prob_atual)
```

## Execução

O monitoramento roda **separado** do pipeline principal (em produção, seria agendado — ex: mensal):

```bash
python src/monitoring/executar_monitoramento.py
```

Saída típica:

```
=== Monitoramento de Drift: Features ===
                    feature    psi  status
utilizacao_credito_rotativo 0.0008 estável
               renda_mensal 0.0004 estável
...
=== Monitoramento de Drift: Previsões do Modelo ===
PSI das previsões: 0.0011 (estável)
```

## Limitação importante (transparência metodológica)

O dataset Give Me Some Credit **não possui coluna temporal** — as linhas não têm data. Por isso, o script demonstra o mecanismo dividindo o próprio dataset por posição: primeiras 80% das linhas como "referência", últimas 20% como "lote atual".

Consequência honesta dessa escolha:

- Os PSIs próximos de zero obtidos **confirmam que o mecanismo de cálculo funciona** (duas fatias da mesma população homogênea são, de fato, estatisticamente idênticas);
- Eles **não comprovam ausência de drift em produção** — essa validação exigiria dados reais coletados ao longo do tempo.

Em um deploy real, a única mudança necessária seria substituir a fatia simulada pelos dados reais dos clientes analisados no período (por exemplo, registrados pela API a cada requisição) — o cálculo e os thresholds permanecem os mesmos.
