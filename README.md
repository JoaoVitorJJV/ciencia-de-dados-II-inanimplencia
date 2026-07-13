# Predição de Inadimplência de Clientes

Sistema completo de análise de risco de crédito: pipeline de machine learning de ponta a ponta, API REST e interface web para análise de solicitações de cartão de crédito.

Projeto desenvolvido para a disciplina **Ciência de Dados II** (NAP1 + NAP2).

---

## O que o sistema faz

A partir dos dados financeiros de um cliente, o sistema entrega **quatro leituras de risco** em uma única análise:

| Leitura | Técnica | O que responde |
|---|---|---|
| Probabilidade de inadimplência | XGBoost (supervisionado) | "Qual a chance deste cliente não pagar?" |
| Perfil de risco | Clustering hierárquico + KNN | "Com que grupo de clientes ele se parece?" |
| Score de anomalia | Autoencoder (PyTorch) | "O comportamento dele foge do padrão normal?" |
| Explicação da decisão | SHAP | "Quais dados pesaram nessa análise, e em que direção?" |

**Dataset:** [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle) — 150.000 registros, 11 variáveis.

---

## Estrutura do projeto

```
projeto/
├── data/
│   ├── raw/                  # cs-training.csv (dataset bruto, vem da Kaggle)
│   └── processed/            # dataset_tratado.csv (gerado pelo pipeline)
├── src/
│   ├── config.py             # paths, seed e parâmetros centralizados
│   ├── main.py               # ORQUESTRADOR: executa o pipeline completo
│   ├── data/
│   │   ├── load.py           # carregamento + renomeação de colunas
│   │   └── preprocess.py     # limpeza, missing values, outliers
│   ├── features/
│   │   ├── clustering.py     # perfis de risco (clustering hierárquico)
│   │   ├── dimensionality_reduction.py  # PCA
│   │   ├── autoencoder.py    # detecção de anomalias
│   │   └── feature_selection.py         # seleção automática de atributos
│   ├── models/
│   │   ├── train.py          # treino dos modelos supervisionados
│   │   ├── evaluate.py       # ROC, Precision-Recall, análise de custos
│   │   └── explain.py        # explicabilidade (SHAP)
│   └── monitoring/
│       ├── drift.py          # cálculo de PSI
│       └── executar_monitoramento.py    # script de monitoramento
├── api/                      # API REST (FastAPI)
├── frontend/                 # Interface web "Radar de Risco" (React)
├── models/                   # artefatos treinados (.pkl / .pt)
├── reports/                  # gráficos gerados (ROC, PR, SHAP, PCA)
└── docs/                     # documentação detalhada (esta pasta)
```

---

## Início rápido

```bash
# 1. Ambiente
python -m venv venv
venv\Scripts\activate            # Windows | Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
pip install -r api/requirements.txt

# 2. Dataset em data/raw/cs-training.csv, depois rodar o pipeline
python src/main.py

# 3. Subir a API (terminal 1)
python -m uvicorn main:app --app-dir api

# 4. Subir o front-end (terminal 2)
cd frontend && npm install && npm run dev
```

- API: http://127.0.0.1:8000 (documentação interativa em `/docs`)
- Front-end: http://localhost:5173

---

## Documentação detalhada

A pasta [`docs/`](docs/) contém a documentação completa do projeto:

| Documento | Conteúdo |
|---|---|
| [**docs/pipeline.md**](docs/pipeline.md) | **O documento central**: o pipeline completo do `main.py`, etapa por etapa, com referências aos arquivos e trechos de código |
| [docs/modelagem-nao-supervisionada.md](docs/modelagem-nao-supervisionada.md) | Detalhes das três técnicas da trilha não supervisionada: clustering, PCA e autoencoder |
| [docs/avaliacao-e-explicabilidade.md](docs/avaliacao-e-explicabilidade.md) | Curvas ROC/PR, análise de custos, threshold de decisão e SHAP |
| [docs/api.md](docs/api.md) | A API FastAPI: endpoint, artefatos consumidos, contrato de entrada/saída |
| [docs/frontend.md](docs/frontend.md) | O front-end "Radar de Risco": arquitetura, conversão de escalas, tradução para linguagem de negócio |
| [docs/monitoramento.md](docs/monitoramento.md) | Monitoramento de drift com PSI e suas limitações no contexto do projeto |

---

## Resultados principais

| Modelo | AUC-ROC |
|---|---|
| Regressão Logística (baseline) | 0,848 |
| Random Forest | 0,825 |
| **XGBoost (campeão)** | **0,850** |
| MLP | 0,839 |

Perfis de risco descobertos pelo clustering (taxa real de inadimplência por grupo): **baixo risco** 3,2% → **moderado** 4,6% → **alto** 5,3% → **muito alto** 33,3%.

O autoencoder, treinado apenas com bons pagadores, sinaliza anomalias cuja taxa de inadimplência (18,8%) é quase 3x a da base geral (6,6%).

---

## Stack

**Pipeline/ML:** Python, pandas, scikit-learn, XGBoost, imbalanced-learn (SMOTE), SciPy, PyTorch, SHAP, matplotlib
**API:** FastAPI, Pydantic, Uvicorn
**Front-end:** React 18, Vite
**Versionamento:** Git/GitHub
