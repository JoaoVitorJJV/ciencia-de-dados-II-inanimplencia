# API REST (FastAPI)

A camada de deploy do projeto: expõe todo o sistema de risco através de um único endpoint HTTP.

**Arquivos:** [`api/main.py`](../api/main.py) (aplicação) e [`api/schemas.py`](../api/schemas.py) (contratos de entrada/saída)

---

## Papel da API

A API **não treina nada** — ela é a ponte entre os artefatos que o pipeline produziu (`models/`) e qualquer sistema consumidor (no nosso caso, o front-end React). Um cliente entra, quatro leituras saem:

```
POST /prever
   entrada: dados de 1 cliente (JSON)
   saída:   probabilidade + classificação
            perfil de risco + taxa histórica
            score de anomalia + flag
            explicação SHAP (fator a fator)
```

## Carregamento dos artefatos

Todos os modelos são carregados **uma única vez, na subida da API** — nunca por requisição (isso manteria a latência baixa mesmo sob carga):

```python
# api/main.py — executa uma vez, no boot
modelo_xgb = joblib.load(f"{MODELS_DIR}/modelo_xgb_campeao.pkl")
scaler_modelo = joblib.load(f"{MODELS_DIR}/scaler.pkl")
threshold_decisao = joblib.load(f"{MODELS_DIR}/threshold_decisao.pkl")
knn_cluster = joblib.load(f"{MODELS_DIR}/knn_surrogate_cluster.pkl")
autoencoder.load_state_dict(torch.load(f"{MODELS_DIR}/autoencoder.pt"))
shap_explainer = joblib.load(f"{MODELS_DIR}/shap_explainer.pkl")
# ... e os demais scalers/mapas auxiliares
```

> Se a pasta `models/` estiver vazia, a API falha na subida — rode `python src/main.py` antes.

## Validação de entrada (Pydantic)

O schema `ClienteInput` valida tipos e faixas **antes** de qualquer código de negócio rodar — requisições inválidas são rejeitadas com erro 422 automático:

```python
# api/schemas.py
class ClienteInput(BaseModel):
    utilizacao_credito_rotativo: float = Field(..., ge=0, le=1)  # fração 0-1
    idade: int = Field(..., ge=18, le=150)
    debt_ratio: float = Field(..., ge=0, le=1)                   # fração 0-1
    renda_mensal: float = Field(..., ge=0)
    # ... demais campos, todos com limites
```

**Atenção à escala:** `utilizacao_credito_rotativo` e `debt_ratio` entram como **frações de 0 a 1** (0,35 = 35%). A conversão de percentual "humano" (0-100) para essa escala é responsabilidade do consumidor — o front-end faz isso automaticamente ([frontend.md](frontend.md#conversão-de-escalas)).

## O fluxo do endpoint

Dentro de `POST /prever`, as quatro leituras em sequência:

```python
# 1. Probabilidade (XGBoost) + classificação pelo threshold de custo
x_modelo_scaled = scaler_modelo.transform(df_cliente[colunas_modelo])
probabilidade = float(modelo_xgb.predict_proba(x_modelo_scaled)[0, 1])
classificacao = "inadimplente" if probabilidade >= threshold_decisao else "adimplente"

# 2. Perfil de risco (KNN surrogate do clustering) + contexto histórico
cluster_id = int(knn_cluster.predict(x_cluster_scaled)[0])
perfil_risco = mapa_perfis[cluster_id]
taxa_historica = float(taxa_por_perfil[perfil_risco])

# 3. Score de anomalia (autoencoder)
erro = torch.mean((x_auto_tensor - reconstrucao) ** 2, dim=1).item()
anomalia_detectada = erro > threshold_anomalia

# 4. Explicação SHAP, ordenada por magnitude
shap_values = shap_explainer.shap_values(x_modelo_scaled)
```

Note que a classificação usa o **threshold de custo (0,25)** persistido pela análise de custos — não o 0,5 padrão. Detalhes em [avaliacao-e-explicabilidade.md](avaliacao-e-explicabilidade.md#análise-de-custos-e-o-threshold-de-decisão).

## Exemplo de requisição e resposta

**Requisição:**

```json
POST /prever
{
  "utilizacao_credito_rotativo": 0.3,
  "idade": 45,
  "atrasos_30_59_dias": 0,
  "debt_ratio": 0.35,
  "renda_mensal": 5000.0,
  "linhas_credito_abertas": 8,
  "atrasos_90_dias": 0,
  "emprestimos_imobiliarios": 1,
  "atrasos_60_89_dias": 0,
  "dependentes": 1
}
```

**Resposta (resumida):**

```json
{
  "probabilidade_inadimplencia": 0.0545,
  "classificacao": "adimplente",
  "perfil_risco": "risco moderado",
  "taxa_inadimplencia_historica_perfil": 0.0461,
  "score_anomalia": 0.0631,
  "anomalia_detectada": false,
  "explicacao": [
    { "feature": "linhas_credito_abertas", "valor_shap": -0.854, "direcao": "reduz" },
    { "feature": "emprestimos_imobiliarios", "valor_shap": -0.632, "direcao": "reduz" }
  ]
}
```

## CORS

Para o front-end (que roda em outra porta) poder chamar a API a partir do navegador:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # em produção, restringir ao domínio real
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Execução

A partir da **raiz do projeto**:

```bash
python -m uvicorn main:app --app-dir api
```

> O parâmetro `--app-dir api` instrui o próprio uvicorn a resolver os módulos a partir da pasta `api/` — a forma que evita erros de `ModuleNotFoundError` no Windows. Para desenvolvimento, adicione `--reload`.

Documentação interativa (Swagger) gerada automaticamente: **http://127.0.0.1:8000/docs** — permite testar o endpoint direto do navegador, com exemplo pré-preenchido.
