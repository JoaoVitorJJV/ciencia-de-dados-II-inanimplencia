"""
API REST (FastAPI) para o modelo de predição de inadimplência.

Endpoint único POST /prever, que recebe os dados de 1 cliente e devolve:
1. Probabilidade de inadimplência (XGBoost campeão)
2. Perfil de risco (clustering hierárquico + KNN surrogate)
3. Score de anomalia (autoencoder)
4. Explicação da decisão (SHAP)

Execução (a partir da raiz do projeto):
    uvicorn api.main:app --reload
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Permite importar os módulos de src/ e o schemas.py local sem depender
# de como o uvicorn foi invocado (evita problemas de sys.path no Windows)
API_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(API_DIR)
sys.path.insert(0, API_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from config import MODELS_DIR
from features.autoencoder import Autoencoder
from features.clustering import FEATURES_CLUSTERING

from schemas import ClienteInput, ContribuicaoFeature, PrevisaoResponse

app = FastAPI(
    title="API de Predição de Inadimplência",
    description="Previsão de risco de crédito com perfil de risco, "
                 "detecção de anomalia e explicabilidade (SHAP).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Carregamento dos artefatos (uma vez, na subida da API) ---

modelo_xgb = joblib.load(f"{MODELS_DIR}/modelo_xgb_campeao.pkl")
scaler_modelo = joblib.load(f"{MODELS_DIR}/scaler.pkl")
colunas_modelo = joblib.load(f"{MODELS_DIR}/colunas_features.pkl")
threshold_decisao = joblib.load(f"{MODELS_DIR}/threshold_decisao.pkl")

scaler_cluster = joblib.load(f"{MODELS_DIR}/scaler_clustering.pkl")
knn_cluster = joblib.load(f"{MODELS_DIR}/knn_surrogate_cluster.pkl")
mapa_perfis = joblib.load(f"{MODELS_DIR}/mapa_perfis_risco.pkl")
taxa_por_perfil = joblib.load(f"{MODELS_DIR}/taxa_inadimplencia_por_perfil.pkl")

scaler_autoencoder = joblib.load(f"{MODELS_DIR}/scaler_autoencoder.pkl")
threshold_anomalia = joblib.load(f"{MODELS_DIR}/threshold_anomalia.pkl")
input_dim_autoencoder = joblib.load(f"{MODELS_DIR}/autoencoder_input_dim.pkl")
autoencoder = Autoencoder(input_dim=input_dim_autoencoder)
autoencoder.load_state_dict(torch.load(f"{MODELS_DIR}/autoencoder.pt"))
autoencoder.eval()

shap_explainer = joblib.load(f"{MODELS_DIR}/shap_explainer.pkl")


@app.post("/prever", response_model=PrevisaoResponse)
def prever(cliente: ClienteInput):
    dados = cliente.model_dump()
    df_cliente = pd.DataFrame([dados])

    try:
        # 1. Previsão de inadimplência
        x_modelo = df_cliente[colunas_modelo]
        x_modelo_scaled = scaler_modelo.transform(x_modelo)
        probabilidade = float(modelo_xgb.predict_proba(x_modelo_scaled)[0, 1])
        classificacao = (
            "inadimplente" if probabilidade >= threshold_decisao else "adimplente"
        )

        # 2. Perfil de risco
        x_cluster = df_cliente[FEATURES_CLUSTERING]
        x_cluster_scaled = scaler_cluster.transform(x_cluster)
        cluster_id = int(knn_cluster.predict(x_cluster_scaled)[0])
        perfil_risco = mapa_perfis[cluster_id]
        taxa_historica = float(taxa_por_perfil[perfil_risco])

        # 3. Score de anomalia
        x_auto = df_cliente[FEATURES_CLUSTERING]
        x_auto_scaled = scaler_autoencoder.transform(x_auto)
        x_auto_tensor = torch.tensor(x_auto_scaled, dtype=torch.float32)
        with torch.no_grad():
            reconstrucao = autoencoder(x_auto_tensor)
            erro = torch.mean((x_auto_tensor - reconstrucao) ** 2, dim=1).item()
        anomalia_detectada = erro > threshold_anomalia

        # 4. Explicação SHAP (usa os mesmos dados escalados do modelo)
        shap_values = shap_explainer.shap_values(x_modelo_scaled)
        contribuicoes = sorted(
            zip(colunas_modelo, shap_values[0]),
            key=lambda par: abs(par[1]),
            reverse=True,
        )
        explicacao = [
            ContribuicaoFeature(
                feature=feature,
                valor_shap=float(valor),
                direcao="aumenta" if valor > 0 else "reduz",
            )
            for feature, valor in contribuicoes
        ]

        return PrevisaoResponse(
            probabilidade_inadimplencia=probabilidade,
            classificacao=classificacao,
            perfil_risco=perfil_risco,
            taxa_inadimplencia_historica_perfil=taxa_historica,
            score_anomalia=erro,
            anomalia_detectada=anomalia_detectada,
            explicacao=explicacao,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))