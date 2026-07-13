"""
Autoencoder para detecção de anomalias em clientes.

Estratégia: o autoencoder é treinado apenas com clientes ADIMPLENTES
(inadimplente == 0), aprendendo a reconstruir o padrão de comportamento
"normal". Clientes cujo erro de reconstrução é alto (o modelo "não
reconhece" aquele padrão) recebem um score de anomalia alto - isso
inclui potencialmente inadimplentes e outliers comportamentais, mesmo
sem usar o rótulo diretamente no treino.

Implementado em PyTorch (maior compatibilidade com Python 3.13 do que
TensorFlow, que ainda tem suporte parcial para essa versão do Python).
"""
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from config import MODELS_DIR, RANDOM_STATE
from features.clustering import FEATURES_CLUSTERING as FEATURES_AUTOENCODER

torch.manual_seed(RANDOM_STATE)

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def treinar_autoencoder(df: pd.DataFrame, epochs: int = 50, lr: float = 1e-3):
    """
    Treina o autoencoder apenas com clientes adimplentes e calcula o
    score de anomalia (erro de reconstrução) para todo o dataset.
    """
    x = df[FEATURES_AUTOENCODER]

    scaler = MinMaxScaler()
    x_scaled_full = scaler.fit_transform(x)

    # Treino só com adimplentes (aprende o padrão "normal")
    mask_adimplente = (df["inadimplente"] == 0).values
    x_treino = x_scaled_full[mask_adimplente]

    x_treino_tensor = torch.tensor(x_treino, dtype=torch.float32)
    x_full_tensor = torch.tensor(x_scaled_full, dtype=torch.float32)

    modelo = Autoencoder(input_dim=x_treino.shape[1])
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    criterio = nn.MSELoss()

    modelo.train()
    for epoch in range(epochs):
        otimizador.zero_grad()
        reconstrucao = modelo(x_treino_tensor)
        perda = criterio(reconstrucao, x_treino_tensor)
        perda.backward()
        otimizador.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss (MSE): {perda.item():.6f}")

    # Score de anomalia = erro de reconstrução por cliente, em todo o dataset
    modelo.eval()
    with torch.no_grad():
        reconstrucao_full = modelo(x_full_tensor)
        erro_reconstrucao = torch.mean((x_full_tensor - reconstrucao_full) ** 2, dim=1)

    df = df.copy()
    df["score_anomalia"] = erro_reconstrucao.numpy()

    # Threshold: percentil 95 do erro em clientes adimplentes (referência "normal")
    threshold = np.percentile(
        df.loc[mask_adimplente, "score_anomalia"], 95
    )
    df["anomalia_detectada"] = (df["score_anomalia"] > threshold).astype(int)

    print("\n--- Autoencoder: Detecção de Anomalias ---")
    print(f"Threshold (percentil 95 dos adimplentes): {threshold:.6f}")
    print(f"Anomalias detectadas: {df['anomalia_detectada'].sum()} "
          f"({df['anomalia_detectada'].mean():.2%} do dataset)")

    # Quão bem o score de anomalia se relaciona com a inadimplência real
    taxa_inad_em_anomalias = df.loc[df["anomalia_detectada"] == 1, "inadimplente"].mean()
    taxa_inad_geral = df["inadimplente"].mean()
    print(f"Taxa de inadimplência dentro das anomalias: {taxa_inad_em_anomalias:.2%}")
    print(f"Taxa de inadimplência geral do dataset: {taxa_inad_geral:.2%}\n")

    torch.save(modelo.state_dict(), f"{MODELS_DIR}/autoencoder.pt")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler_autoencoder.pkl")
    joblib.dump(threshold, f"{MODELS_DIR}/threshold_anomalia.pkl")
    joblib.dump(x.shape[1], f"{MODELS_DIR}/autoencoder_input_dim.pkl")

    return df, modelo, threshold


def calcular_score_anomalia(df_novo: pd.DataFrame) -> pd.DataFrame:
    """
    Usa os artefatos já treinados para calcular o score de anomalia de
    clientes novos (ex: recebidos pela API).
    """
    scaler = joblib.load(f"{MODELS_DIR}/scaler_autoencoder.pkl")
    threshold = joblib.load(f"{MODELS_DIR}/threshold_anomalia.pkl")
    input_dim = joblib.load(f"{MODELS_DIR}/autoencoder_input_dim.pkl")

    modelo = Autoencoder(input_dim=input_dim)
    modelo.load_state_dict(torch.load(f"{MODELS_DIR}/autoencoder.pt"))
    modelo.eval()

    x_scaled = scaler.transform(df_novo[FEATURES_AUTOENCODER])
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        reconstrucao = modelo(x_tensor)
        erro = torch.mean((x_tensor - reconstrucao) ** 2, dim=1)

    df_novo = df_novo.copy()
    df_novo["score_anomalia"] = erro.numpy()
    df_novo["anomalia_detectada"] = (df_novo["score_anomalia"] > threshold).astype(int)
    return df_novo