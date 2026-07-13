"""
Clustering hierárquico para segmentação de perfis de risco.

O clustering hierárquico aglomerativo tem complexidade O(n2) em memória, inviável para
150.000 linhas. Por isso:

  1. Amostra representativa (ex: 5.000 clientes) é usada para rodar o
     clustering hierárquico de fato e descobrir os perfis de risco.
  2. Um classificador surrogate (KNN) é treinado nessa amostra rotulada,
     permitindo atribuir o cluster a qualquer cliente novo (todo o dataset
     original, e futuramente, clientes recebidos pela API) sem re-rodar o
     algoritmo hierárquico.

Perfis são interpretados (nomeados) olhando a taxa de inadimplência real de
cada cluster - isso é feito DEPOIS do clustering, só para interpretação;
a variável 'inadimplente' não entra como feature no algoritmo.
"""
import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from config import MODELS_DIR, RANDOM_STATE

# Features usadas para o perfil de risco (comportamento, não dados pessoais)
FEATURES_CLUSTERING = [
    "utilizacao_credito_rotativo",
    "debt_ratio",
    "renda_mensal",
    "atrasos_30_59_dias",
    "atrasos_60_89_dias",
    "atrasos_90_dias",
    "linhas_credito_abertas",
    "emprestimos_imobiliarios",
    "dependentes",
    "idade",
]

N_CLUSTERS = 4
TAMANHO_AMOSTRA = 5000


def _amostrar(df: pd.DataFrame, n=TAMANHO_AMOSTRA) -> pd.DataFrame:
    n = min(n, len(df))
    return df.sample(n=n, random_state=RANDOM_STATE)


def treinar_clustering(df: pd.DataFrame):
    """
    Executa o clustering hierárquico na amostra, treina o surrogate KNN,
    aplica o cluster a todo o dataset e nomeia os perfis pela taxa de
    inadimplência observada em cada grupo.

    Retorna o DataFrame original com duas colunas novas:
    - cluster_id: id numérico do cluster
    - perfil_risco: rótulo interpretável (baixo/médio/alto/muito alto risco)
    """
    amostra = _amostrar(df)
    x_amostra = amostra[FEATURES_CLUSTERING]

    scaler = StandardScaler()
    x_amostra_scaled = scaler.fit_transform(x_amostra)

    # Linkage de Ward minimiza a variância dentro de cada cluster
    Z = linkage(x_amostra_scaled, method="ward")
    labels_amostra = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")

    # Surrogate: aprende a mapear features -> cluster, para generalizar
    # a atribuição além da amostra usada no clustering hierárquico
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_amostra_scaled, labels_amostra)

    # Aplica a todo o dataset
    x_full_scaled = scaler.transform(df[FEATURES_CLUSTERING])
    df = df.copy()
    df["cluster_id"] = knn.predict(x_full_scaled)

    # Interpretação: taxa de inadimplência por cluster define o rótulo
    taxa_por_cluster = df.groupby("cluster_id")["inadimplente"].mean().sort_values()
    ordem_risco = ["baixo risco", "risco moderado", "risco alto", "risco muito alto"]
    mapa_nomes = {
        cluster_id: ordem_risco[i]
        for i, cluster_id in enumerate(taxa_por_cluster.index)
    }
    df["perfil_risco"] = df["cluster_id"].map(mapa_nomes)

    print("--- Clustering Hierárquico: Perfis de Risco ---")
    resumo = (
        df.groupby("perfil_risco")
        .agg(
            qtd_clientes=("inadimplente", "size"),
            taxa_inadimplencia=("inadimplente", "mean"),
            renda_media=("renda_mensal", "mean"),
            debt_ratio_medio=("debt_ratio", "mean"),
        )
        .sort_values("taxa_inadimplencia")
    )
    print(resumo, "\n")

    joblib.dump(scaler, f"{MODELS_DIR}/scaler_clustering.pkl")
    joblib.dump(knn, f"{MODELS_DIR}/knn_surrogate_cluster.pkl")
    joblib.dump(mapa_nomes, f"{MODELS_DIR}/mapa_perfis_risco.pkl")
    joblib.dump(
        resumo["taxa_inadimplencia"].to_dict(),
        f"{MODELS_DIR}/taxa_inadimplencia_por_perfil.pkl",
    )

    return df, resumo


def atribuir_perfil_risco(df_novo: pd.DataFrame) -> pd.DataFrame:
    """
    Usa os artefatos já treinados (scaler + KNN surrogate + mapa de nomes)
    para atribuir perfil de risco a clientes novos (ex: recebidos pela API).
    """
    scaler = joblib.load(f"{MODELS_DIR}/scaler_clustering.pkl")
    knn = joblib.load(f"{MODELS_DIR}/knn_surrogate_cluster.pkl")
    mapa_nomes = joblib.load(f"{MODELS_DIR}/mapa_perfis_risco.pkl")

    x_scaled = scaler.transform(df_novo[FEATURES_CLUSTERING])
    df_novo = df_novo.copy()
    df_novo["cluster_id"] = knn.predict(x_scaled)
    df_novo["perfil_risco"] = df_novo["cluster_id"].map(mapa_nomes)
    return df_novo