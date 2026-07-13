"""
Redução de dimensionalidade via PCA (Principal Component Analysis).

Completa a trilha "Não Supervisionado" do NAP2 (junto com clustering
hierárquico e autoencoder). Serve a dois propósitos:

1. Reduzir as features comportamentais a 2 componentes principais,
   permitindo visualizar os perfis de risco em um plano 2D (útil para
   o dashboard).
2. Quantificar quanta variância dos dados originais é preservada nessa
   redução - item explicitamente pedido pelo PDF (redução de
   dimensionalidade como técnica, não só como visualização).

Usa as mesmas features do clustering (comportamentais), para que os
dois métodos não supervisionados sejam diretamente comparáveis.
"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import MODELS_DIR, REPORTS_DIR
from features.clustering import FEATURES_CLUSTERING as FEATURES_PCA


def treinar_pca(df: pd.DataFrame, n_componentes: int = 2):
    """
    Ajusta o PCA nas features comportamentais e retorna o DataFrame com
    as componentes principais adicionadas, além do resumo de variância
    explicada por componente.
    """
    x = df[FEATURES_PCA]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    pca = PCA(n_components=n_componentes, random_state=42)
    componentes = pca.fit_transform(x_scaled)

    df = df.copy()
    for i in range(n_componentes):
        df[f"pca_{i + 1}"] = componentes[:, i]

    variancia_explicada = pca.explained_variance_ratio_
    variancia_acumulada = variancia_explicada.cumsum()

    print("--- PCA: Redução de Dimensionalidade ---")
    print(f"Features originais: {len(FEATURES_PCA)} -> Componentes: {n_componentes}")
    for i, (var, var_acum) in enumerate(zip(variancia_explicada, variancia_acumulada)):
        print(f"  PC{i + 1}: {var:.2%} da variância (acumulado: {var_acum:.2%})")
    print()

    # Visualização 2D dos perfis de risco no espaço reduzido (se já existir
    # a coluna de perfil_risco, gerada pelo clustering)
    if n_componentes >= 2:
        plt.figure(figsize=(7, 6))
        if "perfil_risco" in df.columns:
            for perfil in df["perfil_risco"].unique():
                subset = df[df["perfil_risco"] == perfil]
                plt.scatter(subset["pca_1"], subset["pca_2"], s=4, alpha=0.4, label=perfil)
            plt.legend(markerscale=4)
        else:
            plt.scatter(df["pca_1"], df["pca_2"], s=4, alpha=0.4, c=df["inadimplente"])
        plt.xlabel(f"PC1 ({variancia_explicada[0]:.1%} da variância)")
        plt.ylabel(f"PC2 ({variancia_explicada[1]:.1%} da variância)")
        plt.title("PCA - Clientes no espaço de componentes principais")
        plt.tight_layout()
        plt.savefig(f"{REPORTS_DIR}/pca_scatter.png", dpi=120)
        plt.close()
        print(f"Gráfico PCA salvo em {REPORTS_DIR}/pca_scatter.png\n")

    joblib.dump(scaler, f"{MODELS_DIR}/scaler_pca.pkl")
    joblib.dump(pca, f"{MODELS_DIR}/pca.pkl")

    resumo_variancia = {
        f"PC{i + 1}": {"variancia": v, "variancia_acumulada": va}
        for i, (v, va) in enumerate(zip(variancia_explicada, variancia_acumulada))
    }

    return df, resumo_variancia


def aplicar_pca(df_novo: pd.DataFrame) -> pd.DataFrame:
    """
    Usa os artefatos já treinados para projetar clientes novos no mesmo
    espaço de componentes principais (ex: clientes recebidos pela API).
    """
    scaler = joblib.load(f"{MODELS_DIR}/scaler_pca.pkl")
    pca = joblib.load(f"{MODELS_DIR}/pca.pkl")

    x_scaled = scaler.transform(df_novo[FEATURES_PCA])
    componentes = pca.transform(x_scaled)

    df_novo = df_novo.copy()
    for i in range(componentes.shape[1]):
        df_novo[f"pca_{i + 1}"] = componentes[:, i]

    return df_novo