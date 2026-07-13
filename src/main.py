"""
Ponto de entrada do pipeline: carregar -> tratar -> treinar.

Execução (a partir da raiz do projeto):
    python -m src.main
"""
from config import DATA_PROCESSED, DATA_RAW
from data.load import carregar_dados
from data.preprocess import tratar_dados
from models.train import preparar_dados, treinar_todos
from features.feature_selection import selecionar_features
from features.clustering import treinar_clustering
from features.dimensionality_reduction import treinar_pca
from features.autoencoder import treinar_autoencoder
from models.evaluate import avaliacao_completa
from models.explain import gerar_shap_summary, explicar_cliente


def main():
    print("1. Carregando dados brutos...")
    df = carregar_dados(DATA_RAW)
    print(f"   {df.shape[0]} linhas, {df.shape[1]} colunas carregadas.\n")

    print("2. Tratando dados (missing values, outliers)...")
    df = tratar_dados(df)
    df.to_csv(DATA_PROCESSED, index=False)
    print(f"   Dados tratados salvos em: {DATA_PROCESSED}\n")

    print("3. Treinando modelos supervisionados (LR, RF, XGBoost, MLP)...\n")
    modelos = treinar_todos(df)

    print("4. Seleção automática de atributos...\n")
    resultado_selecao = selecionar_features(df)

    print("5. Clustering hierárquico (perfis de risco)...\n")
    df, resumo_clusters = treinar_clustering(df)

    print("6. PCA (redução de dimensionalidade)...\n")
    df, resumo_variancia = treinar_pca(df)

    print("7. Autoencoder (detecção de anomalias)...\n")
    df, autoencoder, threshold = treinar_autoencoder(df)

    print("8. Avaliação avançada do modelo campeão (ROC, PR, custos)...\n")
    colunas_extras = [
        "cluster_id", "perfil_risco", "pca_1", "pca_2",
        "score_anomalia", "anomalia_detectada",
    ]
    _, x_test, _, y_test, feature_names = preparar_dados(
        df.drop(columns=colunas_extras),
        persistir_scaler=False,
    )
    metricas = avaliacao_completa(modelos["xgb"], x_test, y_test)

    print("9. Explicabilidade (SHAP)...\n")
    explainer, shap_values = gerar_shap_summary(modelos["xgb"], x_test, feature_names)
    explicar_cliente(explainer, x_test, feature_names, indice=0)

    print("Pipeline concluído. Modelos, avaliação e explicabilidade salvos "
          "em /models e /reports.")
    return modelos, df


if __name__ == "__main__":
    main()