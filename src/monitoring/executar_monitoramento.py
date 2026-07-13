"""
Demonstração do monitoramento de drift.

Em produção, este script rodaria periodicamente (ex: 1x por mês),
comparando o lote de clientes analisados no período contra a distribuição
de referência (o conjunto de treino do modelo campeão).

Como o projeto ainda não tem tráfego real de produção, a demonstração usa
uma divisão temporal simulada do próprio dataset tratado: 80% como
"referência" (o que o modelo aprendeu) e 20% como "novo lote" (simulando
clientes que chegariam depois). Isso é só para fins de demonstração e
validação do código - em produção, "df_atual" seria substituído pelos
dados reais dos clientes analisados no período.

Execução (a partir da raiz do projeto):
    python src/monitoring/executar_monitoramento.py
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

# Garante que 'src' está no path, já que este script fica em uma subpasta
# (src/monitoring/) - sem isso, "from config import ..." quebra ao rodar
# diretamente com "python src/monitoring/executar_monitoramento.py"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_PROCESSED, MODELS_DIR
from features.clustering import FEATURES_CLUSTERING
from monitoring.drift import relatorio_drift_features, monitorar_distribuicao_predicoes


def main():
    df = pd.read_csv(DATA_PROCESSED)

    # Divisão simulada: 80% "referência" (treino), 20% "novo lote" (produção)
    corte = int(len(df) * 0.8)
    df_referencia = df.iloc[:corte]
    df_atual = df.iloc[corte:]

    print("=== Monitoramento de Drift: Features ===\n")
    features_monitoradas = FEATURES_CLUSTERING
    relatorio = relatorio_drift_features(df_referencia, df_atual, features_monitoradas)
    print(relatorio.to_string(index=False))

    alertas = relatorio[relatorio["status"] != "estável"]
    if len(alertas) > 0:
        print(f"\n⚠ {len(alertas)} feature(s) com mudança de distribuição detectada.")
    else:
        print("\nNenhuma feature com mudança relevante de distribuição.")

    # --- Monitoramento da distribuição das previsões ---
    print("\n=== Monitoramento de Drift: Previsões do Modelo ===\n")
    modelo = joblib.load(f"{MODELS_DIR}/modelo_xgb_campeao.pkl")
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    colunas = joblib.load(f"{MODELS_DIR}/colunas_features.pkl")

    x_referencia = scaler.transform(df_referencia[colunas])
    x_atual = scaler.transform(df_atual[colunas])

    prob_referencia = modelo.predict_proba(x_referencia)[:, 1]
    prob_atual = modelo.predict_proba(x_atual)[:, 1]

    resultado = monitorar_distribuicao_predicoes(prob_referencia, prob_atual)
    print(f"PSI das previsões: {resultado['psi_predicoes']} ({resultado['status']})")
    print(f"Probabilidade média (referência): {resultado['taxa_media_referencia']:.4f}")
    print(f"Probabilidade média (novo lote):  {resultado['taxa_media_atual']:.4f}")


if __name__ == "__main__":
    main()