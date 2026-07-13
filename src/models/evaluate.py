"""
Avaliação avançada do modelo campeão (XGBoost):
- Curvas ROC e Precision-Recall
- Análise de custos (falso negativo vs. falso positivo)

Análise de custos: em crédito, um Falso Negativo (dizer que o cliente é bom
e ele não paga) custa muito mais caro que um Falso Positivo (negar crédito
a um bom pagador). Os pesos abaixo são um ponto de partida didático - o
valor real dependeria do ticket médio de empréstimo do negócio.
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
)

from config import MODELS_DIR, REPORTS_DIR

CUSTO_FALSO_NEGATIVO = 5
CUSTO_FALSO_POSITIVO = 1


def plotar_curva_roc(y_test, y_prob, nome_modelo="XGBoost"):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{nome_modelo} (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatório")
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.title(f"Curva ROC - {nome_modelo}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/curva_roc.png", dpi=120)
    plt.close()
    print(f"Curva ROC salva em {REPORTS_DIR}/curva_roc.png (AUC = {auc_score:.3f})")

    return auc_score


def plotar_curva_precision_recall(y_test, y_prob, nome_modelo="XGBoost"):
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=nome_modelo)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Curva Precision-Recall - {nome_modelo}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/curva_precision_recall.png", dpi=120)
    plt.close()
    print(f"Curva Precision-Recall salva em {REPORTS_DIR}/curva_precision_recall.png")

    return precision, recall, thresholds


def analise_de_custo(y_test, y_prob, thresholds=None):
    """
    Varre thresholds de decisão e calcula o custo total (ponderado) em
    cada um, para encontrar o ponto de corte que minimiza custo real do
    negócio - não necessariamente o threshold 0.5 padrão.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05)

    resultados = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
        custo_total = fn * CUSTO_FALSO_NEGATIVO + fp * CUSTO_FALSO_POSITIVO
        resultados.append(
            {"threshold": t, "fn": fn, "fp": fp, "custo_total": custo_total}
        )

    melhor = min(resultados, key=lambda r: r["custo_total"])

    print("--- Análise de Custos ---")
    print(f"Custo assumido: FN = {CUSTO_FALSO_NEGATIVO}x, FP = {CUSTO_FALSO_POSITIVO}x")
    print(f"Melhor threshold: {melhor['threshold']:.2f} "
          f"(FN={melhor['fn']}, FP={melhor['fp']}, custo total={melhor['custo_total']})\n")

    return resultados, melhor


def avaliacao_completa(modelo, x_test, y_test, nome_modelo="XGBoost"):
    y_prob = modelo.predict_proba(x_test)[:, 1]

    auc_score = plotar_curva_roc(y_test, y_prob, nome_modelo)
    plotar_curva_precision_recall(y_test, y_prob, nome_modelo)
    resultados_custo, melhor_threshold = analise_de_custo(y_test, y_prob)
    joblib.dump(melhor_threshold["threshold"], f"{MODELS_DIR}/threshold_decisao.pkl")
    return {
        "auc_roc": auc_score,
        "resultados_custo": resultados_custo,
        "melhor_threshold": melhor_threshold,
    }