"""
Explicabilidade do modelo campeão (XGBoost) usando SHAP.

Gera:
- Summary plot (importância global das features)
- Force plot de um cliente individual (explicação local, útil pra API)
"""
import joblib
import matplotlib.pyplot as plt
import shap

from config import MODELS_DIR, REPORTS_DIR


def gerar_shap_summary(modelo, x_test, feature_names, max_display=15):
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(x_test)

    plt.figure()
    shap.summary_plot(
        shap_values, x_test, feature_names=feature_names,
        max_display=max_display, show=False,
    )
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot salvo em {REPORTS_DIR}/shap_summary.png")

    joblib.dump(explainer, f"{MODELS_DIR}/shap_explainer.pkl")

    return explainer, shap_values


def explicar_cliente(explainer, x_cliente, feature_names, indice=0):
    shap_values_cliente = explainer.shap_values(x_cliente)

    contribuicoes = sorted(
        zip(feature_names, shap_values_cliente[indice]),
        key=lambda par: abs(par[1]),
        reverse=True,
    )

    print(f"--- Explicação SHAP - Cliente {indice} ---")
    for feature, valor in contribuicoes[:10]:
        direcao = "aumenta" if valor > 0 else "reduz"
        print(f"  {feature}: {valor:+.4f} ({direcao} risco de inadimplência)")

    return contribuicoes