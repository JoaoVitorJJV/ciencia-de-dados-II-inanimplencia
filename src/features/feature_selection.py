"""
Seleção automática de atributos (Feature Engineering Avançado).

Usa a importância de features do XGBoost com
SelectFromModel para descartar automaticamente as features de menor
poder preditivo (abaixo da mediana de importância), e compara o
AUC-ROC do modelo completo vs. do modelo reduzido - validando
empiricamente se a seleção automática mantém a performance.
"""
import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from config import MODELS_DIR, RANDOM_STATE
from models.train import preparar_dados


def selecionar_features(df, threshold="median"):
    x_train, x_test, y_train, y_test, colunas = preparar_dados(
        df, persistir_scaler=False
    )

    # Modelo completo (todas as features), usado como referência de
    # importância e como baseline de comparação
    modelo_completo = XGBClassifier(
        n_estimators=100, learning_rate=0.1,
        random_state=RANDOM_STATE, eval_metric="logloss",
    )
    modelo_completo.fit(x_train, y_train)
    auc_completo = roc_auc_score(
        y_test, modelo_completo.predict_proba(x_test)[:, 1]
    )

    # Seleção automática: mantém apenas features com importância
    # acima do threshold (mediana, por padrão)
    selector = SelectFromModel(modelo_completo, threshold=threshold, prefit=True)
    x_train_sel = selector.transform(x_train)
    x_test_sel = selector.transform(x_test)

    mascara = selector.get_support()
    colunas_selecionadas = [c for c, mantida in zip(colunas, mascara) if mantida]
    colunas_descartadas = [c for c, mantida in zip(colunas, mascara) if not mantida]

    # Modelo reduzido, retreinado só com as features selecionadas
    modelo_reduzido = XGBClassifier(
        n_estimators=100, learning_rate=0.1,
        random_state=RANDOM_STATE, eval_metric="logloss",
    )
    modelo_reduzido.fit(x_train_sel, y_train)
    auc_reduzido = roc_auc_score(
        y_test, modelo_reduzido.predict_proba(x_test_sel)[:, 1]
    )

    print("--- Seleção Automática de Atributos ---")
    print(f"Features originais ({len(colunas)}): {colunas}")
    print(f"\nFeatures selecionadas ({len(colunas_selecionadas)}): {colunas_selecionadas}")
    print(f"Features descartadas ({len(colunas_descartadas)}): {colunas_descartadas}\n")
    print(f"AUC-ROC modelo completo:  {auc_completo:.4f}")
    print(f"AUC-ROC modelo reduzido:  {auc_reduzido:.4f}")
    print(f"Diferença: {auc_reduzido - auc_completo:+.4f}\n")

    joblib.dump(colunas_selecionadas, f"{MODELS_DIR}/features_selecionadas.pkl")
    joblib.dump(selector, f"{MODELS_DIR}/selector_features.pkl")
    joblib.dump(modelo_reduzido, f"{MODELS_DIR}/modelo_xgb_reduzido.pkl")

    return {
        "colunas_selecionadas": colunas_selecionadas,
        "colunas_descartadas": colunas_descartadas,
        "auc_completo": auc_completo,
        "auc_reduzido": auc_reduzido,
        "modelo_reduzido": modelo_reduzido,
    }