"""
Treinamento e avaliação dos modelos do NAP1: Regressão Logística (baseline),
Random Forest, XGBoost e MLP. O XGBoost é salvo como modelo campeão para o NAP2.
"""
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

from config import MODELS_DIR, RANDOM_STATE, TEST_SIZE


def preparar_dados(df, persistir_scaler=True):
    """
    Split treino/teste -> normalização (fit só no treino) -> SMOTE (só no treino).
    Isso evita data leakage: normalização e balanceamento nunca "veem" o teste.

    persistir_scaler=False permite reutilizar esta função (ex: para reobter
    x_test/y_test na etapa de avaliação) sem sobrescrever o scaler.pkl
    já salvo durante o treino original.
    """
    x = df.drop(columns=["inadimplente"])
    y = df["inadimplente"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    smote = SMOTE(random_state=RANDOM_STATE)
    x_train_bal, y_train_bal = smote.fit_resample(x_train_scaled, y_train)

    if persistir_scaler:
        joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")

    return x_train_bal, x_test_scaled, y_train_bal, y_test, x.columns.tolist()


def avaliar(nome, modelo, x_test, y_test):
    y_pred = modelo.predict(x_test)
    y_prob = modelo.predict_proba(x_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)

    print(f"--- {nome} ---")
    print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
    print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
    print("AUC-ROC:", auc, "\n")

    return {"y_pred": y_pred, "y_prob": y_prob, "auc_roc": auc}


def treinar_todos(df):
    x_train, x_test, y_train, y_test, colunas = preparar_dados(df)

    # --- Baseline: Regressão Logística ---
    modelo_lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    modelo_lr.fit(x_train, y_train)
    avaliar("Baseline: Regressão Logística", modelo_lr, x_test, y_test)

    # --- Random Forest ---
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    modelo_rf.fit(x_train, y_train)
    avaliar("Random Forest", modelo_rf, x_test, y_test)

    # --- XGBoost ---
    modelo_xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    modelo_xgb.fit(x_train, y_train)
    avaliar("XGBoost", modelo_xgb, x_test, y_test)

    # --- MLP (rede neural simples) ---
    modelo_mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=100,
        random_state=RANDOM_STATE,
    )
    modelo_mlp.fit(x_train, y_train)
    avaliar("Rede Neural MLP", modelo_mlp, x_test, y_test)

    # XGBoost = modelo campeão do NAP1, retreinado e persistido para o NAP2
    joblib.dump(modelo_xgb, f"{MODELS_DIR}/modelo_xgb_campeao.pkl")
    joblib.dump(colunas, f"{MODELS_DIR}/colunas_features.pkl")

    return {"lr": modelo_lr, "rf": modelo_rf, "xgb": modelo_xgb, "mlp": modelo_mlp}