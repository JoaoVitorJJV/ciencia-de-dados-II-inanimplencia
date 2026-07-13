"""
Monitoramento de drift: detecta se a distribuição dos dados (ou das
previsões) mudou entre o período de referência (treino) e um novo lote
de dados (produção).

Usa PSI (Population Stability Index), o indicador padrão da indústria
de crédito/risco para esse tipo de comparação. Convenção de leitura:
- PSI < 0.10: sem mudança relevante
- 0.10 <= PSI < 0.25: mudança moderada, vale atenção
- PSI >= 0.25: mudança significativa, investigar/possivelmente retreinar
"""
import numpy as np
import pandas as pd


def calcular_psi(referencia: np.ndarray, atual: np.ndarray, buckets: int = 10) -> float:
    """
    Calcula o PSI de uma única variável entre dois períodos.

    A variável de referência define os limites dos buckets (percentis);
    depois comparamos a proporção de pontos em cada bucket entre
    referência e atual.
    """
    referencia = np.asarray(referencia, dtype=float)
    atual = np.asarray(atual, dtype=float)

    limites = np.unique(
        np.percentile(referencia, np.linspace(0, 100, buckets + 1))
    )
    if len(limites) < 3:
        # Variável quase constante na referência; PSI não é informativo
        return 0.0

    freq_referencia, _ = np.histogram(referencia, bins=limites)
    freq_atual, _ = np.histogram(atual, bins=limites)

    prop_referencia = np.clip(freq_referencia / len(referencia), 1e-4, None)
    prop_atual = np.clip(freq_atual / len(atual), 1e-4, None)

    psi = np.sum((prop_atual - prop_referencia) * np.log(prop_atual / prop_referencia))
    return float(psi)


def _interpretar_psi(psi: float) -> str:
    if psi < 0.10:
        return "estável"
    elif psi < 0.25:
        return "mudança moderada"
    else:
        return "mudança significativa"


def relatorio_drift_features(
    df_referencia: pd.DataFrame, df_atual: pd.DataFrame, features: list
) -> pd.DataFrame:
    """
    Gera o relatório de PSI para cada feature, comparando o período de
    referência (treino) com um novo lote de dados (produção).
    """
    linhas = []
    for feature in features:
        psi = calcular_psi(df_referencia[feature], df_atual[feature])
        linhas.append(
            {
                "feature": feature,
                "psi": round(psi, 4),
                "status": _interpretar_psi(psi),
            }
        )

    relatorio = pd.DataFrame(linhas).sort_values("psi", ascending=False)
    return relatorio


def monitorar_distribuicao_predicoes(
    prob_referencia: np.ndarray, prob_atual: np.ndarray
) -> dict:
    """
    Monitora se a distribuição das PROBABILIDADES PREVISTAS mudou entre
    referência e o lote atual. Diferente do drift de features, este é um
    proxy de performance útil quando ainda não há rótulo real disponível
    (em crédito, o resultado real de inadimplência só é conhecido meses
    depois da decisão) - se o modelo passa a prever probabilidades muito
    diferentes do padrão histórico, é sinal de alerta mesmo sem saber
    ainda se ele está "certo".
    """
    psi = calcular_psi(prob_referencia, prob_atual)
    return {
        "psi_predicoes": round(psi, 4),
        "status": _interpretar_psi(psi),
        "taxa_media_referencia": float(np.mean(prob_referencia)),
        "taxa_media_atual": float(np.mean(prob_atual)),
    }