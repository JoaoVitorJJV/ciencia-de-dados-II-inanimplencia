"""
Tratamento de dados: duplicatas, missing values e outliers.

Outliers tratados com limites de domínio (não IQR puro), pois o IQR
gera limites superiores irreais em variáveis financeiras com forte
assimetria (ex: renda_mensal) - decisão já validada no NAP1.
"""
import pandas as pd


def tratar_dados(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()

    # Missing values
    df["renda_mensal"] = df["renda_mensal"].fillna(df["renda_mensal"].median())
    df["dependentes"] = df["dependentes"].fillna(df["dependentes"].mode()[0])

    # renda_mensal: limite inferior por IQR (com fator 3.0, mais permissivo),
    # limite superior fixo (valor impossível para pessoa física)
    q1_renda = df["renda_mensal"].quantile(0.25)
    q3_renda = df["renda_mensal"].quantile(0.75)
    iqr_renda = q3_renda - q1_renda
    limite_inferior_renda = max(q1_renda - 3.0 * iqr_renda, 0)
    limite_superior_renda = 100_000
    df["renda_mensal"] = df["renda_mensal"].clip(
        upper=limite_superior_renda, lower=limite_inferior_renda
    )

    # utilizacao_credito_rotativo e debt_ratio: percentuais, devem ficar entre 0 e 1
    df["utilizacao_credito_rotativo"] = df["utilizacao_credito_rotativo"].clip(
        upper=1, lower=0
    )
    df["debt_ratio"] = df["debt_ratio"].clip(upper=1, lower=0)

    # idade: limites de domínio realistas
    df["idade"] = df["idade"].clip(upper=150, lower=18)

    # atrasos: valores 96 e 98 são códigos de erro/sentinela conhecidos do
    # dataset Give Me Some Credit, não contagens reais. Tentar apenas
    # limitar (clip) esses valores não resolve o problema: as mesmas linhas
    # têm as 3 colunas de atraso simultaneamente no valor sentinela, e
    # também têm padrões inconsistentes associados (ex: 0 linhas de
    # crédito abertas e 0 empréstimos imobiliários, mas 100% de utilização
    # de crédito rotativo) - uma assinatura de registro corrompido, não de
    # cliente real com comportamento extremo. Por isso, essas linhas são
    # removidas (não "clipadas"). Representam ~0.2% do dataset, então a
    # remoção não introduz viés relevante.
    mask_sentinela = (
        (df["atrasos_30_59_dias"] >= 96)
        | (df["atrasos_60_89_dias"] >= 96)
        | (df["atrasos_90_dias"] >= 96)
    )
    df = df.loc[~mask_sentinela].copy()

    # Após remover os registros sentinela, as contagens de atraso não
    # deveriam mais ultrapassar a faixa observada na população legítima.
    df["atrasos_30_59_dias"] = df["atrasos_30_59_dias"].clip(lower=0)
    df["atrasos_60_89_dias"] = df["atrasos_60_89_dias"].clip(lower=0)
    df["atrasos_90_dias"] = df["atrasos_90_dias"].clip(lower=0)

    return df