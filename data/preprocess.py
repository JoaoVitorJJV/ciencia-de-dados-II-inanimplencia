import pandas as pd

def tratar_dados(df):
    df = df.drop_duplicates()

    df['renda_mensal'] = df['renda_mensal'].fillna(df['renda_mensal'].median())
    df['dependentes'] = df['dependentes'].fillna(df['dependentes'].mode()[0])

    q1_renda = df['renda_mensal'].quantile(0.25)
    q3_renda = df['renda_mensal'].quantile(0.75)
    iqr_renda = q3_renda - q1_renda
    limite_inferior_renda = max(q1_renda - 3.0 * iqr_renda, 0)
    limite_superior_renda = 100000
    df['renda_mensal'] = df['renda_mensal'].clip(upper=limite_superior_renda, lower=limite_inferior_renda)

    df['utilizacao_credito_rotativo'] = df['utilizacao_credito_rotativo'].clip(upper=1, lower=0)
    df['debt_ratio'] = df['debt_ratio'].clip(upper=1, lower=0)
    df['idade'] = df['idade'].clip(upper=150, lower=18)
    df['atrasos_30_59_dias'] = df['atrasos_30_59_dias'].clip(lower=0)
    df['atrasos_60_89_dias'] = df['atrasos_60_89_dias'].clip(lower=0)
    df['atrasos_90_dias'] = df['atrasos_90_dias'].clip(lower=0)

    return df