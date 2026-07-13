import pandas as pd
from src.config import COLUNAS_PT

def carregar_dados(caminho):
    df = pd.read_csv(caminho, index_col=0)
    df.columns = COLUNAS_PT
    return df