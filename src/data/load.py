"""Carregamento do dataset bruto."""
import pandas as pd

from config import COLUNAS_PT


def carregar_dados(caminho: str) -> pd.DataFrame:
    """
    Carrega o dataset Give Me Some Credit e renomeia as colunas para português.

    Parameters
    ----------
    caminho : str
        Caminho para o arquivo cs-training.csv

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(caminho, index_col=0)
    df.columns = COLUNAS_PT
    return df
