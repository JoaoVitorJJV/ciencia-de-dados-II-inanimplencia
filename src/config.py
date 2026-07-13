"""
Configurações centrais do projeto — Predição de Inadimplência.
Todos os módulos importam paths e parâmetros a partir daqui,
para evitar valores "mágicos" espalhados pelo código.
"""
import os

# --- Reprodutibilidade ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw", "cs-training.csv")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "dataset_tratado.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# --- Dataset: Give Me Some Credit (Kaggle, cs-training.csv, ~150.000 registros) ---
# Colunas renomeadas para português para facilitar o entendimento
COLUNAS_PT = [
    "inadimplente",
    "utilizacao_credito_rotativo",
    "idade",
    "atrasos_30_59_dias",
    "debt_ratio",
    "renda_mensal",
    "linhas_credito_abertas",
    "atrasos_90_dias",
    "emprestimos_imobiliarios",
    "atrasos_60_89_dias",
    "dependentes",
]

COLUNAS_ATRASO = ["atrasos_30_59_dias", "atrasos_60_89_dias", "atrasos_90_dias"]
