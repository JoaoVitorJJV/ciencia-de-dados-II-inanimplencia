"""
Schemas Pydantic: validação de entrada e formato de resposta da API.
"""
from pydantic import BaseModel, Field


class ClienteInput(BaseModel):
    utilizacao_credito_rotativo: float = Field(
        ..., ge=0, le=1, description="Percentual de uso do crédito rotativo (0 a 1)"
    )
    idade: int = Field(..., ge=18, le=150)
    atrasos_30_59_dias: int = Field(..., ge=0)
    debt_ratio: float = Field(..., ge=0, le=1)
    renda_mensal: float = Field(..., ge=0)
    linhas_credito_abertas: int = Field(..., ge=0)
    atrasos_90_dias: int = Field(..., ge=0)
    emprestimos_imobiliarios: int = Field(..., ge=0)
    atrasos_60_89_dias: int = Field(..., ge=0)
    dependentes: int = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "utilizacao_credito_rotativo": 0.3,
                "idade": 45,
                "atrasos_30_59_dias": 0,
                "debt_ratio": 0.35,
                "renda_mensal": 5000.0,
                "linhas_credito_abertas": 8,
                "atrasos_90_dias": 0,
                "emprestimos_imobiliarios": 1,
                "atrasos_60_89_dias": 0,
                "dependentes": 1,
            }
        }


class ContribuicaoFeature(BaseModel):
    feature: str
    valor_shap: float
    direcao: str  


class PrevisaoResponse(BaseModel):
    probabilidade_inadimplencia: float
    classificacao: str

    perfil_risco: str
    taxa_inadimplencia_historica_perfil: float

    score_anomalia: float
    anomalia_detectada: bool

    explicacao: list[ContribuicaoFeature]