// URL base da API FastAPI. Ajuste se estiver rodando em outra porta/host.
export const API_URL = "http://localhost:8000";

/**
 * Definição dos campos do formulário.
 *
 * A API espera `utilizacao_credito_rotativo` e `debt_ratio` como frações
 * de 0 a 1 (ex: 0.35 = 35%). Para quem preenche o formulário, é muito
 * mais natural pensar em porcentagem (0 a 100). Por isso esses dois campos
 * são tratados como "percentual" na interface: o usuário digita 0-100,
 * e convertemos para 0-1 só na hora de montar o payload da API.
 */
export const CAMPOS = [
  {
    grupo: "Identificação",
    itens: [
      {
        name: "idade",
        label: "Idade",
        tipo: "numero",
        unidade: "anos",
        min: 18,
        max: 100,
        default: 35,
        ajuda:
          "Idade do solicitante do cartão, em anos completos.",
      },
      {
        name: "dependentes",
        label: "Dependentes",
        tipo: "numero",
        unidade: "pessoas",
        min: 0,
        max: 15,
        default: 0,
        ajuda:
          "Quantas pessoas dependem financeiramente do solicitante — filhos, cônjuge sem renda própria, etc.",
      },
    ],
  },
  {
    grupo: "Renda e dívidas",
    itens: [
      {
        name: "renda_mensal",
        label: "Renda mensal",
        tipo: "numero",
        unidade: "R$",
        min: 0,
        max: 500000,
        default: 4000,
        ajuda:
          "Quanto o solicitante recebe por mês, somando salário e outras rendas fixas.",
      },
      {
        name: "debt_ratio",
        label: "Comprometimento de renda com dívidas",
        tipo: "percentual",
        unidade: "%",
        min: 0,
        max: 100,
        default: 30,
        ajuda:
          "Qual parcela da renda mensal já é usada para pagar dívidas em aberto (financiamentos, empréstimos, outras parcelas). Se a renda é R$ 4.000 e R$ 1.200 já vão para dívidas, isso é 30%.",
      },
      {
        name: "utilizacao_credito_rotativo",
        label: "Uso do limite do crédito rotativo",
        tipo: "percentual",
        unidade: "%",
        min: 0,
        max: 100,
        default: 30,
        ajuda:
          "Quanto do limite disponível em cartões de crédito o solicitante costuma usar. Limite de R$ 1.000 e uso médio de R$ 300 = 30%.",
      },
    ],
  },
  {
    grupo: "Linhas de crédito ativas",
    itens: [
      {
        name: "linhas_credito_abertas",
        label: "Linhas de crédito abertas",
        tipo: "numero",
        unidade: "linhas",
        min: 0,
        max: 60,
        default: 6,
        ajuda:
          "Número de cartões de crédito, empréstimos ou financiamentos que o solicitante tem ativos hoje.",
      },
      {
        name: "emprestimos_imobiliarios",
        label: "Financiamentos imobiliários",
        tipo: "numero",
        unidade: "financiamentos",
        min: 0,
        max: 20,
        default: 0,
        ajuda:
          "Quantos financiamentos de imóvel (casa, apartamento, terreno) o solicitante tem em aberto.",
      },
    ],
  },
  {
    grupo: "Histórico de atrasos",
    itens: [
      {
        name: "atrasos_30_59_dias",
        label: "Atrasos de 30 a 59 dias",
        tipo: "numero",
        unidade: "vezes",
        min: 0,
        max: 20,
        default: 0,
        ajuda:
          "Quantas vezes o solicitante atrasou uma fatura ou parcela entre 30 e 59 dias, no histórico recente.",
      },
      {
        name: "atrasos_60_89_dias",
        label: "Atrasos de 60 a 89 dias",
        tipo: "numero",
        unidade: "vezes",
        min: 0,
        max: 20,
        default: 0,
        ajuda:
          "Quantas vezes o solicitante atrasou um pagamento entre 60 e 89 dias — um atraso mais sério que o anterior.",
      },
      {
        name: "atrasos_90_dias",
        label: "Atrasos de 90 dias ou mais",
        tipo: "numero",
        unidade: "vezes",
        min: 0,
        max: 20,
        default: 0,
        ajuda:
          "Quantas vezes o solicitante atrasou um pagamento por 90 dias ou mais — o tipo de atraso mais grave no histórico de crédito.",
      },
    ],
  },
];

// Nomes amigáveis das features, para exibir na explicação (em vez do
// nome técnico da coluna, ex: "atrasos_30_59_dias").
export const NOMES_AMIGAVEIS = Object.fromEntries(
  CAMPOS.flatMap((grupo) => grupo.itens).map((campo) => [campo.name, campo.label])
);

/**
 * Monta o payload da API a partir dos valores do formulário, convertendo
 * os campos "percentual" (0-100, como o usuário digita) para a fração
 * 0-1 que a API espera.
 */
export function montarPayload(valoresFormulario) {
  const payload = {};
  for (const grupo of CAMPOS) {
    for (const campo of grupo.itens) {
      const valor = Number(valoresFormulario[campo.name]);
      payload[campo.name] = campo.tipo === "percentual" ? valor / 100 : valor;
    }
  }
  return payload;
}

export async function preverRisco(valoresFormulario) {
  const payload = montarPayload(valoresFormulario);

  const resposta = await fetch(`${API_URL}/prever`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!resposta.ok) {
    const detalhe = await resposta.text();
    throw new Error(`Falha ao consultar a API (${resposta.status}): ${detalhe}`);
  }

  return resposta.json();
}
