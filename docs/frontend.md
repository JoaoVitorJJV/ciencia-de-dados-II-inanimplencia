# Front-end "Radar de Risco" (React)

A camada de visualização do projeto: uma interface de software de risco que consome a API e traduz os resultados técnicos para a linguagem de um analista de crédito.

**Pasta:** [`frontend/`](../frontend/) — React 18 + Vite, sem dependências de UI externas (CSS próprio).

---

## Conceito

Em vez de um dashboard genérico, a interface simula a ferramenta que um **analista de crédito** usaria para avaliar pedidos de cartão: uma ficha do solicitante à esquerda, o resultado da análise à direita. Isso também cumpre o item *Integração com Sistemas* do NAP2 — o front-end é um sistema independente consumindo a API via HTTP.

## Estrutura

```
frontend/
├── index.html                 # carrega as fontes (Space Grotesk, IBM Plex)
├── vite.config.js
└── src/
    ├── main.jsx               # bootstrap do React
    ├── App.jsx                # layout: formulário + painel de resultado
    ├── App.css / index.css    # design system (tokens de cor, tipografia)
    ├── api.js                 # cliente HTTP + definição dos campos + conversão de escala
    ├── riskScale.js           # a escala de 4 cores de risco, compartilhada
    └── components/
        ├── ClientForm.jsx     # a ficha do solicitante
        ├── ResultPanel.jsx    # o painel de resultado
        ├── RiskGauge.jsx      # medidor semicircular (SVG animado)
        └── FactorLedger.jsx   # demonstrativo de fatores (explicação SHAP)
```

## Conversão de escalas

A API espera `debt_ratio` e `utilizacao_credito_rotativo` como **frações 0-1**, mas humanos pensam em **percentual 0-100**. O front-end resolve isso de forma transparente: os dois campos são marcados como `tipo: "percentual"` na definição, e a conversão acontece só na montagem do payload:

```javascript
// src/api.js
export function montarPayload(valoresFormulario) {
  const payload = {};
  for (const campo of todosOsCampos) {
    const valor = Number(valoresFormulario[campo.name]);
    payload[campo.name] = campo.tipo === "percentual" ? valor / 100 : valor;
  }
  return payload;
}
```

O usuário digita "35" (%); a API recebe `0.35`. Nenhum dos dois lados precisa saber da convenção do outro.

## Explicações leigas nos campos

Cada campo do formulário carrega uma explicação escrita **como se o leitor não fizesse parte do projeto** — sem jargão de ciência de dados. A explicação aparece ao focar no campo:

```javascript
// src/api.js — exemplo de definição de campo
{
  name: "debt_ratio",
  label: "Comprometimento de renda com dívidas",
  tipo: "percentual",
  ajuda: "Qual parcela da renda mensal já é usada para pagar dívidas em aberto. " +
         "Se a renda é R$ 4.000 e R$ 1.200 já vão para dívidas, isso é 30%.",
}
```

## Tradução dos resultados para linguagem de negócio

A API devolve dados técnicos; o front-end traduz cada um:

| Dado bruto da API | Como aparece na tela |
|---|---|
| `probabilidade_inadimplencia: 0.0545` | Medidor semicircular com agulha em 5,4% |
| `classificacao: "adimplente"` | *"Sem sinais relevantes de risco"* |
| `classificacao: "inadimplente"` | *"Risco acima do aceitável — recomenda-se análise manual"* |
| `perfil_risco` + `taxa_inadimplencia_historica_perfil` | Badge colorido + *"Historicamente, X% dos clientes com este perfil se tornaram inadimplentes"* |
| `anomalia_detectada: true` | Alerta: *"Comportamento fora do padrão detectado"*, com explicação do que isso significa |
| `explicacao` (valores SHAP) | Demonstrativo de fatores com barras divergentes e nomes amigáveis |

Note o cuidado ético na redação: o resultado é sempre apresentado como **recomendação** ("recomenda-se análise manual"), nunca como recusa automática — e o rodapé deixa explícito que o modelo não substitui um especialista.

## A escala de risco compartilhada

Uma única escala de 4 cores (verde → âmbar → laranja → vermelho) é definida em um só lugar e reutilizada em **todos** os elementos visuais — medidor, badge de perfil, barras de explicação:

```javascript
// src/riskScale.js
export const ESCALA_RISCO = [
  { limite: 0.1,  cor: "#35d0a8", nome: "baixo risco" },
  { limite: 0.3,  cor: "#f2b84b", nome: "risco moderado" },
  { limite: 0.5,  cor: "#f0813d", nome: "risco alto" },
  { limite: 1.01, cor: "#e14f4f", nome: "risco muito alto" },
];
```

O olhar do analista aprende a associação cor → risco uma única vez e a reconhece em qualquer parte da tela.

## O medidor (assinatura visual)

`RiskGauge.jsx` desenha um arco semicircular em SVG, dividido nas 4 faixas de risco, com uma agulha que varre (animada) até o ângulo correspondente à probabilidade:

```javascript
const anguloFinal = -90 + probabilidade * 180;   // 0% = -90°, 100% = +90°
```

## Tratamento de erros

Se a API estiver fora do ar, o painel exibe o erro com a instrução prática ("Confira se a API está rodando em http://127.0.0.1:8000") em vez de quebrar silenciosamente — o estado de erro é parte do design.

## Execução

```bash
cd frontend
npm install     # só na primeira vez
npm run dev     # abre em http://localhost:5173
```

A API precisa estar de pé **antes** ([api.md](api.md#execução)) — é ela quem responde as análises.
