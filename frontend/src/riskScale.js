// Escala de risco compartilhada por todo o app: o medidor, o badge de
// perfil e as barras da explicação usam exatamente as mesmas 4 cores,
// na mesma ordem, para que o olhar aprenda a associar cor -> risco uma
// única vez e reconheça em qualquer lugar da tela.

export const ESCALA_RISCO = [
  { limite: 0.1, cor: "#35d0a8", nome: "baixo risco" },
  { limite: 0.3, cor: "#f2b84b", nome: "risco moderado" },
  { limite: 0.5, cor: "#f0813d", nome: "risco alto" },
  { limite: 1.01, cor: "#e14f4f", nome: "risco muito alto" },
];

export function corPorProbabilidade(probabilidade) {
  const faixa = ESCALA_RISCO.find((f) => probabilidade < f.limite);
  return faixa ? faixa.cor : ESCALA_RISCO[ESCALA_RISCO.length - 1].cor;
}

export function corPorPerfil(perfilRisco) {
  const encontrado = ESCALA_RISCO.find((f) => f.nome === perfilRisco);
  return encontrado ? encontrado.cor : "#8ea0b8";
}

// Interpola entre os 4 tons para dar um gradiente contínuo no arco do
// medidor, em vez de faixas abruptas.
export function gradienteRisco() {
  return `linear-gradient(90deg, ${ESCALA_RISCO.map((f) => f.cor).join(", ")})`;
}
