import React from "react";
import { NOMES_AMIGAVEIS } from "../api.js";

/**
 * Cada fator vira uma linha de "razão": nome à esquerda, barra divergindo
 * de uma linha central (esquerda = reduz risco, direita = aumenta risco),
 * valor numérico em mono à direita — como um demonstrativo de variação
 * financeira, reforçando o tom de instrumento de análise, não brinquedo.
 */
export default function FactorLedger({ explicacao }) {
  const maiorMagnitude = Math.max(...explicacao.map((f) => Math.abs(f.valor_shap)));

  return (
    <div className="factor-ledger">
      <div className="factor-ledger__header">
        <span>Fator considerado</span>
        <span>Efeito na decisão</span>
      </div>

      {explicacao.map((fator, indice) => {
        const larguraPercentual = (Math.abs(fator.valor_shap) / maiorMagnitude) * 100;
        const aumenta = fator.direcao === "aumenta";

        return (
          <div
            className="factor-ledger__row"
            key={fator.feature}
            style={{ transitionDelay: `${indice * 45}ms` }}
          >
            <span className="factor-ledger__name">
              {NOMES_AMIGAVEIS[fator.feature] ?? fator.feature}
            </span>

            <div className="factor-ledger__bar-track">
              <div className="factor-ledger__center-line" />
              <div
                className={`factor-ledger__bar factor-ledger__bar--${
                  aumenta ? "up" : "down"
                }`}
                style={{ width: `${larguraPercentual / 2}%` }}
              />
            </div>

            <span
              className={`factor-ledger__tag factor-ledger__tag--${
                aumenta ? "up" : "down"
              }`}
            >
              {aumenta ? "aumenta risco" : "reduz risco"}
            </span>
          </div>
        );
      })}
    </div>
  );
}
