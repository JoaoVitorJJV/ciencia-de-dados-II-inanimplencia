import React, { useEffect, useState } from "react";
import { corPorProbabilidade } from "../riskScale.js";

/**
 * Medidor semicircular: um arco de 180° dividido nas 4 faixas de risco,
 * com uma agulha que varre até o ângulo correspondente à probabilidade.
 * É o elemento de assinatura visual do app — o mesmo princípio (agulha
 * sobre arco calibrado) que qualquer analista de risco reconheceria de
 * instrumentos de medição física.
 */
export default function RiskGauge({ probabilidade }) {
  const [anguloAnimado, setAnguloAnimado] = useState(-90);

  const anguloFinal = -90 + probabilidade * 180;

  useEffect(() => {
    const raf = requestAnimationFrame(() => setAnguloAnimado(anguloFinal));
    return () => cancelAnimationFrame(raf);
  }, [anguloFinal]);

  const cor = corPorProbabilidade(probabilidade);
  const cx = 150;
  const cy = 150;
  const raio = 110;

  const agulhaRad = (anguloAnimado * Math.PI) / 180;
  const pontaX = cx + raio * 0.82 * Math.cos(agulhaRad);
  const pontaY = cy + raio * 0.82 * Math.sin(agulhaRad);

  const faixas = [
    { de: -90, ate: -72, cor: "#35d0a8" },
    { de: -72, ate: -18, cor: "#f2b84b" },
    { de: -18, ate: 36, cor: "#f0813d" },
    { de: 36, ate: 90, cor: "#e14f4f" },
  ];

  const arco = (de, ate, corFaixa) => {
    const rad1 = (de * Math.PI) / 180;
    const rad2 = (ate * Math.PI) / 180;
    const x1 = cx + raio * Math.cos(rad1);
    const y1 = cy + raio * Math.sin(rad1);
    const x2 = cx + raio * Math.cos(rad2);
    const y2 = cy + raio * Math.sin(rad2);
    return (
      <path
        key={`${de}-${ate}`}
        d={`M ${x1} ${y1} A ${raio} ${raio} 0 0 1 ${x2} ${y2}`}
        stroke={corFaixa}
        strokeWidth="20"
        strokeLinecap="butt"
        fill="none"
        opacity="0.9"
      />
    );
  };

  return (
    <div className="risk-gauge">
      <svg viewBox="0 0 300 170" width="100%" height="auto">
        {faixas.map((f) => arco(f.de, f.ate, f.cor))}

        {/* Agulha */}
        <line
          x1={cx}
          y1={cy}
          x2={pontaX}
          y2={pontaY}
          stroke="var(--text-primary)"
          strokeWidth="3"
          strokeLinecap="round"
          style={{ transition: "all 0.9s cubic-bezier(0.22, 1, 0.36, 1)" }}
        />
        <circle cx={cx} cy={cy} r="6" fill="var(--text-primary)" />
      </svg>

      <div className="risk-gauge__reading">
        <span className="risk-gauge__value" style={{ color: cor }}>
          {(probabilidade * 100).toFixed(1)}%
        </span>
        <span className="risk-gauge__label">probabilidade de inadimplência</span>
      </div>
    </div>
  );
}
