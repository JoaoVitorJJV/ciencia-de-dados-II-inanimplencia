import React from "react";
import RiskGauge from "./RiskGauge.jsx";
import FactorLedger from "./FactorLedger.jsx";
import { corPorPerfil } from "../riskScale.js";

const TEXTO_DECISAO = {
  adimplente: {
    titulo: "Sem sinais relevantes de risco",
    descricao:
      "Com os dados informados, o modelo não encontrou motivo para recusar o pedido automaticamente.",
  },
  inadimplente: {
    titulo: "Risco acima do aceitável",
    descricao:
      "Com os dados informados, o modelo recomenda uma análise manual antes de aprovar o pedido.",
  },
};

export default function ResultPanel({ resultado, erro, carregando }) {
  if (erro) {
    return (
      <aside className="result-panel result-panel--erro">
        <p className="eyebrow">Não foi possível concluir a análise</p>
        <h3>A API não respondeu como esperado</h3>
        <p>{erro}</p>
        <p className="result-panel__dica">
          Confira se a API (FastAPI) está rodando em{" "}
          <code>http://127.0.0.1:8000</code>.
        </p>
      </aside>
    );
  }

  if (!resultado && !carregando) {
    return (
      <aside className="result-panel result-panel--vazio">
        <p className="eyebrow">Aguardando dados</p>
        <h3>Preencha a ficha ao lado para começar</h3>
        <p>
          O resultado da análise — probabilidade de inadimplência, perfil
          de risco e os fatores que mais pesaram na decisão — aparece
          aqui assim que você enviar os dados do solicitante.
        </p>
      </aside>
    );
  }

  if (carregando && !resultado) {
    return (
      <aside className="result-panel result-panel--carregando">
        <p className="eyebrow">Analisando</p>
        <h3>Consultando o modelo de risco…</h3>
      </aside>
    );
  }

  const decisao = TEXTO_DECISAO[resultado.classificacao];
  const corPerfil = corPorPerfil(resultado.perfil_risco);

  return (
    <aside className="result-panel">
      <p className="eyebrow">Resultado da análise</p>

      <RiskGauge probabilidade={resultado.probabilidade_inadimplencia} />

      <div
        className={`decision-chip decision-chip--${resultado.classificacao}`}
      >
        <h3>{decisao.titulo}</h3>
        <p>{decisao.descricao}</p>
      </div>

      <div className="profile-badge" style={{ borderColor: corPerfil }}>
        <span className="profile-badge__dot" style={{ background: corPerfil }} />
        <div>
          <p className="profile-badge__label">Perfil de risco</p>
          <p className="profile-badge__value" style={{ color: corPerfil }}>
            {resultado.perfil_risco}
          </p>
          <p className="profile-badge__contexto">
            Historicamente, {(
              resultado.taxa_inadimplencia_historica_perfil * 100
            ).toFixed(1)}
            % dos clientes com este perfil se tornaram inadimplentes.
          </p>
        </div>
      </div>

      {resultado.anomalia_detectada && (
        <div className="anomaly-banner">
          <span className="anomaly-banner__icone">⚠</span>
          <div>
            <p className="anomaly-banner__titulo">
              Comportamento fora do padrão detectado
            </p>
            <p>
              O perfil financeiro deste solicitante foge do padrão
              observado em clientes que pagam em dia. Isso não significa
              recusa automática, mas recomenda-se atenção extra na análise
              manual.
            </p>
          </div>
        </div>
      )}

      <div className="ledger-section">
        <h4>Por que o modelo chegou a esse resultado</h4>
        <p className="ledger-section__intro">
          Estes são os dados do solicitante que mais influenciaram o
          resultado, do mais para o menos importante.
        </p>
        <FactorLedger explicacao={resultado.explicacao} />
      </div>
    </aside>
  );
}
