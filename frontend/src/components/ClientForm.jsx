import React, { useState } from "react";
import { CAMPOS } from "../api.js";

export default function ClientForm({ onSubmit, carregando }) {
  const valoresIniciais = Object.fromEntries(
    CAMPOS.flatMap((g) => g.itens).map((c) => [c.name, c.default])
  );
  const [valores, setValores] = useState(valoresIniciais);
  const [campoAtivo, setCampoAtivo] = useState(null);

  function atualizarCampo(nome, valor) {
    setValores((atual) => ({ ...atual, [nome]: valor }));
  }

  function handleSubmit(evento) {
    evento.preventDefault();
    onSubmit(valores);
  }

  return (
    <form className="client-form" onSubmit={handleSubmit}>
      <header className="client-form__header">
        <p className="eyebrow">Ficha do solicitante</p>
        <h2>Dados para análise</h2>
        <p className="client-form__subtitle">
          Preencha as informações abaixo para calcular o risco de
          inadimplência do pedido de cartão.
        </p>
      </header>

      {CAMPOS.map((grupo) => (
        <fieldset className="client-form__group" key={grupo.grupo}>
          <legend>{grupo.grupo}</legend>

          {grupo.itens.map((campo) => (
            <div className="field" key={campo.name}>
              <label htmlFor={campo.name}>
                {campo.label}
                <span className="field__unidade">{campo.unidade}</span>
              </label>

              <input
                id={campo.name}
                type="number"
                min={campo.min}
                max={campo.max}
                step={campo.tipo === "percentual" ? 1 : 1}
                value={valores[campo.name]}
                onFocus={() => setCampoAtivo(campo.name)}
                onBlur={() => setCampoAtivo(null)}
                onChange={(e) => atualizarCampo(campo.name, e.target.value)}
                required
              />

              <p
                className={`field__ajuda ${
                  campoAtivo === campo.name ? "field__ajuda--visivel" : ""
                }`}
              >
                {campo.ajuda}
              </p>
            </div>
          ))}
        </fieldset>
      ))}

      <button type="submit" className="btn-analisar" disabled={carregando}>
        {carregando ? "Analisando…" : "Analisar risco"}
      </button>
    </form>
  );
}
