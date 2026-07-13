import React, { useState } from "react";
import ClientForm from "./components/ClientForm.jsx";
import ResultPanel from "./components/ResultPanel.jsx";
import { preverRisco } from "./api.js";
import "./App.css";

export default function App() {
  const [resultado, setResultado] = useState(null);
  const [erro, setErro] = useState(null);
  const [carregando, setCarregando] = useState(false);

  async function handleSubmit(valoresFormulario) {
    setCarregando(true);
    setErro(null);
    try {
      const dados = await preverRisco(valoresFormulario);
      setResultado(dados);
    } catch (e) {
      setErro(e.message);
      setResultado(null);
    } finally {
      setCarregando(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header__brand">
          <span className="app-header__mark" aria-hidden="true" />
          <div>
            <h1>Radar de Risco</h1>
            <p>Console de análise de crédito para novos cartões</p>
          </div>
        </div>
      </header>

      <main className="app-main">
        <ClientForm onSubmit={handleSubmit} carregando={carregando} />
        <ResultPanel resultado={resultado} erro={erro} carregando={carregando} />
      </main>

      <footer className="app-footer">
        <p>
          Radar de Risco — ferramenta interna de apoio à decisão. Os
          resultados são uma recomendação baseada em modelo estatístico e
          não substituem a análise de um especialista.
        </p>
      </footer>
    </div>
  );
}
