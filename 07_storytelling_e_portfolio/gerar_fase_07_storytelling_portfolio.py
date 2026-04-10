from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.offline import get_plotlyjs


ROOT = Path(__file__).resolve().parents[1]
PHASE_DIR = ROOT / "07_storytelling_e_portfolio"
RESULTS_DIR = PHASE_DIR / "resultados"
DASHBOARD_DIR = PHASE_DIR / "dashboard"
CASE_DIR = PHASE_DIR / "case_portfolio"

PHASE05_SUMMARY = ROOT / "05_analise_sobrevivencia" / "resultados" / "00_resumo_fase_05.json"
PHASE06_SUMMARY = ROOT / "06_modelagem_radar_risco" / "resultados" / "00_resumo_fase_06.json"

KM_HORIZONS = ROOT / "05_analise_sobrevivencia" / "resultados" / "km_horizontes_geral.csv"
CAUSES_10Y = ROOT / "05_analise_sobrevivencia" / "resultados" / "competing_risks_horizontes.csv"
VALID_MODELS = ROOT / "06_modelagem_radar_risco" / "resultados" / "comparacao_modelos_validacao.csv"
TEST_MODELS = ROOT / "06_modelagem_radar_risco" / "resultados" / "comparacao_modelos_teste.csv"
CALIBRATION_CURVE = ROOT / "06_modelagem_radar_risco" / "resultados" / "curva_calibracao_teste.csv"
BANDS = ROOT / "06_modelagem_radar_risco" / "resultados" / "bandas_risco_teste.csv"
STABILITY_UF = ROOT / "06_modelagem_radar_risco" / "resultados" / "estabilidade_uf_teste.csv"
STABILITY_CNAE = ROOT / "06_modelagem_radar_risco" / "resultados" / "estabilidade_cnae_teste.csv"
GLOBAL_IMPORTANCE = ROOT / "06_modelagem_radar_risco" / "resultados" / "shap_importancia_global.csv"
LOCAL_EXPLANATIONS = ROOT / "06_modelagem_radar_risco" / "resultados" / "shap_explicacoes_locais.csv"


def ascii_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").strip()


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    CASE_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_int_ptbr(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def metric_card(title: str, value: str, subtitle: str = "") -> str:
    subtitle_html = f"<div class='metric-sub'>{subtitle}</div>" if subtitle else ""
    return f"""
    <div class="metric-card">
      <div class="metric-title">{title}</div>
      <div class="metric-value">{value}</div>
      {subtitle_html}
    </div>
    """


def figure_block(title: str, fig: go.Figure) -> str:
    html = to_html(fig, include_plotlyjs=False, full_html=False, config={"displayModeBar": False, "responsive": True})
    return f"""
    <section class="chart-card">
      <h3>{title}</h3>
      {html}
    </section>
    """


def build_dashboard_html(
    phase05: dict,
    phase06: dict,
    km_horizons: pd.DataFrame,
    causes_10y: pd.DataFrame,
    valid_models: pd.DataFrame,
    test_models: pd.DataFrame,
    calibration_curve: pd.DataFrame,
    bands: pd.DataFrame,
    stability_uf: pd.DataFrame,
    stability_cnae: pd.DataFrame,
    global_importance: pd.DataFrame,
    local_explanations: pd.DataFrame,
) -> str:
    km_plot = km_horizons[km_horizons["followup_suficiente"]].copy()
    km_plot["horizonte_label"] = km_plot["horizonte"].map(
        {
            "1_ano": "1 ano",
            "3_anos": "3 anos",
            "5_anos": "5 anos",
            "10_anos": "10 anos",
        }
    )
    fig_survival = px.line(
        km_plot,
        x="horizonte_label",
        y="sobrevivencia_km",
        markers=True,
        title="Sobrevivencia acumulada estimada",
        labels={"horizonte_label": "Horizonte", "sobrevivencia_km": "Sobrevivencia"},
    )
    fig_survival.update_layout(template="plotly_white", height=360)

    causes_top = causes_10y[(causes_10y["horizonte"] == "10_anos") & (causes_10y["followup_suficiente"])].copy()
    causes_top = causes_top.sort_values("cif", ascending=True).tail(6)
    fig_causes = px.bar(
        causes_top,
        x="cif",
        y="causa",
        orientation="h",
        title="Principais motivos de baixa em 10 anos",
        labels={"cif": "Incidencia acumulada", "causa": "Motivo"},
    )
    fig_causes.update_layout(template="plotly_white", height=380)

    compare_models = pd.concat([valid_models.assign(split="Validacao"), test_models.assign(split="Teste")], ignore_index=True)
    fig_models = px.bar(
        compare_models,
        x="modelo",
        y="pr_auc",
        color="split",
        barmode="group",
        title="Comparacao de modelos por PR-AUC",
        labels={"pr_auc": "PR-AUC", "modelo": "Modelo"},
    )
    fig_models.update_layout(template="plotly_white", height=380)

    fig_cal = go.Figure()
    for label, sub in calibration_curve.groupby("modelo"):
        ordered = sub.sort_values("bin")
        fig_cal.add_trace(
            go.Scatter(
                x=ordered["score_medio_bin"],
                y=ordered["taxa_real_bin"],
                mode="lines+markers",
                name=label,
            )
        )
    fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="ideal", line=dict(dash="dash", color="#888")))
    fig_cal.update_layout(
        template="plotly_white",
        height=360,
        title="Calibracao do score no teste",
        xaxis_title="Probabilidade prevista media",
        yaxis_title="Taxa observada",
    )

    fig_bands = px.bar(
        bands.sort_values("score_medio"),
        x="banda_risco",
        y="taxa_evento",
        color="banda_risco",
        title="Taxa real de baixa por banda de risco",
        labels={"banda_risco": "Banda", "taxa_evento": "Taxa de evento"},
    )
    fig_bands.update_layout(template="plotly_white", height=340, showlegend=False)

    fig_importance = px.bar(
        global_importance.head(10).sort_values("mean_abs_shap"),
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="Principais sinais do score",
        labels={"mean_abs_shap": "Importancia media", "feature": "Feature"},
    )
    fig_importance.update_layout(template="plotly_white", height=420)

    fig_uf = px.bar(
        stability_uf.head(10).sort_values("lift_alto_risco"),
        x="grupo",
        y="lift_alto_risco",
        title="Lift no alto risco por UF",
        labels={"grupo": "UF", "lift_alto_risco": "Lift"},
    )
    fig_uf.update_layout(template="plotly_white", height=360)

    fig_cnae = px.bar(
        stability_cnae.head(10).sort_values("lift_alto_risco"),
        x="grupo",
        y="lift_alto_risco",
        title="Lift no alto risco por divisao CNAE",
        labels={"grupo": "CNAE divisao", "lift_alto_risco": "Lift"},
    )
    fig_cnae.update_layout(template="plotly_white", height=360)

    local_html_rows = []
    for row in local_explanations.head(5).itertuples(index=False):
        local_html_rows.append(
            f"""
            <tr>
              <td>{row.cnpj_basico}</td>
              <td>{row.score_calibrado:.3f}</td>
              <td>{ascii_text(row.top1_feature)} / {ascii_text(row.top2_feature)} / {ascii_text(row.top3_feature)}</td>
            </tr>
            """
        )

    model_name = ascii_text(phase06["final_model"]["modelo"])
    calibration_name = ascii_text(phase06["final_model"]["calibracao"])
    threshold = phase06["final_model"]["threshold_high"]
    plotly_js_bundle = get_plotlyjs()

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Radar de Risco de Baixa de Empresas Brasileiras</title>
  <script>{plotly_js_bundle}</script>
  <style>
    :root {{
      --bg: #f4efe8;
      --panel: #fffaf2;
      --ink: #1f2a2e;
      --muted: #5c6a70;
      --accent: #0b6e4f;
      --line: #d7cbb8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: radial-gradient(circle at top, #fff7eb 0%, var(--bg) 55%, #ece2d4 100%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1360px;
      margin: 0 auto;
      padding: 28px 24px 56px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(11,110,79,0.95), rgba(23,49,77,0.92));
      color: #fffef8;
      border-radius: 26px;
      padding: 30px 34px;
      box-shadow: 0 20px 50px rgba(20, 30, 40, 0.18);
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 40px;
      line-height: 1.05;
    }}
    .hero p {{
      max-width: 900px;
      margin: 0;
      font-size: 18px;
      line-height: 1.55;
      color: rgba(255,255,255,0.92);
    }}
    .pill-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    .pill {{
      border: 1px solid rgba(255,255,255,0.25);
      border-radius: 999px;
      padding: 8px 14px;
      font-size: 14px;
      letter-spacing: 0.02em;
      background: rgba(255,255,255,0.08);
    }}
    .section-title {{
      margin: 34px 0 12px;
      font-size: 28px;
      color: #17314d;
    }}
    .section-copy {{
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 17px;
      line-height: 1.6;
      max-width: 980px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: 20px;
    }}
    .metric-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px 18px 16px;
      box-shadow: 0 8px 24px rgba(60, 45, 25, 0.06);
    }}
    .metric-title {{
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .metric-value {{
      margin-top: 8px;
      font-size: 34px;
      font-weight: 700;
      color: var(--accent);
    }}
    .metric-sub {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 10px;
    }}
    .chart-card, .narrative-card, .table-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px 20px;
      box-shadow: 0 8px 24px rgba(60, 45, 25, 0.06);
    }}
    .chart-card h3, .narrative-card h3, .table-card h3 {{
      margin: 0 0 10px;
      font-size: 20px;
      color: #17314d;
    }}
    .narrative-card p, .narrative-card li {{
      color: var(--muted);
      line-height: 1.65;
      font-size: 16px;
    }}
    .narrative-card ul {{
      margin: 10px 0 0 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
    }}
    th {{
      color: #17314d;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .footer-note {{
      margin-top: 24px;
      font-size: 14px;
      color: var(--muted);
      line-height: 1.6;
    }}
    @media (max-width: 1040px) {{
      .metrics, .grid-2 {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 760px) {{
      .metrics, .grid-2 {{ grid-template-columns: 1fr; }}
      .hero h1 {{ font-size: 30px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Radar de Risco de Baixa de Empresas Brasileiras</h1>
      <p>
        Case de Data Science ponta a ponta usando Dados Abertos do CNPJ para responder duas perguntas de negocio:
        quanto tempo uma empresa sobrevive e quais sinais ajudam a antecipar risco de baixa em 12 meses sem leakage temporal.
      </p>
      <div class="pill-row">
        <div class="pill">Snapshot empresarial: 67,2 mi de empresas</div>
        <div class="pill">Modelo final: {model_name}</div>
        <div class="pill">Calibracao: {calibration_name}</div>
        <div class="pill">Threshold alto risco: {threshold:.4f}</div>
      </div>
    </section>

    <h2 class="section-title">Panorama</h2>
    <p class="section-copy">
      O portfolio foi desenhado para ser legivel por recrutadores e gestores: problema, desenho temporal, resultados e limitacoes ficam no mesmo lugar.
      O dashboard separa o que e leitura estrutural do cadastro do que e score preditivo de curto prazo.
    </p>
    <div class="metrics">
      {metric_card("Empresas na sobrevivencia", format_int_ptbr(phase05["overview"]["n_empresas"]), "Base estrutural usada nas curvas de sobrevivencia.")}
      {metric_card("Mediana de sobrevivencia", f"{phase05['overview']['mediana_km_anos']:.2f} anos", "Estimativa Kaplan-Meier da duracao ate baixa ou censura.")}
      {metric_card("PR-AUC no teste", f"{phase06['test_metrics']['pr_auc']:.4f}", "Capacidade do radar de concentrar eventos em um contexto desbalanceado.")}
      {metric_card("Lift no top 10%", f"{phase06['test_metrics']['lift_top_10']:.2f}x", "Quanto o decil de maior risco supera a taxa base do teste.")}
    </div>

    <div class="grid-2">
      {figure_block("Sobrevivencia ao longo do tempo", fig_survival)}
      {figure_block("Principais motivos de baixa", fig_causes)}
    </div>

    <h2 class="section-title">Radar 12M</h2>
    <p class="section-copy">
      O score final foi escolhido por validacao temporal, calibrado fora da selecao e convertido em bandas de risco com threshold operacional.
      O objetivo nao e maximizar uma metrica isolada, mas entregar um radar defensavel para uso real.
    </p>
    <div class="metrics">
      {metric_card("ROC-AUC", f"{phase06['test_metrics']['roc_auc']:.4f}", "Discriminacao geral do modelo final no teste.")}
      {metric_card("Brier", f"{phase06['test_metrics']['brier']:.4f}", "Qualidade probabilistica do score calibrado.")}
      {metric_card("Precision alto risco", f"{phase06['test_metrics']['precision_alto_risco']:.3f}", "Taxa de evento dentro da faixa de alto risco.")}
      {metric_card("Recall alto risco", f"{phase06['test_metrics']['recall_alto_risco']:.3f}", "Quanto dos eventos do teste foi capturado pela faixa de alto risco.")}
    </div>
    <div class="grid-2">
      {figure_block("Comparacao de modelos", fig_models)}
      {figure_block("Curva de calibracao", fig_cal)}
      {figure_block("Bandas de risco", fig_bands)}
      {figure_block("Principais sinais do score", fig_importance)}
    </div>

    <h2 class="section-title">Estabilidade</h2>
    <p class="section-copy">
      O projeto nao encerra na metrica agregada. O radar foi auditado por UF, divisao CNAE e cobertura da fonte tributaria para evitar uma narrativa falsa de desempenho uniforme.
    </p>
    <div class="grid-2">
      {figure_block("Lift do alto risco por UF", fig_uf)}
      {figure_block("Lift do alto risco por CNAE", fig_cnae)}
    </div>

    <h2 class="section-title">Narrativa do Case</h2>
    <div class="grid-2">
      <section class="narrative-card">
        <h3>O que este case prova</h3>
        <ul>
          <li>Capacidade de estruturar um projeto grande em fases auditaveis, do dado bruto ao score final.</li>
          <li>Controle de leakage temporal e separacao entre exploracao, sobrevivencia e modelagem preditiva.</li>
          <li>Capacidade de escolher um modelo por criterio defensavel, e nao por metricas isoladas.</li>
          <li>Disciplina para documentar problemas metodologicos encontrados no caminho e corrigi-los.</li>
        </ul>
      </section>
      <section class="narrative-card">
        <h3>Limitacoes que ficam explicitas</h3>
        <ul>
          <li>As features do radar sao propositalmente enxutas para preservar validade temporal.</li>
          <li>A cobertura de `Regimes Tributarios` e baixa e gera comportamento desigual por subgrupo.</li>
          <li>O score e preditivo, nao causal.</li>
          <li>Alguns recortes institucionais extremos, como certas naturezas juridicas, exigem leitura contextual.</li>
        </ul>
      </section>
    </div>

    <section class="table-card" style="margin-top: 18px;">
      <h3>Exemplos de explicacao local</h3>
      <table>
        <thead>
          <tr>
            <th>CNPJ basico</th>
            <th>Score</th>
            <th>Top 3 sinais</th>
          </tr>
        </thead>
        <tbody>
          {''.join(local_html_rows)}
        </tbody>
      </table>
    </section>

    <p class="footer-note">
      Este dashboard resume a versao de portfolio do projeto. Ele foi construido para comunicar valor rapido a recrutadores e gestores sem esconder as limitacoes do dado:
      snapshot cadastral, cobertura tributaria parcial e associacoes preditivas em vez de causalidade.
    </p>
  </div>
</body>
</html>
"""


def build_recruiter_case(phase05: dict, phase06: dict) -> str:
    return f"""# Case Resumido Para Recrutador

## Problema

Construir um case de Data Science ponta a ponta com Dados Abertos do CNPJ para responder:

- quanto tempo empresas brasileiras sobrevivem
- quais sinais ajudam a antecipar baixa em 12 meses

## O que foi feito

- Estruturei a base bruta em fases auditaveis.
- Construi uma base de sobrevivencia com {format_int_ptbr(phase05['overview']['n_empresas'])} empresas.
- Modelei um radar 12M com split temporal real entre treino, validacao e teste.
- Comparei baselines, modelos de arvore e regressao logistica.
- Calibrei o score final e defini threshold operacional.

## Resultado principal

- Modelo final: `{ascii_text(phase06['final_model']['modelo'])}`
- PR-AUC no teste: `{phase06['test_metrics']['pr_auc']:.4f}`
- ROC-AUC no teste: `{phase06['test_metrics']['roc_auc']:.4f}`
- Lift top 10% no teste: `{phase06['test_metrics']['lift_top_10']:.2f}x`

## Por que esse case e forte

- Tem desenho temporal defensavel.
- Registra e corrige erros metodologicos encontrados no caminho.
- Separa exploracao, sobrevivencia e previsao.
- Traduz resultado tecnico em score utilizavel.

## Limitacoes

- Score preditivo, nao causal.
- Features propositalmente enxutas para evitar leakage.
- Cobertura de `Regimes Tributarios` ainda parcial.
"""


def build_technical_case(phase05: dict, phase06: dict) -> str:
    return f"""# Case Tecnico Para Entrevista

## 1. Contexto

O projeto usa Dados Abertos do CNPJ para atacar dois problemas complementares:

1. sobrevivencia empresarial
2. classificacao de risco de baixa em 12 meses

## 2. Desenho do dado

- Snapshot bruto principal: `2026-03`
- Unidade principal: empresa aproximada pela matriz
- Evento de sobrevivencia: `situacao_cadastral = 08`
- Censura: `2026-03-15`
- Target do radar: `y_baixa_12m`

## 3. Controles metodologicos

- Remocao de colunas `_snapshot` do radar para evitar leakage
- Split temporal absoluto
- Auditoria de drift e cobertura antes da modelagem
- Amostragem deterministica por hash para reproducibilidade
- Separacao de selecao, calibracao e threshold tuning

## 4. Sobrevivencia

- Mediana Kaplan-Meier: `{phase05['overview']['mediana_km_anos']:.2f}` anos
- Sobrevivencia em 10 anos: `57,2%`
- Cox exploratorio com restricoes metodologicas
- Riscos competitivos por motivo de baixa

## 5. Modelagem do radar

- Baselines: heuristico, baseline temporal, regressao logistica
- Modelos principais: Random Forest, XGBoost, LightGBM
- Regra final de selecao: tolerancia de empate em PR-AUC e desempate por Brier/log-loss/simplicidade
- Modelo final: `{ascii_text(phase06['final_model']['modelo'])}`
- Calibracao: `{ascii_text(phase06['final_model']['calibracao'])}`

## 6. Resultado final no teste

- ROC-AUC: `{phase06['test_metrics']['roc_auc']:.4f}`
- PR-AUC: `{phase06['test_metrics']['pr_auc']:.4f}`
- Brier: `{phase06['test_metrics']['brier']:.4f}`
- Threshold alto risco: `{phase06['final_model']['threshold_high']:.4f}`
- Precision alto risco: `{phase06['test_metrics']['precision_alto_risco']:.3f}`
- Recall alto risco: `{phase06['test_metrics']['recall_alto_risco']:.3f}`

## 7. O que eu destacaria em entrevista

- Os erros metodologicos nao foram escondidos; eles viraram parte da trilha de auditoria.
- O modelo final nao foi escolhido por "vencer" marginalmente uma metrica isolada.
- O projeto fecha em score calibrado, estabilidade e explicabilidade, nao apenas AUC.
"""


def build_executive_narrative(phase05: dict, phase06: dict) -> str:
    return f"""# Narrativa Executiva

Empresas brasileiras nao falham de forma aleatoria. A analise de sobrevivencia mostrou mediana de vida de aproximadamente {phase05['overview']['mediana_km_anos']:.1f} anos, com forte concentracao de risco nos primeiros anos de operacao. A modelagem preditiva transformou esse padrao em um radar 12M capaz de concentrar mais de {phase06['test_metrics']['lift_top_10']:.1f} vezes a taxa base no decil de maior risco.

O principal valor deste projeto nao esta apenas na metrica. Ele esta no desenho: base tratada fase a fase, controle de leakage temporal, criterio auditado de selecao do modelo, calibracao do score e traducao do resultado tecnico para um produto legivel por negocio.

O radar nao deve ser lido como causalidade. Ele e um instrumento de priorizacao. Seu melhor uso seria apoiar triagem, monitoramento e segmentacao de empresas que merecem acompanhamento mais proximo, desde que suas limitacoes de cobertura e escopo sejam explicitadas.
"""


def build_phase_report(phase05: dict, phase06: dict, dashboard_path: Path) -> str:
    return f"""# Fase 07 - Storytelling e Portfolio

Esta fase transforma o projeto tecnico em entregaveis legiveis para recrutadores, gestores e entrevistas tecnicas.

## O que foi produzido

- Dashboard HTML single-file com Plotly embarcado em `{dashboard_path.name}`
- Versao resumida para recrutador
- Versao tecnica para entrevista
- Narrativa executiva do case

## Mensagem central do portfolio

O projeto mostra capacidade de:

- estruturar uma base massiva e ruidosa
- controlar leakage temporal
- combinar sobrevivencia e modelagem preditiva
- escolher um modelo final por criterio defensavel
- comunicar limitacoes sem enfraquecer o case

## Elementos do case que mais importam

- sobrevivencia mediana de {phase05['overview']['mediana_km_anos']:.2f} anos
- modelo final `{ascii_text(phase06['final_model']['modelo'])}`
- PR-AUC de {phase06['test_metrics']['pr_auc']:.4f} no teste
- lift de {phase06['test_metrics']['lift_top_10']:.2f}x no top 10%

## Uso recomendado

- recrutador: ler a versao curta e abrir o dashboard
- entrevista tecnica: usar a versao tecnica e os relatorios das fases
- portfolio final: usar o dashboard como vitrine e os documentos como apoio
"""


def build_methodology_review() -> str:
    return """# Fase 07 - Revisao Metodologica

## Parecer geral

A Fase 7 ficou aprovada. O projeto deixou de ser apenas uma colecao de etapas tecnicas e passou a ter uma camada clara de comunicacao para portfolio.

## Problemas metodologicos identificados e como foram tratados

### 1. Risco de vender o dashboard como se fosse um produto operacional pronto

**Problema**

Um dashboard de portfolio pode facilmente ser lido como sistema produtivo terminado.

**Tratamento**

- O material foi apresentado como camada de storytelling e demonstracao.
- Limitacoes de cobertura, causalidade e escopo ficaram explicitas no dashboard e nos textos.

### 2. Risco de misturar sobrevivencia estrutural com score preditivo

**Problema**

Sem cuidado narrativo, o portfolio poderia fundir evento acumulado no snapshot com risco futuro de 12 meses.

**Tratamento**

- O dashboard separa bloco estrutural e bloco do radar.
- A narrativa executiva preserva a mesma separacao metodologica das fases anteriores.

### 3. Risco de mostrar apenas metricas e esconder o processo

**Problema**

Isso faria o projeto parecer um exercicio escolar de modelagem.

**Tratamento**

- O case enfatiza desenho temporal, auditoria, calibracao e trade-offs de selecao do modelo final.

### 4. Risco de inconsistencia tecnica no empacotamento final

**Problema**

Na primeira versao, o dashboard dependia de CDN externo do Plotly e um dos textos saiu com ruido de encoding.

**Tratamento**

- O HTML final passou a incorporar o bundle do Plotly inline, ficando realmente single-file.
- Os textos do case foram normalizados em ASCII.
- O resumo da fase passou a registrar o artefato por caminho relativo, evitando ruido desnecessario do path absoluto do Windows.

## Conclusao

A Fase 7 entrega uma camada de comunicacao coerente com o rigor tecnico acumulado nas fases anteriores. O ganho principal foi transformar o projeto em um case de portfolio legivel sem diluir as limitacoes metodologicas reais.
"""


def main() -> None:
    ensure_dirs()

    phase05 = load_json(PHASE05_SUMMARY)
    phase06 = load_json(PHASE06_SUMMARY)

    km_horizons = pd.read_csv(KM_HORIZONS)
    causes_10y = pd.read_csv(CAUSES_10Y)
    valid_models = pd.read_csv(VALID_MODELS)
    test_models = pd.read_csv(TEST_MODELS)
    calibration_curve = pd.read_csv(CALIBRATION_CURVE)
    bands = pd.read_csv(BANDS)
    stability_uf = pd.read_csv(STABILITY_UF)
    stability_cnae = pd.read_csv(STABILITY_CNAE)
    global_importance = pd.read_csv(GLOBAL_IMPORTANCE)
    local_explanations = pd.read_csv(LOCAL_EXPLANATIONS)

    dashboard_html = build_dashboard_html(
        phase05=phase05,
        phase06=phase06,
        km_horizons=km_horizons,
        causes_10y=causes_10y,
        valid_models=valid_models,
        test_models=test_models,
        calibration_curve=calibration_curve,
        bands=bands,
        stability_uf=stability_uf,
        stability_cnae=stability_cnae,
        global_importance=global_importance,
        local_explanations=local_explanations,
    )

    dashboard_path = DASHBOARD_DIR / "dashboard_portfolio_cnpj.html"
    dashboard_path.write_text(dashboard_html, encoding="utf-8")

    recruiter_case = build_recruiter_case(phase05, phase06)
    technical_case = build_technical_case(phase05, phase06)
    executive_narrative = build_executive_narrative(phase05, phase06)
    phase_report = build_phase_report(phase05, phase06, dashboard_path)
    methodology_review = build_methodology_review()

    (CASE_DIR / "CASE_RECRUTADOR.md").write_text(recruiter_case, encoding="utf-8")
    (CASE_DIR / "CASE_TECNICO_ENTREVISTA.md").write_text(technical_case, encoding="utf-8")
    (CASE_DIR / "NARRATIVA_EXECUTIVA.md").write_text(executive_narrative, encoding="utf-8")
    (PHASE_DIR / "RELATORIO_FASE_07_STORYTELLING_PORTFOLIO.md").write_text(phase_report, encoding="utf-8")
    (PHASE_DIR / "RELATORIO_FASE_07_REVISAO_METODOLOGICA.md").write_text(methodology_review, encoding="utf-8")

    summary = {
        "dashboard_artifact": "dashboard/dashboard_portfolio_cnpj.html",
        "dashboard_file": dashboard_path.name,
        "final_model": phase06["final_model"]["modelo"],
        "test_pr_auc": phase06["test_metrics"]["pr_auc"],
        "test_lift_top_10": phase06["test_metrics"]["lift_top_10"],
        "survival_median_years": phase05["overview"]["mediana_km_anos"],
        "artifacts": [
            "dashboard/dashboard_portfolio_cnpj.html",
            "case_portfolio/CASE_RECRUTADOR.md",
            "case_portfolio/CASE_TECNICO_ENTREVISTA.md",
            "case_portfolio/NARRATIVA_EXECUTIVA.md",
            "RELATORIO_FASE_07_STORYTELLING_PORTFOLIO.md",
            "RELATORIO_FASE_07_REVISAO_METODOLOGICA.md",
        ],
    }
    (RESULTS_DIR / "00_resumo_fase_07.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
