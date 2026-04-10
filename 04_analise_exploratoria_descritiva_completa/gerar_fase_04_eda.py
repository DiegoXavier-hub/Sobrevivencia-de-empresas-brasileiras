from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = Path.home() / "AppData" / "Local" / "Temp" / "duckdb_cnpj_fase_04"
GRAFICOS_DIR = ROOT / "04_analise_exploratoria_descritiva_completa" / "graficos"
RESULTADOS_DIR = ROOT / "04_analise_exploratoria_descritiva_completa" / "resultados"
SURVIVAL_PATH = ROOT / "02_preprocessamento" / "dados_saida" / "survival_empresa.parquet"
RADAR_TESTE_PATH = ROOT / "02_preprocessamento" / "dados_saida" / "teste_radar_12m.parquet"
REPORT_PATH = ROOT / "04_analise_exploratoria_descritiva_completa" / "RELATORIO_FASE_04_EDA.md"
SUMMARY_PATH = RESULTADOS_DIR / "00_resumo_fase_04.json"

AGE_BUCKET_ORDER = ["00-11m", "12-23m", "24-59m", "05-09a", "10-19a", "20a+"]
SIMPLE_MEI_ORDER = ["Simples+MEI", "Simples sem MEI", "Nao Simples/Nao MEI", "Nao Simples + MEI"]
REGIME_ORDER = ["Sem cobertura", "Lucro presumido", "Lucro real", "Imune/isenta", "Outros com cobertura"]


def normalize_numbers(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: normalize_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_numbers(v) for v in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def build_con() -> duckdb.DuckDBPyConnection:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA temp_directory='{TMP_DIR.as_posix()}'")
    con.execute("PRAGMA threads=4")
    con.execute("SET preserve_insertion_order=false")
    return con


def br_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def save_df(df: pd.DataFrame, name: str) -> None:
    df.to_csv(RESULTADOS_DIR / name, index=False, encoding="utf-8")


def style_plot() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 160


def annotate_barh(ax: plt.Axes, fmt: str = "{:.1%}") -> None:
    for patch in ax.patches:
        value = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax.text(value, y, " " + fmt.format(value), va="center", ha="left", fontsize=8)


def annotate_bar(ax: plt.Axes, fmt: str = "{:.1%}") -> None:
    for patch in ax.patches:
        value = patch.get_height()
        x = patch.get_x() + patch.get_width() / 2
        ax.text(x, value, fmt.format(value), va="bottom", ha="center", fontsize=8)


def plot_age_hist(df: pd.DataFrame) -> None:
    order_map = {label: i for i, label in enumerate(AGE_BUCKET_ORDER)}
    plot_df = df.copy()
    plot_df["ord"] = plot_df["faixa_idade"].map(order_map)
    plot_df = plot_df.sort_values("ord")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=plot_df, x="faixa_idade", y="share", color="#2a6f97", ax=ax)
    ax.set_title("Distribuicao de Idade Empresarial")
    ax.set_xlabel("Faixa de idade")
    ax.set_ylabel("Share das empresas")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    annotate_bar(ax)
    fig.tight_layout()
    fig.savefig(GRAFICOS_DIR / "graf_01_idade_empresas_hist.png")
    plt.close(fig)


def plot_event_share(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df["share"] = plot_df["n"] / plot_df["n"].sum()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    sns.barplot(data=plot_df, x="grupo", y="share", hue="grupo", palette=["#c0392b", "#247ba0"], legend=False, ax=ax)
    ax.set_title("Proporcao Observada de Baixa no Snapshot")
    ax.set_xlabel("")
    ax.set_ylabel("Share")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    annotate_bar(ax)
    plt.xticks(rotation=10)
    fig.tight_layout()
    fig.savefig(GRAFICOS_DIR / "graf_02_evento_baixa_bar.png")
    plt.close(fig)


def plot_cohort_rate(df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.bar(df["coorte_abertura_ano"], df["n"], color="#d9d9d9")
    ax1.set_ylabel("Numero de empresas")
    ax1.set_xlabel("Coorte de abertura")
    ax1.tick_params(axis="x", rotation=45)
    ax2 = ax1.twinx()
    ax2.plot(df["coorte_abertura_ano"], df["taxa_baixa"], color="#c0392b", marker="o", linewidth=2.0)
    ax2.set_ylabel("Taxa de baixa observada")
    ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax1.set_title("Coortes de Abertura: Volume e Taxa Observada de Baixa")
    fig.tight_layout()
    fig.savefig(GRAFICOS_DIR / "graf_03_coorte_volume_taxa_baixa.png")
    plt.close(fig)


def plot_horizontal_rate(df: pd.DataFrame, y_col: str, rate_col: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, max(4.5, len(df) * 0.36)))
    sns.barplot(data=df, y=y_col, x=rate_col, color="#c0392b", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Taxa")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    annotate_barh(ax)
    fig.tight_layout()
    fig.savefig(GRAFICOS_DIR / filename)
    plt.close(fig)


def plot_horizontal_volume(df: pd.DataFrame, y_col: str, n_col: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, max(4.5, len(df) * 0.36)))
    sns.barplot(data=df, y=y_col, x=n_col, color="#2a6f97", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Numero de empresas")
    ax.set_ylabel("")
    for patch in ax.patches:
        value = int(patch.get_width())
        y = patch.get_y() + patch.get_height() / 2
        ax.text(value, y, f" {value:,}".replace(",", "."), va="center", ha="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(GRAFICOS_DIR / filename)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame) -> None:
    pivot = df.pivot(index="regiao", columns="porte", values="taxa_baixa")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1%",
        cmap="Reds",
        ax=ax,
        cbar_kws={"format": plt.matplotlib.ticker.PercentFormatter(1.0)},
    )
    ax.set_title("Taxa Observada de Baixa por Regiao e Porte")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(GRAFICOS_DIR / "graf_08_heatmap_regiao_porte_taxa_baixa.png")
    plt.close(fig)


def plot_capital_box(df: pd.DataFrame) -> None:
    stats = []
    for _, row in df.iterrows():
        stats.append(
            {
                "label": row["grupo"],
                "whislo": row["q05"],
                "q1": row["q25"],
                "med": row["q50"],
                "q3": row["q75"],
                "whishi": row["q95"],
                "fliers": [],
            }
        )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bxp(
        stats,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#8ecae6"},
        medianprops={"color": "#c0392b", "linewidth": 2},
    )
    ax.set_title("Capital Social por Grupo do Evento")
    ax.set_ylabel("log10(capital_social + 1)")
    fig.tight_layout()
    fig.savefig(GRAFICOS_DIR / "graf_09_capital_social_evento_boxplot.png")
    plt.close(fig)


def plot_natureza(df: pd.DataFrame) -> None:
    plot_df = df.sort_values("n", ascending=True)
    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ax1.barh(plot_df["natureza"], plot_df["n"], color="#2a6f97")
    ax1.set_xlabel("Numero de empresas")
    ax1.set_ylabel("")
    ax2 = ax1.twiny()
    ax2.plot(plot_df["taxa_baixa"], plot_df["natureza"], color="#c0392b", marker="o")
    ax2.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax2.set_xlabel("Taxa de baixa observada")
    ax1.set_title("Naturezas Juridicas de Maior Volume")
    fig.tight_layout()
    fig.savefig(GRAFICOS_DIR / "graf_10_natureza_volume_taxa.png")
    plt.close(fig)


def plot_ordered_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    order: list[str],
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    order_map = {label: i for i, label in enumerate(order)}
    plot_df = df.copy()
    plot_df["ord"] = plot_df[x_col].map(order_map)
    plot_df = plot_df.sort_values("ord")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=plot_df, x=x_col, y=y_col, color="#c0392b", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    annotate_bar(ax)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(GRAFICOS_DIR / filename)
    plt.close(fig)


def plot_regime_risk(df: pd.DataFrame) -> None:
    order_map = {label: i for i, label in enumerate(REGIME_ORDER)}
    plot_df = df.copy()
    plot_df["ord"] = plot_df["grupo_regime"].map(order_map)
    plot_df = plot_df.sort_values("ord")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sns.barplot(data=plot_df, x="grupo_regime", y="taxa_12m", color="#264653", ax=ax)
    ax.set_title("Split de Teste: Taxa de Baixa em 12 Meses por Regime Observado")
    ax.set_xlabel("")
    ax.set_ylabel("Taxa de baixa em 12 meses")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    annotate_bar(ax)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(GRAFICOS_DIR / "graf_13_radar_teste_regime_taxa12m.png")
    plt.close(fig)


def build_report(tables: dict[str, pd.DataFrame]) -> None:
    total_n = int(tables["evento_baixa"]["n"].sum())
    event_count = int(tables["evento_baixa"].loc[tables["evento_baixa"]["grupo"] == "Baixada", "n"].iloc[0])
    total_event_rate = event_count / total_n
    median_months = 57.337061
    top_cnae = tables["cnae_div_volume"].iloc[0]
    top_uf = tables["uf_taxa"].iloc[0]
    top_age_risk = tables["radar_idade_teste"].sort_values("taxa_12m", ascending=False).iloc[0]
    top_simples = tables["radar_simples_mei_teste"].sort_values("taxa_12m", ascending=False).iloc[0]
    top_profile = tables["perfis_risco_teste"].iloc[0]
    low_profile = tables["perfis_risco_teste"].sort_values("taxa_12m", ascending=True).iloc[0]
    outliers = tables["outliers_datas_antigas"].set_index("faixa_data_inicio")["n"].to_dict()

    high_profiles_rows = [
        [row["faixa_idade"], row["grupo_simples_mei"], row["grupo_cobertura_regime"], row["n"], row["taxa_12m"]]
        for _, row in tables["perfis_risco_teste"].head(10).iterrows()
    ]
    low_profiles_rows = [
        [row["faixa_idade"], row["grupo_simples_mei"], row["grupo_cobertura_regime"], row["n"], row["taxa_12m"]]
        for _, row in tables["perfis_risco_teste"].sort_values("taxa_12m", ascending=True).head(10).iterrows()
    ]

    lines = [
        "# Fase 04 - Analise Exploratoria e Descritiva",
        "",
        "Esta fase descreve a estrutura observada da base empresarial e, separadamente, o risco de curto prazo medido no split de teste do radar 12M.",
        "",
        "## Regras de leitura da EDA",
        "",
        "- `survival_empresa` foi usada para descrever estrutura observada da carteira: idade, evento de baixa acumulado no snapshot, distribuicao setorial e territorial.",
        "- `teste_radar_12m` foi usado para descrever risco de curto prazo em 12 meses com variaveis observaveis na data de previsao.",
        "- Nao interpretar as taxas desta fase como relacoes causais; elas sao descritivas.",
        "",
        "## Resumo executivo",
        "",
        f"- A carteira analitica tem {br_int(total_n)} empresas.",
        f"- A taxa observada de baixa no snapshot e {total_event_rate:.1%}.",
        f"- A mediana de tempo observada na base e {median_months:.1f} meses.",
        f"- A divisao CNAE de maior volume e `{top_cnae['cnae_div']}`, com {br_int(int(top_cnae['n']))} empresas.",
        f"- A UF com maior taxa observada de baixa foi `{top_uf['uf']}` ({top_uf['taxa_baixa']:.1%}).",
        f"- No split de teste do radar, a faixa etaria de maior risco 12M foi `{top_age_risk['faixa_idade']}` ({top_age_risk['taxa_12m']:.1%}).",
        f"- O grupo `Simples/MEI` de maior risco 12M foi `{top_simples['grupo']}` ({top_simples['taxa_12m']:.1%}).",
        "",
        "## Achados principais",
        "",
        "### 1. Estrutura geral",
        "",
        "- O portfolio e concentrado em microempresas, e isso precisa ser lembrado em toda comparacao entre grupos.",
        "- A distribuicao de idade e assimetrica: ha grande massa de empresas relativamente jovens, mas tambem uma cauda longa de empresas muito antigas.",
        "",
        "### 2. Setor e territorio",
        "",
        "- O varejo (`CNAE divisao 47`) domina em volume.",
        "- As taxas observadas de baixa variam de forma relevante por CNAE, UF e municipio, mas sempre devem ser lidas junto com o volume minimo do grupo.",
        "- O heatmap por regiao e porte mostra que o porte muda muito a leitura territorial do risco observado.",
        "- Algumas naturezas juridicas especiais ou transitorias, como candidaturas eleitorais e formas descontinuadas, podem inflar comparacoes de taxa e exigem leitura contextual.",
        "",
        "### 3. Curto prazo no radar 12M",
        "",
        "- No split de teste, empresas mais novas concentram risco de baixa muito maior no horizonte de 12 meses.",
        "- `Simples+MEI` aparece com risco elevado de curto prazo na fotografia temporal do teste, o que sugere forte composicao por empresas jovens e sensiveis, nao causalidade simples.",
        "- Empresas com cobertura em `Regimes Tributarios` parecem menos arriscadas no curto prazo, mas essa leitura sofre forte selecao por cobertura.",
        "",
        "### 4. Outliers historicos",
        "",
        f"- Ha {br_int(int(outliers.get('1900-1949', 0)))} empresas com inicio entre 1900 e 1949 e {br_int(int(outliers.get('Antes de 1900', 0)))} anteriores a 1900.",
        "- Esses casos devem ser tratados como outliers historicos a acompanhar nas fases seguintes, nao como erro automaticamente removivel.",
        "",
        "## Perfis de maior risco aparente no teste 12M",
        "",
        md_table(["Faixa de idade", "Grupo Simples/MEI", "Cobertura regime", "Volume", "Taxa 12m"], high_profiles_rows),
        "",
        "## Perfis de menor risco aparente no teste 12M",
        "",
        md_table(["Faixa de idade", "Grupo Simples/MEI", "Cobertura regime", "Volume", "Taxa 12m"], low_profiles_rows),
        "",
        "## Cuidado interpretativo",
        "",
        f"- O perfil de maior risco aparente observado foi `{top_profile['faixa_idade']} | {top_profile['grupo_simples_mei']} | {top_profile['grupo_cobertura_regime']}` ({top_profile['taxa_12m']:.1%}).",
        f"- O perfil de menor risco aparente observado foi `{low_profile['faixa_idade']} | {low_profile['grupo_simples_mei']} | {low_profile['grupo_cobertura_regime']}` ({low_profile['taxa_12m']:.1%}).",
        "- Esses perfis sao descritivos e misturam composicao, selecao e calendario. Eles nao devem ser lidos como causalidade nem como regra de negocio pronta.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_summary(tables: dict[str, pd.DataFrame]) -> None:
    total_n = int(tables["evento_baixa"]["n"].sum())
    event_count = int(tables["evento_baixa"].loc[tables["evento_baixa"]["grupo"] == "Baixada", "n"].iloc[0])
    summary = {
        "overview": {
            "survival_rows": total_n,
            "event_rate_snapshot": event_count / total_n,
        },
        "top_cnae_div_volume": normalize_numbers(tables["cnae_div_volume"].head(5).to_dict(orient="records")),
        "top_uf_taxa": normalize_numbers(tables["uf_taxa"].head(5).to_dict(orient="records")),
        "top_municipio_taxa": normalize_numbers(tables["municipio_taxa"].head(10).to_dict(orient="records")),
        "radar_test_top_profiles": normalize_numbers(tables["perfis_risco_teste"].head(10).to_dict(orient="records")),
        "outliers_datas_antigas": normalize_numbers(tables["outliers_datas_antigas"].to_dict(orient="records")),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    style_plot()
    con = build_con()
    survival = SURVIVAL_PATH.resolve().as_posix()
    radar_teste = RADAR_TESTE_PATH.resolve().as_posix()

    sql_map = {
        "idade_hist": f"""
            WITH base AS (
                SELECT
                    CASE
                        WHEN tempo_em_meses < 12 THEN '00-11m'
                        WHEN tempo_em_meses < 24 THEN '12-23m'
                        WHEN tempo_em_meses < 60 THEN '24-59m'
                        WHEN tempo_em_meses < 120 THEN '05-09a'
                        WHEN tempo_em_meses < 240 THEN '10-19a'
                        ELSE '20a+'
                    END AS faixa_idade
                FROM parquet_scan('{survival}')
            )
            SELECT faixa_idade, COUNT(*) AS n, ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (), 6) AS share
            FROM base GROUP BY 1
        """,
        "evento_baixa": f"""
            SELECT CASE WHEN evento_baixa = 1 THEN 'Baixada' ELSE 'Censurada/Ativa no snapshot' END AS grupo, COUNT(*) AS n
            FROM parquet_scan('{survival}') GROUP BY 1 ORDER BY n DESC
        """,
        "coorte_taxa": f"""
            SELECT coorte_abertura_ano, COUNT(*) AS n, ROUND(AVG(CAST(evento_baixa AS DOUBLE)), 6) AS taxa_baixa
            FROM parquet_scan('{survival}')
            WHERE coorte_abertura_ano BETWEEN 1990 AND 2025
            GROUP BY 1 HAVING COUNT(*) >= 1000 ORDER BY 1
        """,
        "cnae_div_volume": f"""
            SELECT SUBSTR(cnae_fiscal_principal_snapshot, 1, 2) AS cnae_div, COUNT(*) AS n, ROUND(AVG(CAST(evento_baixa AS DOUBLE)), 6) AS taxa_baixa
            FROM parquet_scan('{survival}') GROUP BY 1 ORDER BY n DESC LIMIT 15
        """,
        "cnae_div_risco": f"""
            SELECT SUBSTR(cnae_fiscal_principal_snapshot, 1, 2) AS cnae_div, COUNT(*) AS n, ROUND(AVG(CAST(evento_baixa AS DOUBLE)), 6) AS taxa_baixa
            FROM parquet_scan('{survival}') GROUP BY 1 HAVING COUNT(*) >= 100000 ORDER BY taxa_baixa DESC LIMIT 15
        """,
        "uf_taxa": f"""
            SELECT uf_snapshot AS uf, COUNT(*) AS n, ROUND(AVG(CAST(evento_baixa AS DOUBLE)), 6) AS taxa_baixa
            FROM parquet_scan('{survival}') GROUP BY 1 ORDER BY taxa_baixa DESC, n DESC
        """,
        "municipio_taxa": f"""
            SELECT municipio_desc_snapshot AS municipio, uf_snapshot AS uf, COUNT(*) AS n, ROUND(AVG(CAST(evento_baixa AS DOUBLE)), 6) AS taxa_baixa
            FROM parquet_scan('{survival}')
            GROUP BY 1, 2 HAVING COUNT(*) >= 50000
            ORDER BY taxa_baixa DESC, n DESC LIMIT 20
        """,
        "regiao_porte_heatmap": f"""
            SELECT regiao_snapshot AS regiao, porte_desc_snapshot AS porte, COUNT(*) AS n, ROUND(AVG(CAST(evento_baixa AS DOUBLE)), 6) AS taxa_baixa
            FROM parquet_scan('{survival}')
            WHERE regiao_snapshot IS NOT NULL AND porte_desc_snapshot <> 'PORTE_NAO_MAPEADO'
            GROUP BY 1, 2
        """,
        "capital_box": f"""
            SELECT
                CASE WHEN evento_baixa = 1 THEN 'Baixada' ELSE 'Ativa/Censurada' END AS grupo,
                approx_quantile(log10(capital_social_snapshot + 1), 0.05) AS q05,
                approx_quantile(log10(capital_social_snapshot + 1), 0.25) AS q25,
                approx_quantile(log10(capital_social_snapshot + 1), 0.50) AS q50,
                approx_quantile(log10(capital_social_snapshot + 1), 0.75) AS q75,
                approx_quantile(log10(capital_social_snapshot + 1), 0.95) AS q95
            FROM parquet_scan('{survival}')
            WHERE capital_social_snapshot IS NOT NULL AND capital_social_snapshot >= 0
            GROUP BY 1 ORDER BY grupo
        """,
        "natureza_volume": f"""
            SELECT natureza_juridica_desc_snapshot AS natureza, COUNT(*) AS n, ROUND(AVG(CAST(evento_baixa AS DOUBLE)), 6) AS taxa_baixa
            FROM parquet_scan('{survival}')
            GROUP BY 1 HAVING COUNT(*) >= 50000 ORDER BY n DESC LIMIT 12
        """,
        "porte_taxa": f"""
            SELECT porte_desc_snapshot AS porte, COUNT(*) AS n, ROUND(AVG(CAST(evento_baixa AS DOUBLE)), 6) AS taxa_baixa
            FROM parquet_scan('{survival}')
            WHERE porte_desc_snapshot <> 'PORTE_NAO_MAPEADO'
            GROUP BY 1 ORDER BY n DESC
        """,
        "radar_idade_teste": f"""
            WITH base AS (
                SELECT
                    CASE
                        WHEN idade_empresa_meses < 12 THEN '00-11m'
                        WHEN idade_empresa_meses < 24 THEN '12-23m'
                        WHEN idade_empresa_meses < 60 THEN '24-59m'
                        WHEN idade_empresa_meses < 120 THEN '05-09a'
                        WHEN idade_empresa_meses < 240 THEN '10-19a'
                        ELSE '20a+'
                    END AS faixa_idade,
                    y_baixa_12m
                FROM parquet_scan('{radar_teste}')
            )
            SELECT faixa_idade, COUNT(*) AS n, ROUND(AVG(CAST(y_baixa_12m AS DOUBLE)), 6) AS taxa_12m
            FROM base GROUP BY 1
        """,
        "radar_simples_mei_teste": f"""
            SELECT
                CASE
                    WHEN flag_optante_simples_t AND flag_optante_mei_t THEN 'Simples+MEI'
                    WHEN flag_optante_simples_t AND NOT flag_optante_mei_t THEN 'Simples sem MEI'
                    WHEN NOT flag_optante_simples_t AND NOT flag_optante_mei_t THEN 'Nao Simples/Nao MEI'
                    ELSE 'Nao Simples + MEI'
                END AS grupo,
                COUNT(*) AS n,
                ROUND(AVG(CAST(y_baixa_12m AS DOUBLE)), 6) AS taxa_12m
            FROM parquet_scan('{radar_teste}') GROUP BY 1
        """,
        "radar_regime_teste": f"""
            SELECT
                CASE
                    WHEN NOT flag_tem_regime_t THEN 'Sem cobertura'
                    WHEN flag_lucro_presumido_t THEN 'Lucro presumido'
                    WHEN flag_lucro_real_t THEN 'Lucro real'
                    WHEN flag_imune_ou_isenta_t THEN 'Imune/isenta'
                    ELSE 'Outros com cobertura'
                END AS grupo_regime,
                COUNT(*) AS n,
                ROUND(AVG(CAST(y_baixa_12m AS DOUBLE)), 6) AS taxa_12m
            FROM parquet_scan('{radar_teste}') GROUP BY 1
        """,
        "perfis_risco_teste": f"""
            WITH base AS (
                SELECT
                    CASE
                        WHEN idade_empresa_meses < 12 THEN '00-11m'
                        WHEN idade_empresa_meses < 24 THEN '12-23m'
                        WHEN idade_empresa_meses < 60 THEN '24-59m'
                        WHEN idade_empresa_meses < 120 THEN '05-09a'
                        WHEN idade_empresa_meses < 240 THEN '10-19a'
                        ELSE '20a+'
                    END AS faixa_idade,
                    CASE
                        WHEN flag_optante_simples_t AND flag_optante_mei_t THEN 'Simples+MEI'
                        WHEN flag_optante_simples_t AND NOT flag_optante_mei_t THEN 'Simples sem MEI'
                        WHEN NOT flag_optante_simples_t AND NOT flag_optante_mei_t THEN 'Nao Simples/Nao MEI'
                        ELSE 'Nao Simples + MEI'
                    END AS grupo_simples_mei,
                    CASE WHEN NOT flag_tem_regime_t THEN 'Sem cobertura' ELSE 'Com cobertura' END AS grupo_cobertura_regime,
                    y_baixa_12m
                FROM parquet_scan('{radar_teste}')
            )
            SELECT faixa_idade, grupo_simples_mei, grupo_cobertura_regime, COUNT(*) AS n, ROUND(AVG(CAST(y_baixa_12m AS DOUBLE)), 6) AS taxa_12m
            FROM base GROUP BY 1, 2, 3 HAVING COUNT(*) >= 100000 ORDER BY taxa_12m DESC, n DESC
        """,
        "outliers_datas_antigas": f"""
            SELECT
                CASE
                    WHEN data_inicio_atividade < DATE '1900-01-01' THEN 'Antes de 1900'
                    WHEN data_inicio_atividade < DATE '1950-01-01' THEN '1900-1949'
                    ELSE '1950+'
                END AS faixa_data_inicio,
                COUNT(*) AS n
            FROM parquet_scan('{survival}')
            GROUP BY 1 ORDER BY faixa_data_inicio
        """,
    }

    tables = {name: con.execute(sql).fetchdf() for name, sql in sql_map.items()}

    for name, df in tables.items():
        save_df(df, f"{name}.csv")

    plot_age_hist(tables["idade_hist"])
    plot_event_share(tables["evento_baixa"])
    plot_cohort_rate(tables["coorte_taxa"])
    plot_horizontal_volume(tables["cnae_div_volume"].sort_values("n"), "cnae_div", "n", "Top Divisoes CNAE por Volume", "graf_04_cnae_div_volume.png")
    plot_horizontal_rate(tables["cnae_div_risco"], "cnae_div", "taxa_baixa", "Top Divisoes CNAE por Taxa Observada de Baixa", "graf_05_cnae_div_taxa_baixa.png")
    plot_horizontal_rate(tables["uf_taxa"], "uf", "taxa_baixa", "UFs por Taxa Observada de Baixa", "graf_06_uf_taxa_baixa.png")
    plot_horizontal_rate(
        tables["municipio_taxa"].assign(label=lambda d: d["municipio"] + " - " + d["uf"]),
        "label",
        "taxa_baixa",
        "Municipios com Base Minima por Taxa Observada de Baixa",
        "graf_07_municipio_taxa_baixa.png",
    )
    plot_heatmap(tables["regiao_porte_heatmap"])
    plot_capital_box(tables["capital_box"])
    plot_natureza(tables["natureza_volume"])
    plot_horizontal_rate(tables["porte_taxa"], "porte", "taxa_baixa", "Porte por Taxa Observada de Baixa", "graf_11_porte_taxa_baixa.png")
    plot_ordered_bar(
        tables["radar_idade_teste"],
        "faixa_idade",
        "taxa_12m",
        AGE_BUCKET_ORDER,
        "Split de Teste: Taxa de Baixa em 12 Meses por Faixa de Idade",
        "Taxa de baixa em 12 meses",
        "graf_12_radar_teste_idade_taxa12m.png",
    )
    plot_ordered_bar(
        tables["radar_simples_mei_teste"],
        "grupo",
        "taxa_12m",
        SIMPLE_MEI_ORDER,
        "Split de Teste: Taxa de Baixa em 12 Meses por Simples/MEI",
        "Taxa de baixa em 12 meses",
        "graf_13_radar_teste_simples_mei_taxa12m.png",
    )
    plot_regime_risk(tables["radar_regime_teste"])

    build_summary(tables)
    build_report(tables)


if __name__ == "__main__":
    main()
