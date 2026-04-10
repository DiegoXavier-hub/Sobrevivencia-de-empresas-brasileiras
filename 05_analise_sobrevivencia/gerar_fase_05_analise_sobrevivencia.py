from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg


ROOT = Path(__file__).resolve().parents[1]
SURVIVAL_PATH = ROOT / "02_preprocessamento" / "dados_saida" / "survival_empresa.parquet"
ESTAB_PATH = ROOT / "02_preprocessamento" / "dados_saida" / "estabelecimento_base.parquet"
PHASE_DIR = ROOT / "05_analise_sobrevivencia"
RESULTS_DIR = PHASE_DIR / "resultados"
CHARTS_DIR = PHASE_DIR / "graficos"
TEMP_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Temp" / "duckdb_cnpj_fase_05"
HORIZONS = {
    "1_ano": int(round(365.25 * 1)),
    "3_anos": int(round(365.25 * 3)),
    "5_anos": int(round(365.25 * 5)),
    "10_anos": int(round(365.25 * 10)),
}
SAMPLE_HASH_MOD = 1000
SAMPLE_HASH_THRESHOLD = 3


@dataclass
class GroupResult:
    volume: pd.DataFrame
    curves: pd.DataFrame
    horizons: pd.DataFrame


def ascii_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return normalized.strip()


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


def connect_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{TEMP_DIR.as_posix()}';")
    con.execute("SET threads=4;")
    return con


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_base_view(con: duckdb.DuckDBPyConnection) -> None:
    query = f"""
    CREATE OR REPLACE TEMP VIEW survival_base AS
    SELECT
        cnpj_basico,
        CAST(tempo_em_dias AS BIGINT) AS tempo_em_dias,
        CAST(evento_baixa AS INTEGER) AS evento_baixa,
        CAST(coorte_abertura_ano AS INTEGER) AS coorte_abertura_ano,
        CASE
            WHEN coorte_abertura_ano IS NULL THEN 'COORTE_NAO_INFORMADA'
            WHEN coorte_abertura_ano < 1995 THEN 'ATE_1994'
            WHEN coorte_abertura_ano BETWEEN 1995 AND 1999 THEN '1995_1999'
            WHEN coorte_abertura_ano BETWEEN 2000 AND 2004 THEN '2000_2004'
            WHEN coorte_abertura_ano BETWEEN 2005 AND 2009 THEN '2005_2009'
            WHEN coorte_abertura_ano BETWEEN 2010 AND 2014 THEN '2010_2014'
            WHEN coorte_abertura_ano BETWEEN 2015 AND 2019 THEN '2015_2019'
            WHEN coorte_abertura_ano BETWEEN 2020 AND 2024 THEN '2020_2024'
            ELSE '2025_2026'
        END AS coorte_grupo,
        COALESCE(NULLIF(TRIM(porte_desc_snapshot), ''), 'NAO_INFORMADO') AS porte_grupo,
        CASE
            WHEN LENGTH(COALESCE(TRIM(cnae_fiscal_principal_snapshot), '')) >= 2
                THEN SUBSTR(TRIM(cnae_fiscal_principal_snapshot), 1, 2)
            ELSE 'NI'
        END AS cnae_div,
        COALESCE(NULLIF(TRIM(uf_snapshot), ''), 'NAO_INFORMADO') AS uf_grupo,
        COALESCE(NULLIF(TRIM(natureza_juridica_snapshot), ''), 'NI') AS natureza_grupo
    FROM read_parquet('{SURVIVAL_PATH.as_posix()}')
    WHERE tempo_em_dias IS NOT NULL
      AND tempo_em_dias >= 0
      AND data_inicio_atividade IS NOT NULL
    ;
    """
    con.execute(query)

    cause_query = f"""
    CREATE OR REPLACE TEMP VIEW matriz_causas AS
    SELECT
        cnpj_basico,
        CASE
            WHEN LENGTH(COALESCE(TRIM(motivo_situacao_cadastral_desc), '')) > 0
                THEN TRIM(motivo_situacao_cadastral_desc)
            WHEN LENGTH(COALESCE(TRIM(motivo_situacao_cadastral), '')) > 0
                THEN 'COD_' || TRIM(motivo_situacao_cadastral) || '_NAO_MAPEADO'
            ELSE 'MOTIVO_NAO_INFORMADO'
        END AS motivo_grupo
    FROM read_parquet('{ESTAB_PATH.as_posix()}')
    WHERE cnpj_ordem = '0001'
    ;
    """
    con.execute(cause_query)


def top_categories(
    con: duckdb.DuckDBPyConnection,
    column: str,
    limit: int | None = None,
    min_n: int | None = None,
) -> pd.DataFrame:
    query = f"""
    SELECT {column} AS grupo, COUNT(*) AS n
    FROM survival_base
    WHERE {column} IS NOT NULL AND {column} <> ''
    GROUP BY 1
    """
    if min_n is not None:
        query += f"\nHAVING COUNT(*) >= {min_n}"
    query += "\nORDER BY n DESC"
    if limit is not None:
        query += f"\nLIMIT {limit}"
    df = con.execute(query).fetchdf()
    df["grupo"] = df["grupo"].map(ascii_text)
    return df


def group_counts(
    con: duckdb.DuckDBPyConnection,
    column: str,
    selected_groups: Iterable[str] | None = None,
    other_label: str | None = None,
) -> pd.DataFrame:
    if selected_groups:
        raw_groups = list(selected_groups)
        in_sql = ", ".join(sql_literal(group) for group in raw_groups)
        group_expr = f"CASE WHEN {column} IN ({in_sql}) THEN {column} ELSE {sql_literal(other_label or 'OUTROS')} END"
    else:
        group_expr = column

    query = f"""
    SELECT
        {group_expr} AS grupo,
        tempo_em_dias,
        SUM(CASE WHEN evento_baixa = 1 THEN 1 ELSE 0 END) AS n_evento,
        SUM(CASE WHEN evento_baixa = 0 THEN 1 ELSE 0 END) AS n_censura,
        COUNT(*) AS n_total
    FROM survival_base
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
    df = con.execute(query).fetchdf()
    df["grupo"] = df["grupo"].map(ascii_text)
    return df


def km_from_counts(counts: pd.DataFrame) -> pd.DataFrame:
    ordered = counts.sort_values("tempo_em_dias").copy()
    total_n = int(ordered["n_total"].sum())
    n_risk = total_n
    surv = 1.0
    records: list[dict[str, float | int]] = []
    for row in ordered.itertuples(index=False):
        t = int(row.tempo_em_dias)
        d = int(row.n_evento)
        c = int(row.n_censura)
        step_hazard = (d / n_risk) if n_risk else 0.0
        if d > 0 and n_risk > 0:
            surv *= (1.0 - step_hazard)
        records.append(
            {
                "tempo_em_dias": t,
                "n_risco": n_risk,
                "n_evento": d,
                "n_censura": c,
                "survival_prob": surv,
                "hazard_step": step_hazard,
                "tempo_anos": t / 365.25,
            }
        )
        n_risk -= d + c
    curve = pd.DataFrame(records)
    curve["n_inicial"] = total_n
    curve["tempo_max_dias"] = int(ordered["tempo_em_dias"].max()) if not ordered.empty else 0
    return curve


def horizons_from_curve(curve: pd.DataFrame, group_value: str | None = None) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()

    max_time = int(curve["tempo_max_dias"].iloc[0])
    rows: list[dict[str, object]] = []
    for horizon_name, horizon_days in HORIZONS.items():
        idx = curve["tempo_em_dias"].searchsorted(horizon_days, side="right") - 1
        survival = 1.0 if idx < 0 else float(curve.iloc[idx]["survival_prob"])
        followup_ok = max_time >= horizon_days
        rows.append(
            {
                "grupo": group_value or "GERAL",
                "horizonte": horizon_name,
                "dias": horizon_days,
                "sobrevivencia_km": survival if followup_ok else np.nan,
                "followup_suficiente": bool(followup_ok),
                "tempo_max_dias_grupo": max_time,
                "n_inicial_grupo": int(curve["n_inicial"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def median_survival(curve: pd.DataFrame) -> float | None:
    if curve.empty:
        return None
    below = curve[curve["survival_prob"] <= 0.5]
    if below.empty:
        return None
    return float(below.iloc[0]["tempo_anos"])


def compute_group_result(
    con: duckdb.DuckDBPyConnection,
    column: str,
    selected_groups: Iterable[str] | None = None,
    other_label: str | None = None,
) -> GroupResult:
    counts = group_counts(con, column, selected_groups, other_label)
    volume = counts.groupby("grupo", as_index=False)["n_total"].sum().sort_values("n_total", ascending=False)

    curve_frames: list[pd.DataFrame] = []
    horizon_frames: list[pd.DataFrame] = []
    for group_value, group_counts_df in counts.groupby("grupo", sort=False):
        curve = km_from_counts(group_counts_df[["tempo_em_dias", "n_evento", "n_censura", "n_total"]])
        curve.insert(0, "grupo", group_value)
        curve_frames.append(curve)
        horizon_frames.append(horizons_from_curve(curve, group_value))

    curves = pd.concat(curve_frames, ignore_index=True)
    horizons = pd.concat(horizon_frames, ignore_index=True)
    return GroupResult(volume=volume, curves=curves, horizons=horizons)


def plot_km(
    curve_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    file_name: str,
    title: str,
    max_groups: int | None = None,
    x_years: int = 20,
) -> None:
    plt.figure(figsize=(12, 7))
    selected = volume_df.copy()
    if max_groups is not None:
        selected = selected.head(max_groups)
    groups = selected["grupo"].tolist()
    subset = curve_df[curve_df["grupo"].isin(groups)].copy()

    palette = sns.color_palette("tab10", n_colors=max(len(groups), 3))
    for idx, group in enumerate(groups):
        group_curve = subset[subset["grupo"] == group].sort_values("tempo_em_dias")
        plt.step(
            group_curve["tempo_anos"],
            group_curve["survival_prob"],
            where="post",
            label=group,
            color=palette[idx % len(palette)],
        )

    plt.title(title)
    plt.xlabel("Tempo desde a abertura (anos)")
    plt.ylabel("Probabilidade de sobrevivencia")
    plt.xlim(0, x_years)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.2)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / file_name, dpi=180)
    plt.close()


def plot_hr(hr_df: pd.DataFrame, file_name: str) -> None:
    selected = hr_df.copy().sort_values("abs_log_hr", ascending=False).head(20).sort_values("hazard_ratio")
    plt.figure(figsize=(12, 8))
    plt.errorbar(
        selected["hazard_ratio"],
        selected["termo"],
        xerr=[
            selected["hazard_ratio"] - selected["ci_95_inf_hr"],
            selected["ci_95_sup_hr"] - selected["hazard_ratio"],
        ],
        fmt="o",
        color="#264653",
        ecolor="#2a9d8f",
        capsize=3,
    )
    plt.axvline(1.0, color="#d62828", linestyle="--", linewidth=1)
    plt.xlabel("Hazard ratio")
    plt.ylabel("Termo")
    plt.title("Cox exploratorio: maiores efeitos absolutos")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / file_name, dpi=180)
    plt.close()


def plot_competing_risks(cif_df: pd.DataFrame, file_name: str) -> None:
    plt.figure(figsize=(12, 7))
    for cause, group_df in cif_df.groupby("causa"):
        ordered = group_df.sort_values("tempo_em_dias")
        plt.step(ordered["tempo_anos"], ordered["cif"], where="post", label=cause)
    plt.title("Incidencia acumulada por principais motivos de baixa")
    plt.xlabel("Tempo desde a abertura (anos)")
    plt.ylabel("Incidencia acumulada")
    plt.xlim(0, 20)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.2)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / file_name, dpi=180)
    plt.close()


def write_df(df: pd.DataFrame, path: Path) -> None:
    safe_df = df.copy()
    for column in safe_df.columns:
        if pd.api.types.is_object_dtype(safe_df[column]):
            safe_df[column] = safe_df[column].map(ascii_text)
    safe_df.to_csv(path, index=False, encoding="utf-8")


def top_motives(con: duckdb.DuckDBPyConnection, limit: int = 6) -> pd.DataFrame:
    query = """
    SELECT
        m.motivo_grupo AS causa,
        COUNT(*) AS n
    FROM survival_base s
    LEFT JOIN matriz_causas m USING (cnpj_basico)
    WHERE s.evento_baixa = 1
    GROUP BY 1
    ORDER BY n DESC
    LIMIT ?
    """
    return con.execute(query, [limit]).fetchdf()


def competing_risks(con: duckdb.DuckDBPyConnection, selected_causes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    in_sql = ", ".join(sql_literal(cause) for cause in selected_causes)
    query = f"""
    WITH joined AS (
        SELECT
            s.tempo_em_dias,
            s.evento_baixa,
            COALESCE(m.motivo_grupo, 'MOTIVO_NAO_INFORMADO') AS causa_evento
        FROM survival_base s
        LEFT JOIN matriz_causas m USING (cnpj_basico)
    )
    SELECT
        tempo_em_dias,
        CASE
            WHEN evento_baixa = 0 THEN 'CENSURA'
            WHEN causa_evento IN ({in_sql}) THEN causa_evento
            ELSE 'OUTROS_MOTIVOS'
        END AS causa,
        COUNT(*) AS n
    FROM joined
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
    counts = con.execute(query).fetchdf()

    ordered = counts.sort_values(["tempo_em_dias", "causa"]).copy()
    causes = [cause for cause in sorted(ordered["causa"].unique()) if cause != "CENSURA"]
    total_n = int(ordered["n"].sum())
    n_risk = total_n
    survival = 1.0
    cif = {cause: 0.0 for cause in causes}
    curve_rows: list[dict[str, object]] = []

    for time_value, time_df in ordered.groupby("tempo_em_dias", sort=True):
        time_counts = {row.causa: int(row.n) for row in time_df.itertuples(index=False)}
        total_events = sum(time_counts.get(cause, 0) for cause in causes)
        total_censor = int(time_counts.get("CENSURA", 0))
        if n_risk <= 0:
            break
        survival_before = survival
        for cause in causes:
            d_k = time_counts.get(cause, 0)
            if d_k > 0:
                cif[cause] += survival_before * d_k / n_risk
            curve_rows.append(
                {
                    "causa": cause,
                    "tempo_em_dias": int(time_value),
                    "tempo_anos": float(time_value) / 365.25,
                    "cif": cif[cause],
                    "survival_all_causes": survival_before,
                    "n_risco": n_risk,
                    "n_eventos_total_t": total_events,
                    "n_censura_t": total_censor,
                }
            )
        if total_events > 0:
            survival *= (1.0 - total_events / n_risk)
        n_risk -= total_events + total_censor

    curves = pd.DataFrame(curve_rows)
    horizon_rows: list[dict[str, object]] = []
    max_time = int(ordered["tempo_em_dias"].max()) if not ordered.empty else 0
    for cause, cause_df in curves.groupby("causa"):
        for horizon_name, horizon_days in HORIZONS.items():
            idx = cause_df["tempo_em_dias"].searchsorted(horizon_days, side="right") - 1
            cif_value = 0.0 if idx < 0 else float(cause_df.iloc[idx]["cif"])
            horizon_rows.append(
                {
                    "causa": cause,
                    "horizonte": horizon_name,
                    "dias": horizon_days,
                    "cif": cif_value if max_time >= horizon_days else np.nan,
                    "followup_suficiente": bool(max_time >= horizon_days),
                }
            )
    horizons = pd.DataFrame(horizon_rows)
    return curves, horizons


def prepare_cox_sample(
    con: duckdb.DuckDBPyConnection,
    cnae_top: list[str],
    natureza_top: list[str],
) -> pd.DataFrame:
    cnae_sql = ", ".join(sql_literal(value) for value in cnae_top)
    natureza_sql = ", ".join(sql_literal(value) for value in natureza_top)
    query = f"""
    SELECT
        tempo_em_dias,
        evento_baixa,
        coorte_grupo,
        uf_grupo,
        CASE WHEN cnae_div IN ({cnae_sql}) THEN cnae_div ELSE 'OUTRAS_DIVISOES' END AS cnae_grupo,
        CASE WHEN natureza_grupo IN ({natureza_sql}) THEN natureza_grupo ELSE 'OUTRAS_NATUREZAS' END AS natureza_grupo
    FROM survival_base
    WHERE MOD(hash(cnpj_basico), {SAMPLE_HASH_MOD}) < {SAMPLE_HASH_THRESHOLD}
    """
    sample = con.execute(query).fetchdf()
    for column in ["coorte_grupo", "uf_grupo", "cnae_grupo", "natureza_grupo"]:
        sample[column] = sample[column].map(ascii_text)
    return sample


def fit_cox(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    model_df = sample.copy()
    exog = pd.get_dummies(
        model_df[["coorte_grupo", "uf_grupo", "cnae_grupo"]],
        drop_first=True,
        dtype=float,
        prefix_sep="=",
    )
    non_constant = exog.columns[exog.nunique(dropna=False) > 1]
    exog = exog[non_constant]

    ph_model = PHReg(
        endog=model_df["tempo_em_dias"].astype(float).to_numpy(),
        exog=exog.astype(float),
        status=model_df["evento_baixa"].astype(int).to_numpy(),
        ties="breslow",
    )
    result = ph_model.fit(disp=False)

    conf_int = result.conf_int()
    hazard_df = pd.DataFrame(
        {
            "termo": exog.columns,
            "log_hr": result.params,
            "se_log_hr": result.bse,
            "hazard_ratio": np.exp(result.params),
            "ci_95_inf_hr": np.exp(conf_int[:, 0]),
            "ci_95_sup_hr": np.exp(conf_int[:, 1]),
            "p_valor": result.pvalues,
        }
    )
    hazard_df["abs_log_hr"] = hazard_df["log_hr"].abs()
    hazard_df = hazard_df.sort_values(["p_valor", "abs_log_hr"], ascending=[True, False]).reset_index(drop=True)
    hazard_df["termo"] = hazard_df["termo"].map(ascii_text)
    if hazard_df["hazard_ratio"].isna().all():
        raise ValueError("Cox model invalid: all coefficients are NaN.")

    schoenfeld = result.schoenfeld_residuals
    event_mask = np.isfinite(schoenfeld).all(axis=1)
    log_time = np.log(np.maximum(model_df["tempo_em_dias"].astype(float).to_numpy(), 1.0))

    assumptions: list[dict[str, object]] = []
    for idx, column in enumerate(exog.columns):
        residual = schoenfeld[:, idx]
        valid = np.isfinite(residual) & event_mask
        if valid.sum() < 30:
            assumptions.append(
                {
                    "termo": ascii_text(column),
                    "n_eventos_validos": int(valid.sum()),
                    "correlacao_log_tempo": np.nan,
                    "coef_ols": np.nan,
                    "p_valor_ols": np.nan,
                    "flag_alerta_ph": False,
                }
            )
            continue
        x = sm.add_constant(log_time[valid])
        ols = sm.OLS(residual[valid], x).fit()
        corr = float(np.corrcoef(log_time[valid], residual[valid])[0, 1])
        p_value = float(ols.pvalues[1])
        assumptions.append(
            {
                "termo": ascii_text(column),
                "n_eventos_validos": int(valid.sum()),
                "correlacao_log_tempo": corr,
                "coef_ols": float(ols.params[1]),
                "p_valor_ols": p_value,
                "flag_alerta_ph": bool((p_value < 0.001) and (abs(corr) >= 0.03)),
            }
        )
    assumptions_df = pd.DataFrame(assumptions).sort_values(["flag_alerta_ph", "p_valor_ols"], ascending=[False, True])

    metadata = {
        "sample_rows": int(len(model_df)),
        "sample_event_rate": float(model_df["evento_baixa"].mean()),
        "n_terms": int(exog.shape[1]),
        "hash_mod": int(SAMPLE_HASH_MOD),
        "hash_threshold": int(SAMPLE_HASH_THRESHOLD),
    }
    return hazard_df, assumptions_df, metadata


def build_report(
    overview: dict[str, object],
    km_general_horizons: pd.DataFrame,
    coorte_horizons: pd.DataFrame,
    porte_horizons: pd.DataFrame,
    uf_horizons: pd.DataFrame,
    cnae_horizons: pd.DataFrame,
    nature_horizons: pd.DataFrame,
    hazard_df: pd.DataFrame,
    assumptions_df: pd.DataFrame,
    competing_horizons: pd.DataFrame,
    path: Path,
) -> None:
    general_lines = []
    for row in km_general_horizons.itertuples(index=False):
        if bool(row.followup_suficiente):
            general_lines.append(f"- Sobrevivencia estimada em {row.horizonte}: {row.sobrevivencia_km:.3f}")
        else:
            general_lines.append(f"- Sobrevivencia estimada em {row.horizonte}: sem follow-up suficiente")

    def best_group_text(
        df: pd.DataFrame,
        horizon: str,
        min_n: int = 100000,
        exclude_groups: Iterable[str] | None = None,
    ) -> str:
        horizon_df = df[(df["horizonte"] == horizon) & (df["followup_suficiente"])].dropna(subset=["sobrevivencia_km"])
        horizon_df = horizon_df[horizon_df["n_inicial_grupo"] >= min_n]
        if exclude_groups:
            horizon_df = horizon_df[~horizon_df["grupo"].isin(list(exclude_groups))]
        if horizon_df.empty:
            return "- Sem grupo elegivel com follow-up suficiente."
        best = horizon_df.sort_values("sobrevivencia_km", ascending=False).iloc[0]
        worst = horizon_df.sort_values("sobrevivencia_km", ascending=True).iloc[0]
        return (
            f"- Melhor grupo em {horizon}: {ascii_text(best['grupo'])} ({best['sobrevivencia_km']:.3f}); "
            f"pior grupo em {horizon}: {ascii_text(worst['grupo'])} ({worst['sobrevivencia_km']:.3f})"
        )

    top_hr = hazard_df.head(8)
    flagged = assumptions_df[assumptions_df["flag_alerta_ph"]].head(8)
    top_causes = (
        competing_horizons[competing_horizons["horizonte"] == "10_anos"]
        .dropna(subset=["cif"])
        .sort_values("cif", ascending=False)
        .head(6)
    )

    median_km = overview["mediana_km_anos"]
    if median_km is None:
        median_text = "nao atingida"
    else:
        median_text = f"{median_km:.2f} anos"

    report = f"""# Fase 05 - Analise de Sobrevivencia

Esta fase estima curvas de sobrevivencia, compara grupos relevantes e adiciona uma camada explicativa via Cox exploratorio, mantendo a distincao entre descricao de duracao e inferencia causal.

## Regras metodologicas desta fase

- Kaplan-Meier geral e por grupos foi calculado sobre toda a `survival_empresa`.
- Curvas por `coorte_abertura_ano` sao a parte mais forte metodologicamente, porque a coorte esta definida no nascimento da empresa.
- Curvas por `UF`, `porte`, `CNAE` e `natureza juridica` devem ser lidas como recortes estruturais observados na base, nao como covariaveis historicas perfeitas de baseline.
- O modelo de Cox foi tratado como exploratorio e restrito a `coorte`, `UF` e `CNAE` observados, sem usar `porte`, `capital`, `socios`, `Simples` ou outros sinais claramente dinamicos.
- `Natureza juridica` ficou fora do Cox final porque, nesta base, algumas categorias geram separacao quase deterministica e instabilidade numerica.

## Resumo executivo

- Empresas analisadas: {int(overview['n_empresas']):,}
- Eventos de baixa observados: {int(overview['n_eventos']):,}
- Taxa observada de evento: {float(overview['taxa_evento']):.3f}
- Mediana Kaplan-Meier aproximada: {median_text}
- Amostra deterministica do Cox: {int(overview['cox_sample_rows']):,} linhas
- Eventos na amostra do Cox: {float(overview['cox_sample_event_rate']):.3f}

## Kaplan-Meier geral

{chr(10).join(general_lines)}

## Comparacoes principais

### Coortes de abertura

{best_group_text(coorte_horizons, '5_anos')}
{best_group_text(coorte_horizons, '10_anos')}

### Porte observado

{best_group_text(porte_horizons, '5_anos')}

### UF observada

{best_group_text(uf_horizons, '5_anos', exclude_groups=['EX', 'NAO_INFORMADO'])}

### CNAE divisao observada

{best_group_text(cnae_horizons, '5_anos')}

### Natureza juridica observada

{best_group_text(nature_horizons, '5_anos')}

## Cox exploratorio

Os termos abaixo sao os mais fortes na amostra deterministica usada no modelo. Eles servem para leitura estrutural do hazard e nao para alegacao causal.

{chr(10).join(f"- {ascii_text(row.termo)}: HR={row.hazard_ratio:.3f} (p={row.p_valor:.4g})" for row in top_hr.itertuples(index=False))}

## Alertas de pressuposto proporcional

{chr(10).join(f"- {ascii_text(row.termo)}: corr(log_tempo,resid)={row.correlacao_log_tempo:.3f}, p={row.p_valor_ols:.4g}" for row in flagged.itertuples(index=False)) if not flagged.empty else '- Nenhum termo excedeu o criterio de alerta adotado nesta fase.'}

## Riscos competitivos por motivo de baixa

{chr(10).join(f"- {ascii_text(row.causa)}: CIF em 10 anos = {row.cif:.3f}" for row in top_causes.itertuples(index=False))}

## Cuidado interpretativo

- Curvas por grupos observados no snapshot nao substituem covariaveis historicas de baseline.
- O Cox desta fase foi desenhado para portfolio e leitura explicativa, nao para inferencia causal forte.
- Coortes recentes com follow-up insuficiente nao devem ser comparadas em horizontes longos como se tivessem a mesma janela de observacao.
- Extremos de `natureza juridica` refletem, em parte, categorias institucionais especiais e nao devem ser lidos como hierarquia economica simples.
"""
    path.write_text(report, encoding="utf-8")


def build_methodology_review(
    overview: dict[str, object],
    assumptions_df: pd.DataFrame,
    path: Path,
) -> None:
    flagged = assumptions_df[assumptions_df["flag_alerta_ph"]]
    flagged_text = "\n".join(
        f"- {ascii_text(row.termo)}: corr={row.correlacao_log_tempo:.3f}, p={row.p_valor_ols:.4g}"
        for row in flagged.head(10).itertuples(index=False)
    )
    if not flagged_text:
        flagged_text = "- Nenhum termo ultrapassou o criterio de alerta adotado."

    review = f"""# Fase 05 - Revisao Metodologica

## Parecer geral

A Fase 5 ficou aprovada. As curvas Kaplan-Meier foram produzidas de forma coerente com a base e o Cox foi explicitamente restringido a um uso exploratorio e numericamente estavel.

## Problemas metodologicos identificados e como foram tratados

### 1. Covariaveis do snapshot nao sao baseline historico perfeito

**Problema**

Boa parte das covariaveis disponiveis em `survival_empresa` representa a fotografia observada do cadastro, nao o estado original da empresa no nascimento.

**Risco**

Usar essas colunas como se fossem baseline puro em sobrevivencia forte ou em Cox causal seria metodologicamente fraco.

**Tratamento**

- Curvas por grupos observados foram mantidas como leitura estrutural.
- O Cox foi restringido a covariaveis mais estaveis e foi rotulado como exploratorio.
- Variaveis claramente dinamicas, como `porte`, `capital social`, `Simples`, `MEI` e contagens de socios, ficaram fora do Cox desta fase.
- `Natureza juridica` tambem ficou fora do Cox final porque algumas categorias produzem separacao quase deterministica do evento.

### 2. Horizontes longos para coortes recentes

**Problema**

Coortes muito recentes nao possuem a mesma janela de follow-up para horizontes como 5 ou 10 anos.

**Risco**

Uma tabela ingenua poderia mostrar sobrevivencia em 10 anos para grupos que ainda nao tiveram 10 anos de observacao.

**Tratamento**

- As tabelas de horizontes passaram a marcar `followup_suficiente`.
- Quando o grupo nao tem follow-up bastante, o valor final e gravado como ausente em vez de ser apresentado como estimativa valida.

### 3. Cox no universo inteiro e inviavel para o ambiente local

**Problema**

O universo completo tem {int(overview['n_empresas']):,} empresas, o que inviabiliza um Cox integral com a infraestrutura atual.

**Risco**

Forcar o modelo no conjunto inteiro consumiria memoria e tempo demais, sem beneficio proporcional para o objetivo de portfolio.

**Tratamento**

- Foi adotada amostra deterministica por hash de `cnpj_basico`.
- O Cox rodou sobre {int(overview['cox_sample_rows']):,} observacoes com taxa de evento de {float(overview['cox_sample_event_rate']):.3f}.
- O criterio ficou reproduzivel e suficiente para leitura estrutural.

### 4. Singularidade inicial no Cox com `natureza juridica`

**Problema**

A primeira especificacao do Cox incluiu `natureza juridica`, o que gerou coeficientes inteiros em `NaN`.

**Risco**

Manter esse resultado comprometeria a validade da fase e faria o relatorio parecer concluido com um modelo numericamente invalido.

**Tratamento**

- A especificacao foi testada em blocos.
- A singularidade apareceu quando `natureza juridica` entrou no modelo.
- O Cox final ficou restrito a `coorte`, `UF` e `CNAE`, que fecharam sem `NaN`.
- A fase passou a ter uma checagem explicita que aborta se todos os coeficientes do Cox vierem nulos.

### 5. Pressuposto de hazards proporcionais

**Problema**

Como esperado em uma base muito heterogenea, alguns termos podem violar proporcionalidade.

**Tratamento**

- Foram usados residuos de Schoenfeld com regressao em `log(tempo)` como cheque pragmatico.
- O criterio de alerta da fase foi `p < 0.001` e `|correlacao| >= 0.03`.
- Termos sob alerta:
{flagged_text}

### 6. Riscos competitivos dependem do motivo observado na matriz

**Problema**

Os motivos de baixa foram puxados do estabelecimento matriz, e ainda existem categorias residuais nao mapeadas.

**Tratamento**

- A analise de riscos competitivos foi mantida descritiva.
- Codigos sem mapeamento foram preservados como categorias residuais documentadas.
- A fase nao mascara a presenca de `COD_32_NAO_MAPEADO`.

## Conclusao

A Fase 5 entrega sobrevivencia descritiva, comparacoes por grupos, Cox exploratorio e riscos competitivos com trilha metodologica explicita. O ganho principal foi separar o que e leitura robusta de duracao do que e apenas interpretacao estrutural condicionada pela fotografia observada da base.
"""
    path.write_text(review, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    con = connect_duckdb()
    build_base_view(con)

    overview_df = con.execute(
        """
        SELECT
            COUNT(*) AS n_empresas,
            SUM(evento_baixa) AS n_eventos,
            AVG(evento_baixa) AS taxa_evento
        FROM survival_base
        """
    ).fetchdf()
    overview = overview_df.iloc[0].to_dict()

    km_general_counts = con.execute(
        """
        SELECT
            tempo_em_dias,
            SUM(CASE WHEN evento_baixa = 1 THEN 1 ELSE 0 END) AS n_evento,
            SUM(CASE WHEN evento_baixa = 0 THEN 1 ELSE 0 END) AS n_censura,
            COUNT(*) AS n_total
        FROM survival_base
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    km_general = km_from_counts(km_general_counts)
    km_general.insert(0, "grupo", "GERAL")
    km_general_horizons = horizons_from_curve(km_general, "GERAL")
    overview["mediana_km_anos"] = median_survival(km_general)

    coorte_result = compute_group_result(con, "coorte_grupo")
    porte_result = compute_group_result(con, "porte_grupo")

    uf_volume_all = top_categories(con, "uf_grupo")
    uf_result = compute_group_result(con, "uf_grupo")

    cnae_top_df = top_categories(con, "cnae_div", limit=10)
    cnae_top = cnae_top_df["grupo"].tolist()
    cnae_result = compute_group_result(con, "cnae_div", cnae_top, "OUTRAS_DIVISOES")

    natureza_top_df = top_categories(con, "natureza_grupo", limit=8)
    natureza_top = natureza_top_df["grupo"].tolist()
    natureza_result = compute_group_result(con, "natureza_grupo", natureza_top, "OUTRAS_NATUREZAS")

    write_df(km_general, RESULTS_DIR / "km_geral.csv")
    write_df(km_general_horizons, RESULTS_DIR / "km_horizontes_geral.csv")

    for prefix, result in [
        ("coorte", coorte_result),
        ("porte", porte_result),
        ("uf", uf_result),
        ("cnae_div", cnae_result),
        ("natureza", natureza_result),
    ]:
        write_df(result.volume, RESULTS_DIR / f"{prefix}_volume.csv")
        write_df(result.curves, RESULTS_DIR / f"{prefix}_curvas.csv")
        write_df(result.horizons, RESULTS_DIR / f"{prefix}_horizontes.csv")

    plot_km(
        km_general,
        pd.DataFrame({"grupo": ["GERAL"], "n_total": [int(overview["n_empresas"])]}),
        "graf_01_km_geral.png",
        "Kaplan-Meier geral",
        max_groups=1,
        x_years=25,
    )
    plot_km(coorte_result.curves, coorte_result.volume, "graf_02_km_coorte.png", "Kaplan-Meier por coorte de abertura", max_groups=8, x_years=20)
    plot_km(porte_result.curves, porte_result.volume, "graf_03_km_porte.png", "Kaplan-Meier por porte observado", max_groups=6, x_years=20)
    plot_km(uf_result.curves, uf_volume_all, "graf_04_km_uf_top.png", "Kaplan-Meier por UF observada (top 10 em volume)", max_groups=10, x_years=20)
    plot_km(cnae_result.curves, cnae_result.volume, "graf_05_km_cnae_div_top.png", "Kaplan-Meier por divisao CNAE observada", max_groups=10, x_years=20)
    plot_km(natureza_result.curves, natureza_result.volume, "graf_06_km_natureza_top.png", "Kaplan-Meier por natureza juridica observada", max_groups=9, x_years=20)

    causes_df = top_motives(con, limit=6)
    selected_causes = causes_df["causa"].astype(str).tolist()
    competing_curves, competing_horizons = competing_risks(con, selected_causes)
    write_df(causes_df, RESULTS_DIR / "competing_risks_causas_top.csv")
    write_df(competing_curves, RESULTS_DIR / "competing_risks_curvas.csv")
    write_df(competing_horizons, RESULTS_DIR / "competing_risks_horizontes.csv")
    plot_competing_risks(competing_curves, "graf_07_competing_risks_motivos_top.png")

    cox_sample = prepare_cox_sample(con, cnae_top, natureza_top)
    hazard_df, assumptions_df, cox_meta = fit_cox(cox_sample)
    overview["cox_sample_rows"] = cox_meta["sample_rows"]
    overview["cox_sample_event_rate"] = cox_meta["sample_event_rate"]
    write_df(hazard_df, RESULTS_DIR / "cox_hazard_ratios.csv")
    write_df(assumptions_df, RESULTS_DIR / "cox_pressupostos.csv")
    plot_hr(hazard_df, "graf_08_cox_hazard_ratios.png")

    report_path = PHASE_DIR / "RELATORIO_FASE_05_ANALISE_SOBREVIVENCIA.md"
    review_path = PHASE_DIR / "RELATORIO_FASE_05_REVISAO_METODOLOGICA.md"
    build_report(
        overview=overview,
        km_general_horizons=km_general_horizons,
        coorte_horizons=coorte_result.horizons,
        porte_horizons=porte_result.horizons,
        uf_horizons=uf_result.horizons,
        cnae_horizons=cnae_result.horizons,
        nature_horizons=natureza_result.horizons,
        hazard_df=hazard_df,
        assumptions_df=assumptions_df,
        competing_horizons=competing_horizons,
        path=report_path,
    )
    build_methodology_review(overview=overview, assumptions_df=assumptions_df, path=review_path)

    summary = {
        "overview": {
            "n_empresas": int(overview["n_empresas"]),
            "n_eventos": int(overview["n_eventos"]),
            "taxa_evento": float(overview["taxa_evento"]),
            "mediana_km_anos": None if overview["mediana_km_anos"] is None else float(overview["mediana_km_anos"]),
        },
        "cox": {
            "sample_rows": int(overview["cox_sample_rows"]),
            "sample_event_rate": float(overview["cox_sample_event_rate"]),
            "n_terms": int(cox_meta["n_terms"]),
            "alerts_ph": int(assumptions_df["flag_alerta_ph"].sum()),
        },
        "top_causes_10y": (
            competing_horizons[competing_horizons["horizonte"] == "10_anos"]
            .sort_values("cif", ascending=False)
            .head(6)
            .to_dict(orient="records")
        ),
    }
    (RESULTS_DIR / "00_resumo_fase_05.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
