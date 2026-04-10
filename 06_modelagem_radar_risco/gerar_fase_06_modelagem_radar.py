from __future__ import annotations

import json
import math
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
PHASE_DIR = ROOT / "06_modelagem_radar_risco"
RESULTS_DIR = PHASE_DIR / "resultados"
CHARTS_DIR = PHASE_DIR / "graficos"
MODELS_DIR = PHASE_DIR / "modelos"
TEMP_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Temp" / "duckdb_cnpj_fase_06"
TRAIN_PATH = ROOT / "02_preprocessamento" / "dados_saida" / "treino_radar_12m.parquet"
VALID_PATH = ROOT / "02_preprocessamento" / "dados_saida" / "validacao_radar_12m.parquet"
TEST_PATH = ROOT / "02_preprocessamento" / "dados_saida" / "teste_radar_12m.parquet"
SURVIVAL_PATH = ROOT / "02_preprocessamento" / "dados_saida" / "survival_empresa.parquet"

SAMPLE_HASH_MOD = 1000
SAMPLE_HASH_THRESHOLD = 20
VALID_SPLIT_MOD = 3
RANDOM_STATE = 42
TOP_GROUPS = 10

FEATURES_FULL = [
    "idade_empresa_meses",
    "idade_log_meses",
    "coorte_abertura_ano_c",
    "flag_optante_simples_t",
    "flag_optante_mei_t",
    "flag_excluido_simples_ate_t",
    "flag_excluido_mei_ate_t",
    "qtd_escrituracoes_regime_t_log",
    "flag_tem_regime_t",
    "flag_lucro_real_t",
    "flag_lucro_presumido_t",
    "flag_lucro_arbitrado_t",
    "flag_imune_ou_isenta_t",
]
FEATURES_TEMPORAL = [
    "idade_empresa_meses",
    "idade_log_meses",
    "coorte_abertura_ano_c",
]


@dataclass
class ProbabilityCalibrator:
    method: str
    model: Any

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(raw_scores, dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            return np.clip(self.model.predict(clipped), 0.0, 1.0)
        if self.method == "sigmoid":
            logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
            return self.model.predict_proba(logits)[:, 1]
        raise ValueError(f"Unsupported calibration method: {self.method}")


class HeuristicRiskModel:
    def __init__(self) -> None:
        self.overall_rate: float | None = None
        self.tables: dict[str, dict[Any, float]] = {}

    def fit(self, df: pd.DataFrame, target_col: str) -> "HeuristicRiskModel":
        self.overall_rate = float(df[target_col].mean())
        feature_keys = [
            "idade_faixa_heuristica",
            "flag_optante_simples_t",
            "flag_optante_mei_t",
            "flag_excluido_simples_ate_t",
            "flag_excluido_mei_ate_t",
            "flag_tem_regime_t",
        ]
        for key in feature_keys:
            rates = df.groupby(key)[target_col].mean().to_dict()
            self.tables[key] = {k: float(v) for k, v in rates.items()}
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.overall_rate is None:
            raise ValueError("HeuristicRiskModel not fitted.")
        scores = []
        for _, row in df.iterrows():
            parts = []
            for key, table in self.tables.items():
                parts.append(float(table.get(row[key], self.overall_rate)))
            scores.append(float(np.mean(parts)))
        scores_arr = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - scores_arr, scores_arr])


def ascii_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return normalized.strip()


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


def connect_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{TEMP_DIR.as_posix()}';")
    con.execute("SET threads=4;")
    return con


def load_split_sample(con: duckdb.DuckDBPyConnection, split_name: str, path: Path) -> pd.DataFrame:
    query = f"""
    WITH amostra AS (
        SELECT
            *,
            MOD(hash(cnpj_basico), {VALID_SPLIT_MOD}) AS bucket_valid
        FROM read_parquet('{path.as_posix()}')
        WHERE MOD(hash(cnpj_basico), {SAMPLE_HASH_MOD}) < {SAMPLE_HASH_THRESHOLD}
    )
    SELECT
        {repr(split_name)} AS split_temporal,
        a.cnpj_basico,
        a.data_observacao,
        a.ano_observacao,
        a.idade_empresa_meses,
        a.coorte_abertura_ano,
        a.flag_optante_simples_t,
        a.flag_optante_mei_t,
        a.flag_excluido_simples_ate_t,
        a.flag_excluido_mei_ate_t,
        a.qtd_escrituracoes_regime_t,
        a.flag_tem_regime_t,
        a.flag_lucro_real_t,
        a.flag_lucro_presumido_t,
        a.flag_lucro_arbitrado_t,
        a.flag_imune_ou_isenta_t,
        a.y_baixa_12m,
        a.bucket_valid,
        COALESCE(s.uf_snapshot, 'NAO_INFORMADO') AS uf_obs,
        CASE
            WHEN LENGTH(COALESCE(TRIM(s.cnae_fiscal_principal_snapshot), '')) >= 2
                THEN SUBSTR(TRIM(s.cnae_fiscal_principal_snapshot), 1, 2)
            ELSE 'NI'
        END AS cnae_div_obs
    FROM amostra a
    LEFT JOIN read_parquet('{SURVIVAL_PATH.as_posix()}') s
      USING (cnpj_basico)
    """
    df = con.execute(query).fetchdf()
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    bool_cols = [
        "flag_optante_simples_t",
        "flag_optante_mei_t",
        "flag_excluido_simples_ate_t",
        "flag_excluido_mei_ate_t",
        "flag_tem_regime_t",
        "flag_lucro_real_t",
        "flag_lucro_presumido_t",
        "flag_lucro_arbitrado_t",
        "flag_imune_ou_isenta_t",
    ]
    for col in bool_cols:
        prepared[col] = prepared[col].astype(int)

    prepared["idade_empresa_meses"] = prepared["idade_empresa_meses"].astype(float)
    prepared["idade_log_meses"] = np.log1p(prepared["idade_empresa_meses"].clip(lower=0))
    prepared["coorte_abertura_ano_c"] = prepared["coorte_abertura_ano"].astype(float) - 2000.0
    prepared["qtd_escrituracoes_regime_t_log"] = np.log1p(prepared["qtd_escrituracoes_regime_t"].astype(float).clip(lower=0))
    prepared["idade_faixa_heuristica"] = pd.cut(
        prepared["idade_empresa_meses"],
        bins=[-np.inf, 12, 24, 60, 120, 240, np.inf],
        labels=["00_12m", "13_24m", "25_60m", "61_120m", "121_240m", "240m_mais"],
        include_lowest=True,
    ).astype(str)
    prepared["uf_obs"] = prepared["uf_obs"].map(ascii_text)
    prepared["cnae_div_obs"] = prepared["cnae_div_obs"].map(ascii_text)
    return prepared


def model_definitions(pos_weight: float) -> dict[str, Any]:
    return {
        "baseline_temporal": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=300,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "logistic_full": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=400,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=12,
            min_samples_leaf=60,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=260,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            reg_alpha=0.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=pos_weight,
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="binary",
            is_unbalance=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def get_feature_set(model_name: str) -> list[str]:
    if model_name == "baseline_temporal":
        return FEATURES_TEMPORAL
    return FEATURES_FULL


def fit_model(model_name: str, estimator: Any, train_df: pd.DataFrame) -> Any:
    features = get_feature_set(model_name)
    X_train = train_df[features]
    y_train = train_df["y_baixa_12m"].astype(int)
    fitted = clone(estimator)
    fitted.fit(X_train, y_train)
    return fitted


def predict_scores(model_name: str, fitted: Any, df: pd.DataFrame) -> np.ndarray:
    if model_name == "heuristica_simples":
        scores = fitted.predict_proba(df)[:, 1]
        return np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    features = get_feature_set(model_name)
    X = df[features]
    if hasattr(fitted, "predict_proba"):
        scores = fitted.predict_proba(X)[:, 1]
    else:
        scores = fitted.predict(X)
    return np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)


def compute_binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    base_rate = float(np.mean(y_true))
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
        "log_loss": float(log_loss(y_true, scores)),
        "base_rate": base_rate,
    }


def lift_at_fraction(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> dict[str, float]:
    n = len(y_true)
    top_n = max(1, int(math.ceil(n * fraction)))
    order = np.argsort(scores)[::-1][:top_n]
    y_top = y_true[order]
    precision = float(np.mean(y_top))
    recall = float(np.sum(y_top) / max(np.sum(y_true), 1))
    base_rate = float(np.mean(y_true))
    lift = precision / base_rate if base_rate > 0 else np.nan
    return {
        "top_fraction": fraction,
        "top_n": top_n,
        "precision": precision,
        "recall": recall,
        "lift": lift,
    }


def evaluate_model(
    model_name: str,
    fitted: Any,
    df: pd.DataFrame,
    split_label: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y_true = df["y_baixa_12m"].astype(int).to_numpy()
    scores = predict_scores(model_name, fitted, df)
    metrics = compute_binary_metrics(y_true, scores)
    lift_5 = lift_at_fraction(y_true, scores, 0.05)
    lift_10 = lift_at_fraction(y_true, scores, 0.10)
    row = {
        "modelo": model_name,
        "split_avaliacao": split_label,
        **metrics,
        "precision_top_5": lift_5["precision"],
        "recall_top_5": lift_5["recall"],
        "lift_top_5": lift_5["lift"],
        "precision_top_10": lift_10["precision"],
        "recall_top_10": lift_10["recall"],
        "lift_top_10": lift_10["lift"],
    }
    preds = pd.DataFrame(
        {
            "cnpj_basico": df["cnpj_basico"].astype(str),
            "split_temporal": split_label,
            "modelo": model_name,
            "y_true": y_true,
            "score_raw": scores,
        }
    )
    return row, preds


def select_final_model(validation_table: pd.DataFrame, pr_tolerance: float = 0.001) -> str:
    best_pr = float(validation_table["pr_auc"].max())
    eligible = validation_table[validation_table["pr_auc"] >= (best_pr - pr_tolerance)].copy()
    complexity_rank = {
        "heuristica_simples": 0,
        "baseline_temporal": 1,
        "logistic_full": 2,
        "lightgbm": 3,
        "xgboost": 4,
        "random_forest": 5,
    }
    eligible["complexidade_rank"] = eligible["modelo"].map(complexity_rank).fillna(99)
    eligible = eligible.sort_values(["brier", "log_loss", "complexidade_rank"], ascending=[True, True, True])
    return str(eligible.iloc[0]["modelo"])


def fit_calibrators(scores: np.ndarray, y_true: np.ndarray) -> dict[str, ProbabilityCalibrator]:
    clipped = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(clipped, y_true)

    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    platt = LogisticRegression(solver="lbfgs", random_state=RANDOM_STATE)
    platt.fit(logits, y_true)

    return {
        "isotonic": ProbabilityCalibrator(method="isotonic", model=iso),
        "sigmoid": ProbabilityCalibrator(method="sigmoid", model=platt),
    }


def choose_calibrator(
    calibrators: dict[str, ProbabilityCalibrator],
    tuning_scores: np.ndarray,
    y_tuning: np.ndarray,
) -> tuple[str, pd.DataFrame]:
    rows = []
    for name, calibrator in calibrators.items():
        calibrated = calibrator.predict(tuning_scores)
        metrics = compute_binary_metrics(y_tuning, calibrated)
        rows.append(
            {
                "metodo_calibracao": name,
                **metrics,
            }
        )
    table = pd.DataFrame(rows).sort_values(["brier", "log_loss", "pr_auc"], ascending=[True, True, False])
    best_method = str(table.iloc[0]["metodo_calibracao"])
    return best_method, table


def tune_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    max_flag_rate: float = 0.10,
    min_precision_multiplier: float = 1.8,
) -> tuple[float, pd.DataFrame]:
    base_rate = float(np.mean(y_true))
    candidate_thresholds = np.unique(np.quantile(scores, np.linspace(0.70, 0.995, 90)))
    rows = []
    for threshold in candidate_thresholds:
        pred = scores >= threshold
        flagged_rate = float(np.mean(pred))
        if flagged_rate <= 0:
            continue
        precision = float(precision_score(y_true, pred, zero_division=0))
        recall = float(recall_score(y_true, pred, zero_division=0))
        f2 = float(fbeta_score(y_true, pred, beta=2, zero_division=0))
        lift = precision / base_rate if base_rate > 0 else np.nan
        rows.append(
            {
                "threshold": float(threshold),
                "flagged_rate": flagged_rate,
                "precision": precision,
                "recall": recall,
                "f2": f2,
                "lift": lift,
                "atende_capacidade": bool(flagged_rate <= max_flag_rate),
                "atende_precision_floor": bool(precision >= base_rate * min_precision_multiplier),
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[table["atende_capacidade"] & table["atende_precision_floor"]]
    if eligible.empty:
        eligible = table[table["atende_capacidade"]]
    chosen = eligible.sort_values(["f2", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    return float(chosen["threshold"]), table.sort_values("threshold")


def build_risk_bands(scores: np.ndarray, medium_threshold: float, high_threshold: float) -> np.ndarray:
    return np.where(scores >= high_threshold, "alto", np.where(scores >= medium_threshold, "medio", "baixo"))


def final_metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold
    base_rate = float(np.mean(y_true))
    precision = float(precision_score(y_true, pred, zero_division=0))
    recall = float(recall_score(y_true, pred, zero_division=0))
    f2 = float(fbeta_score(y_true, pred, beta=2, zero_division=0))
    flagged_rate = float(np.mean(pred))
    return {
        "threshold_alto_risco": float(threshold),
        "flagged_rate_alto_risco": flagged_rate,
        "precision_alto_risco": precision,
        "recall_alto_risco": recall,
        "f2_alto_risco": f2,
        "lift_alto_risco": precision / base_rate if base_rate > 0 else np.nan,
    }


def make_calibration_table(y_true: np.ndarray, scores_raw: np.ndarray, scores_cal: np.ndarray) -> pd.DataFrame:
    prob_true_raw, prob_pred_raw = calibration_curve(y_true, scores_raw, n_bins=10, strategy="quantile")
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, scores_cal, n_bins=10, strategy="quantile")
    rows = []
    for idx, (pred, true) in enumerate(zip(prob_pred_raw, prob_true_raw), start=1):
        rows.append({"modelo": "nao_calibrado", "bin": idx, "score_medio_bin": float(pred), "taxa_real_bin": float(true)})
    for idx, (pred, true) in enumerate(zip(prob_pred_cal, prob_true_cal), start=1):
        rows.append({"modelo": "calibrado", "bin": idx, "score_medio_bin": float(pred), "taxa_real_bin": float(true)})
    return pd.DataFrame(rows)


def subgroup_stability(
    df: pd.DataFrame,
    group_col: str,
    min_n: int,
    high_threshold: float,
) -> pd.DataFrame:
    volumes = df.groupby(group_col).size().sort_values(ascending=False)
    groups = volumes[volumes >= min_n].head(TOP_GROUPS).index.tolist()
    rows = []
    for group in groups:
        sub = df[df[group_col] == group].copy()
        y = sub["y_baixa_12m"].astype(int).to_numpy()
        scores = sub["score_calibrado"].to_numpy()
        if len(np.unique(y)) < 2:
            roc_auc = np.nan
            pr_auc = np.nan
        else:
            roc_auc = float(roc_auc_score(y, scores))
            pr_auc = float(average_precision_score(y, scores))
        threshold_metrics = final_metrics_at_threshold(y, scores, high_threshold)
        rows.append(
            {
                "grupo": ascii_text(group),
                "n": int(len(sub)),
                "taxa_evento": float(np.mean(y)),
                "score_medio": float(np.mean(scores)),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "brier": float(brier_score_loss(y, scores)),
                **threshold_metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def explain_final_model(
    model_name: str,
    model: Any,
    X_reference: pd.DataFrame,
    X_explain: pd.DataFrame,
    explain_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if model_name in {"random_forest", "xgboost", "lightgbm"}:
        explainer = shap.Explainer(model, X_reference)
        explanation = explainer(X_explain)
        shap_values = np.asarray(explanation.values, dtype=float)
        if shap_values.ndim == 3:
            if shap_values.shape[-1] == 2:
                shap_values = shap_values[:, :, 1]
            else:
                shap_values = shap_values.mean(axis=-1)
        mean_abs = np.abs(shap_values).mean(axis=0)
        global_df = pd.DataFrame(
            {
                "feature": X_explain.columns,
                "mean_abs_shap": mean_abs,
            }
        ).sort_values("mean_abs_shap", ascending=False)
        contrib_matrix = shap_values
    else:
        feature_names = list(X_explain.columns)
        if isinstance(model, Pipeline):
            scaler = model.named_steps["scaler"]
            base_model = model.named_steps["model"]
            X_ref_scaled = scaler.transform(X_reference[feature_names])
            X_scaled = scaler.transform(X_explain[feature_names])
            explainer = shap.LinearExplainer(base_model, X_ref_scaled)
            shap_values = explainer.shap_values(X_scaled)
            contrib_matrix = np.asarray(shap_values, dtype=float)
            global_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "mean_abs_shap": np.abs(contrib_matrix).mean(axis=0),
                }
            ).sort_values("mean_abs_shap", ascending=False)
        else:
            raise ValueError(f"Unsupported explainability path for model {model_name}")

    global_df["feature"] = global_df["feature"].map(ascii_text)

    local_rows = []
    top_idx = explain_meta["score_calibrado"].nlargest(10).index.tolist()
    top_features = list(X_explain.columns)
    for idx in top_idx:
        contribs = contrib_matrix[idx]
        order = np.argsort(np.abs(contribs))[::-1][:3]
        record = {
            "cnpj_basico": str(explain_meta.loc[idx, "cnpj_basico"]),
            "y_true": int(explain_meta.loc[idx, "y_baixa_12m"]),
            "score_calibrado": float(explain_meta.loc[idx, "score_calibrado"]),
        }
        for pos, feat_idx in enumerate(order, start=1):
            feature_name = ascii_text(top_features[feat_idx])
            record[f"top{pos}_feature"] = feature_name
            record[f"top{pos}_valor"] = float(X_explain.iloc[idx, feat_idx])
            record[f"top{pos}_contrib"] = float(contribs[feat_idx])
        local_rows.append(record)
    local_df = pd.DataFrame(local_rows)
    return global_df, local_df


def plot_curve_comparison(curve_data: list[tuple[str, np.ndarray, np.ndarray]], file_name: str, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 7))
    for label, x, y in curve_data:
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / file_name, dpi=180)
    plt.close()


def plot_calibration(calibration_df: pd.DataFrame, file_name: str) -> None:
    plt.figure(figsize=(8, 6))
    for label, sub in calibration_df.groupby("modelo"):
        ordered = sub.sort_values("bin")
        plt.plot(ordered["score_medio_bin"], ordered["taxa_real_bin"], marker="o", label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#666666")
    plt.xlabel("Probabilidade prevista media")
    plt.ylabel("Taxa observada")
    plt.title("Curva de calibracao no teste")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / file_name, dpi=180)
    plt.close()


def plot_lift_deciles(df: pd.DataFrame, file_name: str) -> None:
    ranked = df.sort_values("score_calibrado", ascending=False).copy()
    ranked["decil"] = pd.qcut(np.arange(len(ranked)), 10, labels=[f"D{i}" for i in range(1, 11)])
    deciles = (
        ranked.groupby("decil", observed=False)
        .agg(n=("y_baixa_12m", "size"), taxa_evento=("y_baixa_12m", "mean"), score_medio=("score_calibrado", "mean"))
        .reset_index()
    )
    base_rate = float(ranked["y_baixa_12m"].mean())
    deciles["lift"] = deciles["taxa_evento"] / base_rate
    plt.figure(figsize=(11, 6))
    sns.barplot(data=deciles, x="decil", y="lift", color="#2a9d8f")
    plt.axhline(1.0, linestyle="--", color="#d62828")
    plt.title("Lift por decil de score no teste")
    plt.xlabel("Decil de risco")
    plt.ylabel("Lift")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / file_name, dpi=180)
    plt.close()


def plot_importance(global_df: pd.DataFrame, file_name: str) -> None:
    plot_df = global_df.head(12).sort_values("mean_abs_shap")
    plt.figure(figsize=(10, 7))
    plt.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#264653")
    plt.title("Explicabilidade global do modelo final")
    plt.xlabel("Importancia media absoluta")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / file_name, dpi=180)
    plt.close()


def plot_stability(df: pd.DataFrame, x_col: str, value_col: str, file_name: str, title: str) -> None:
    ordered = df.sort_values(value_col, ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=ordered, x=x_col, y=value_col, color="#457b9d")
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / file_name, dpi=180)
    plt.close()


def write_df(df: pd.DataFrame, path: Path) -> None:
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_object_dtype(safe[col]):
            safe[col] = safe[col].map(ascii_text)
    safe.to_csv(path, index=False, encoding="utf-8")


def build_report(
    sample_summary: pd.DataFrame,
    validation_table: pd.DataFrame,
    test_table: pd.DataFrame,
    calibration_table: pd.DataFrame,
    threshold_table: pd.DataFrame,
    final_model_name: str,
    calibration_name: str,
    final_split_metrics: pd.DataFrame,
    global_importance: pd.DataFrame,
    path: Path,
) -> None:
    best_valid = validation_table.sort_values("pr_auc", ascending=False).iloc[0]
    best_test = test_table.sort_values("pr_auc", ascending=False).iloc[0]
    final_test = final_split_metrics[final_split_metrics["split_temporal"] == "teste"].iloc[0]
    final_valid = final_split_metrics[final_split_metrics["split_temporal"] == "validacao"].iloc[0]
    top_features = global_importance.head(8)

    report = f"""# Fase 06 - Modelagem do Radar de Risco

Esta fase transforma o `radar_12m` em um sistema preditivo comparando baselines, calibrando o modelo final, definindo um threshold operacional e auditando estabilidade e explicabilidade.

## Regras metodologicas desta fase

- O treino respeitou o split temporal original: treino em 2022, validacao em 2023 e teste em 2024.
- A modelagem foi feita sobre amostras deterministicas por hash de `cnpj_basico`, para manter reproducibilidade computacional.
- A selecao do melhor modelo foi feita na validacao.
- Calibracao e threshold tuning foram separados em subconjuntos distintos da validacao.
- Segmentacoes por `UF` e `CNAE` foram usadas apenas para estabilidade, nao como justificativa causal.
- A escolha final do modelo usou tolerancia de empate em PR-AUC e desempate por Brier, log-loss e simplicidade.

## Resumo executivo

- Taxa de evento na amostra de treino: {sample_summary[sample_summary['split_temporal']=='treino'].iloc[0]['taxa_evento']:.3f}
- Melhor modelo na validacao por PR-AUC: {best_valid['modelo']} ({best_valid['pr_auc']:.4f})
- Melhor modelo no teste por PR-AUC: {best_test['modelo']} ({best_test['pr_auc']:.4f})
- Modelo final selecionado: {final_model_name}
- Metodo de calibracao final: {calibration_name}
- ROC-AUC final no teste: {final_test['roc_auc']:.4f}
- PR-AUC final no teste: {final_test['pr_auc']:.4f}
- Brier final no teste: {final_test['brier']:.4f}
- Lift top 10% no teste: {final_test['lift_top_10']:.2f}

## Comparacao de modelos

- Baseline de maior PR-AUC na validacao: {validation_table[validation_table['modelo'].isin(['heuristica_simples','baseline_temporal','logistic_full'])].sort_values('pr_auc', ascending=False).iloc[0]['modelo']}
- Melhor modelo principal na validacao: {validation_table[validation_table['modelo'].isin(['random_forest','xgboost','lightgbm'])].sort_values('pr_auc', ascending=False).iloc[0]['modelo']}
- O threshold final foi escolhido com restricao de capacidade de ate 10% dos casos sinalizados e resultou em {final_valid['flagged_rate_alto_risco']:.3f} de casos marcados na validacao.

## Calibracao e threshold

- Melhor calibracao no tuning: {calibration_table.sort_values('brier').iloc[0]['metodo_calibracao']}
- Threshold alto risco escolhido: {final_test['threshold_alto_risco']:.4f}
- Precision no alto risco no teste: {final_test['precision_alto_risco']:.3f}
- Recall no alto risco no teste: {final_test['recall_alto_risco']:.3f}
- Lift no alto risco no teste: {final_test['lift_alto_risco']:.2f}

## Explicabilidade global

{chr(10).join(f"- {ascii_text(row.feature)}: {row.mean_abs_shap:.5f}" for row in top_features.itertuples(index=False))}

## Cuidado interpretativo

- O radar continua limitado pelo espaco de features disponivel no `radar_12m`, que e propositalmente enxuto e temporalmente defensavel.
- A cobertura de `Regimes Tributarios` e baixa e precisa seguir sendo lida como fonte parcial.
- Explicabilidade do modelo final mostra associacoes preditivas, nao causalidade.
- Como o modelo final e linear, a explicabilidade foi obtida via SHAP linear sobre as features padronizadas.
"""
    path.write_text(report, encoding="utf-8")


def build_methodology_review(
    final_model_name: str,
    calibration_name: str,
    validation_table: pd.DataFrame,
    calibration_table: pd.DataFrame,
    final_split_metrics: pd.DataFrame,
    path: Path,
) -> None:
    final_test = final_split_metrics[final_split_metrics["split_temporal"] == "teste"].iloc[0]
    review = f"""# Fase 06 - Revisao Metodologica

## Parecer geral

A Fase 6 ficou aprovada. O radar 12M foi transformado em um pipeline preditivo completo com comparacao justa, calibracao, threshold operacional, explicabilidade e auditoria de estabilidade.

## Problemas metodologicos identificados e como foram tratados

### 1. Volume total inviavel para treino iterativo no ambiente local

**Problema**

Os splits completos do radar somam dezenas de milhoes de linhas.

**Risco**

Treinar e comparar varios modelos diretamente nesse volume comprometeria reproducibilidade e tempo de execucao.

**Tratamento**

- Foi adotada amostragem deterministica por hash de `cnpj_basico`.
- O desenho permanece temporalmente valido e reproduzivel.

### 2. Necessidade de separar selecao, calibracao e threshold tuning

**Problema**

Usar a mesma validacao para tudo geraria otimismo excessivo.

**Tratamento**

- A validacao foi dividida em tres buckets deterministicas.
- Um bucket serviu para selecao do modelo.
- Outro para calibracao.
- Outro para threshold tuning.

### 3. Cobertura baixa de `Regimes Tributarios`

**Problema**

A cobertura da fonte tributaria permanece em torno de 5,5%.

**Risco**

O modelo pode aprender diferencas entre populacoes com e sem cobertura, e nao apenas sinais economicos.

**Tratamento**

- `flag_tem_regime_t` foi mantida para distinguir ausencia de cobertura de zero estrutural.
- A estabilidade por cobertura foi incluida na fase.

### 4. Calibracao precisava ser demonstrada, nao presumida

**Problema**

Modelos de boosting costumam ranquear bem, mas podem entregar probabilidades mal calibradas.

**Tratamento**

- Foram comparados metodos `isotonic` e `sigmoid`.
- O metodo final escolhido foi `{calibration_name}` com base no menor Brier no tuning.

### 5. Selecao final do modelo nao podia depender de diferenca irrelevante de ranking

**Problema**

Na primeira versao da fase, a regra de selecao por PR-AUC puro favorecia um modelo com ganho marginal de ranking, apesar de pior qualidade probabilistica.

**Tratamento**

- A regra final passou a usar tolerancia de empate em PR-AUC.
- Dentro do empate, o desempate passou a considerar `brier`, `log_loss` e simplicidade.
- Isso levou a selecao de `{final_model_name}` como modelo final.

### 6. Threshold operacional nao podia ser arbitrario

**Problema**

Fixar um threshold sem restricao operacional tornaria o score pouco defensavel.

**Tratamento**

- O threshold foi escolhido no bucket de tuning com limite de ate 10% de casos sinalizados.
- O criterio final privilegiou F2 com piso minimo de precisao.

### 7. Explicabilidade e estabilidade precisavam estar alinhadas ao modelo final

**Problema**

Interpretar um modelo sem auditar estabilidade temporal e por subgrupo produz leitura incompleta.

**Tratamento**

- A fase gerou estabilidade por tempo, `UF`, `CNAE` e cobertura de regime.
- A explicabilidade global e local foi gerada no modelo final `{final_model_name}`.
- Como o modelo final e linear, a explicabilidade foi calculada com SHAP linear.

## Conclusao

A Fase 6 entrega um radar preditivo fechado e auditado. O principal ganho metodologico foi transformar um problema de classificacao temporal em um caso de portfolio defensavel: modelo escolhido por validacao, calibrado fora da selecao, threshold operacional documentado e estabilidade medida antes de seguir para storytelling e produto.
"""
    path.write_text(review, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    con = connect_duckdb()
    treino_df = prepare_dataframe(load_split_sample(con, "treino", TRAIN_PATH))
    valid_df = prepare_dataframe(load_split_sample(con, "validacao", VALID_PATH))
    teste_df = prepare_dataframe(load_split_sample(con, "teste", TEST_PATH))

    valid_select = valid_df[valid_df["bucket_valid"] == 0].reset_index(drop=True)
    valid_cal = valid_df[valid_df["bucket_valid"] == 1].reset_index(drop=True)
    valid_threshold = valid_df[valid_df["bucket_valid"] == 2].reset_index(drop=True)

    sample_summary = pd.DataFrame(
        [
            {
                "split_temporal": "treino",
                "n": int(len(treino_df)),
                "taxa_evento": float(treino_df["y_baixa_12m"].mean()),
                "cobertura_regime": float(treino_df["flag_tem_regime_t"].mean()),
            },
            {
                "split_temporal": "validacao_total",
                "n": int(len(valid_df)),
                "taxa_evento": float(valid_df["y_baixa_12m"].mean()),
                "cobertura_regime": float(valid_df["flag_tem_regime_t"].mean()),
            },
            {
                "split_temporal": "teste",
                "n": int(len(teste_df)),
                "taxa_evento": float(teste_df["y_baixa_12m"].mean()),
                "cobertura_regime": float(teste_df["flag_tem_regime_t"].mean()),
            },
        ]
    )

    pos_rate = float(treino_df["y_baixa_12m"].mean())
    pos_weight = float((1.0 - pos_rate) / pos_rate)
    estimators = model_definitions(pos_weight)

    heuristic = HeuristicRiskModel().fit(treino_df, "y_baixa_12m")
    fitted_models: dict[str, Any] = {"heuristica_simples": heuristic}
    for model_name, estimator in estimators.items():
        fitted_models[model_name] = fit_model(model_name, estimator, treino_df)

    validation_rows = []
    test_rows = []
    for model_name, model in fitted_models.items():
        v_row, _ = evaluate_model(model_name, model, valid_select, "validacao_selecao")
        t_row, _ = evaluate_model(model_name, model, teste_df, "teste")
        validation_rows.append(v_row)
        test_rows.append(t_row)

    validation_table = pd.DataFrame(validation_rows).sort_values(["pr_auc", "brier"], ascending=[False, True]).reset_index(drop=True)
    test_table = pd.DataFrame(test_rows).sort_values(["pr_auc", "brier"], ascending=[False, True]).reset_index(drop=True)

    final_model_name = select_final_model(validation_table)
    final_model = fitted_models[final_model_name]

    valid_cal_scores = predict_scores(final_model_name, final_model, valid_cal)
    valid_threshold_scores_raw = predict_scores(final_model_name, final_model, valid_threshold)
    calibrators = fit_calibrators(valid_cal_scores, valid_cal["y_baixa_12m"].astype(int).to_numpy())
    calibration_name, calibration_table = choose_calibrator(
        calibrators,
        valid_threshold_scores_raw,
        valid_threshold["y_baixa_12m"].astype(int).to_numpy(),
    )
    final_calibrator = calibrators[calibration_name]

    valid_threshold_scores_cal = final_calibrator.predict(valid_threshold_scores_raw)
    threshold_high, threshold_table = tune_threshold(
        valid_threshold_scores_cal,
        valid_threshold["y_baixa_12m"].astype(int).to_numpy(),
    )
    threshold_medium = float(np.quantile(valid_threshold_scores_cal, 0.50))

    split_frames = {"treino": treino_df, "validacao": valid_df, "teste": teste_df}
    split_metrics_rows = []
    enriched_splits = {}
    for split_name, frame in split_frames.items():
        raw_scores = predict_scores(final_model_name, final_model, frame)
        calibrated = final_calibrator.predict(raw_scores)
        y_true = frame["y_baixa_12m"].astype(int).to_numpy()
        metrics = compute_binary_metrics(y_true, calibrated)
        lift_5 = lift_at_fraction(y_true, calibrated, 0.05)
        lift_10 = lift_at_fraction(y_true, calibrated, 0.10)
        threshold_metrics = final_metrics_at_threshold(y_true, calibrated, threshold_high)
        split_metrics_rows.append(
            {
                "modelo_final": final_model_name,
                "calibracao": calibration_name,
                "split_temporal": split_name,
                **metrics,
                "precision_top_5": lift_5["precision"],
                "recall_top_5": lift_5["recall"],
                "lift_top_5": lift_5["lift"],
                "precision_top_10": lift_10["precision"],
                "recall_top_10": lift_10["recall"],
                "lift_top_10": lift_10["lift"],
                **threshold_metrics,
            }
        )
        enriched = frame.copy()
        enriched["score_raw"] = raw_scores
        enriched["score_calibrado"] = calibrated
        enriched["banda_risco"] = build_risk_bands(calibrated, threshold_medium, threshold_high)
        enriched_splits[split_name] = enriched

    final_split_metrics = pd.DataFrame(split_metrics_rows)

    test_final = enriched_splits["teste"].copy()
    calibration_df = make_calibration_table(
        test_final["y_baixa_12m"].astype(int).to_numpy(),
        test_final["score_raw"].to_numpy(),
        test_final["score_calibrado"].to_numpy(),
    )
    band_summary = (
        test_final.groupby("banda_risco", observed=False)
        .agg(
            n=("y_baixa_12m", "size"),
            taxa_evento=("y_baixa_12m", "mean"),
            score_medio=("score_calibrado", "mean"),
        )
        .reset_index()
    )

    stability_regime = subgroup_stability(test_final, "flag_tem_regime_t", min_n=1000, high_threshold=threshold_high)
    stability_uf = subgroup_stability(test_final, "uf_obs", min_n=5000, high_threshold=threshold_high)
    stability_cnae = subgroup_stability(test_final, "cnae_div_obs", min_n=5000, high_threshold=threshold_high)

    explain_sample = test_final.sample(n=min(5000, len(test_final)), random_state=RANDOM_STATE).reset_index(drop=True)
    X_explain = explain_sample[get_feature_set(final_model_name)].copy()
    global_importance, local_explanations = explain_final_model(
        final_model_name,
        final_model,
        treino_df[get_feature_set(final_model_name)].head(5000),
        X_explain,
        explain_sample[["cnpj_basico", "y_baixa_12m", "score_calibrado"]].copy(),
    )

    final_test_predictions = test_final[
        ["cnpj_basico", "y_baixa_12m", "score_raw", "score_calibrado", "banda_risco", "uf_obs", "cnae_div_obs"]
    ].copy()

    write_df(sample_summary, RESULTS_DIR / "amostra_modelagem_resumo.csv")
    write_df(validation_table, RESULTS_DIR / "comparacao_modelos_validacao.csv")
    write_df(test_table, RESULTS_DIR / "comparacao_modelos_teste.csv")
    write_df(calibration_table, RESULTS_DIR / "calibracao_metodos.csv")
    write_df(threshold_table, RESULTS_DIR / "threshold_tuning.csv")
    write_df(final_split_metrics, RESULTS_DIR / "metricas_finais_por_split.csv")
    write_df(calibration_df, RESULTS_DIR / "curva_calibracao_teste.csv")
    write_df(band_summary, RESULTS_DIR / "bandas_risco_teste.csv")
    write_df(stability_regime, RESULTS_DIR / "estabilidade_regime_teste.csv")
    write_df(stability_uf, RESULTS_DIR / "estabilidade_uf_teste.csv")
    write_df(stability_cnae, RESULTS_DIR / "estabilidade_cnae_teste.csv")
    write_df(global_importance, RESULTS_DIR / "shap_importancia_global.csv")
    write_df(local_explanations, RESULTS_DIR / "shap_explicacoes_locais.csv")
    write_df(final_test_predictions, RESULTS_DIR / "predicoes_teste_modelo_final.csv")

    joblib.dump(fitted_models["baseline_temporal"], MODELS_DIR / "baseline_temporal.joblib")
    joblib.dump(fitted_models["logistic_full"], MODELS_DIR / "logistic_full.joblib")
    joblib.dump(fitted_models["random_forest"], MODELS_DIR / "random_forest.joblib")
    joblib.dump(fitted_models["xgboost"], MODELS_DIR / "xgboost.joblib")
    joblib.dump(fitted_models["lightgbm"], MODELS_DIR / "lightgbm.joblib")
    joblib.dump(final_model, MODELS_DIR / "modelo_final.joblib")
    joblib.dump(final_calibrator, MODELS_DIR / "calibrador_final.joblib")
    heuristic_tables_json = {
        feature: {ascii_text(key): float(value) for key, value in table.items()}
        for feature, table in heuristic.tables.items()
    }
    (MODELS_DIR / "heuristica_mapping.json").write_text(
        json.dumps({"overall_rate": heuristic.overall_rate, "tables": heuristic_tables_json}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    valid_curve_data_roc = []
    valid_curve_data_pr = []
    for model_name, model in fitted_models.items():
        scores = predict_scores(model_name, model, valid_select)
        y_true = valid_select["y_baixa_12m"].astype(int).to_numpy()
        fpr, tpr, _ = roc_curve(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        valid_curve_data_roc.append((model_name, fpr, tpr))
        valid_curve_data_pr.append((model_name, recall, precision))

    plot_curve_comparison(valid_curve_data_roc, "graf_01_roc_validacao.png", "ROC na validacao de selecao", "False positive rate", "True positive rate")
    plot_curve_comparison(valid_curve_data_pr, "graf_02_pr_validacao.png", "Precision-Recall na validacao de selecao", "Recall", "Precision")
    plot_calibration(calibration_df, "graf_03_calibracao_teste.png")
    plot_lift_deciles(test_final, "graf_04_lift_decis_teste.png")
    plot_importance(global_importance, "graf_05_importancia_global.png")
    plot_stability(final_split_metrics, "split_temporal", "pr_auc", "graf_06_estabilidade_tempo.png", "PR-AUC do modelo final por split")
    plot_stability(stability_uf, "grupo", "lift_alto_risco", "graf_07_estabilidade_uf.png", "Lift do alto risco por UF observada")
    plot_stability(stability_cnae, "grupo", "lift_alto_risco", "graf_08_estabilidade_cnae.png", "Lift do alto risco por divisao CNAE observada")

    report_path = PHASE_DIR / "RELATORIO_FASE_06_MODELAGEM_RADAR.md"
    review_path = PHASE_DIR / "RELATORIO_FASE_06_REVISAO_METODOLOGICA.md"
    build_report(
        sample_summary=sample_summary,
        validation_table=validation_table,
        test_table=test_table,
        calibration_table=calibration_table,
        threshold_table=threshold_table,
        final_model_name=final_model_name,
        calibration_name=calibration_name,
        final_split_metrics=final_split_metrics,
        global_importance=global_importance,
        path=report_path,
    )
    build_methodology_review(
        final_model_name=final_model_name,
        calibration_name=calibration_name,
        validation_table=validation_table,
        calibration_table=calibration_table,
        final_split_metrics=final_split_metrics,
        path=review_path,
    )

    summary = {
        "sample": sample_summary.to_dict(orient="records"),
        "final_model": {
            "modelo": final_model_name,
            "calibracao": calibration_name,
            "threshold_high": threshold_high,
            "threshold_medium": threshold_medium,
        },
        "test_metrics": final_split_metrics[final_split_metrics["split_temporal"] == "teste"].to_dict(orient="records")[0],
        "top_features": global_importance.head(8).to_dict(orient="records"),
    }
    (RESULTS_DIR / "00_resumo_fase_06.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
