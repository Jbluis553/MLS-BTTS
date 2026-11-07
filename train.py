"""Entrena y evalua modelos para BTTS-Yes con split temporal (sin shuffle)."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def load_params() -> dict:
    with open(ROOT / "config" / "params.yaml") as f:
        return yaml.safe_load(f)


def main() -> None:
    p = load_params()
    df = pd.read_csv(ROOT / "data" / "processed" / "features.csv",
                     parse_dates=["kickoff"])
    df = df.sort_values("kickoff").reset_index(drop=True)

    if p.get("use_odds_feature", True):
        cov = df["btts_yes_odd"].notna().mean()
        if cov < p.get("min_odds_coverage", 0.5):
            df = df.drop(columns=["btts_yes_odd"])
            print(f"Cuota BTTS con cobertura {cov:.0%} < minima: se omite como feature")

    drop = {"btts_yes", "kickoff", "match_id", "season_id"}
    feature_cols = [c for c in df.columns if c not in drop]
    X, y = df[feature_cols], df["btts_yes"]

    cut = int(len(df) * (1 - p["test_fraction"]))
    X_tr, X_te = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_te = y.iloc[:cut], y.iloc[cut:]
    print(f"Train: {len(X_tr)} | Test: {len(X_te)} | Features: {feature_cols}")

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=p["random_state"]),
    }

    results, best_name, best_ll = {}, None, np.inf
    for name, model in candidates.items():
        cal = CalibratedClassifierCV(model, method="isotonic", cv=3)
        cal.fit(X_tr, y_tr)
        proba = cal.predict_proba(X_te)[:, 1]
        ll = log_loss(y_te, proba)
        results[name] = {
            "log_loss": round(ll, 4),
            "brier": round(brier_score_loss(y_te, proba), 4),
            "roc_auc": round(roc_auc_score(y_te, proba), 4),
        }
        print(f"{name}: {results[name]}")
        if ll < best_ll:
            best_ll, best_name = ll, name
            bundle = {"model": cal, "features": feature_cols,
                      "trained_until": str(df['kickoff'].iloc[cut - 1]),
                      "params": p}
            joblib.dump(bundle, MODELS / "btts_yes_model.joblib")

    prob_true, prob_pred = calibration_curve(y_te,
        bundle["model"].predict_proba(X_te)[:, 1], n_bins=10)
    metrics = {
        "best_model": best_name,
        "results": results,
        "test_period": {"from": str(df['kickoff'].iloc[cut]),
                        "to": str(df['kickoff'].iloc[-1])},
        "test_size": len(X_te),
        "base_rate_btts_yes": round(float(y_te.mean()), 4),
        "calibration": {"prob_true": prob_true.tolist(),
                        "prob_pred": prob_pred.tolist()},
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    lines = ["# Reporte de entrenamiento", "",
             f"- Mejor modelo: **{best_name}**",
             f"- Periodo de test: {metrics['test_period']['from'][:10]} a {metrics['test_period']['to'][:10]} ({len(X_te)} partidos)",
             f"- Tasa base BTTS-Yes en test: {metrics['base_rate_btts_yes']:.1%}", "",
             "| Modelo | Log loss | Brier | ROC-AUC |", "|---|---|---|---|"]
    for n, r in results.items():
        lines.append(f"| {n} | {r['log_loss']} | {r['brier']} | {r['roc_auc']} |")
    (REPORTS / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Guardado: models/btts_yes_model.joblib, reports/metrics.json, reports/report.md")


if __name__ == "__main__":
    main()
