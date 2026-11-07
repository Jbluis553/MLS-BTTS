"""Genera predicciones BTTS-Yes para los proximos partidos programados de la MLS."""
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.build_features import build_features, load_params  # reutiliza logica sin leakage
from src.thestatsapi_client import TheStatsAPIClient

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "predictions"
PRED.mkdir(exist_ok=True)


def main() -> None:
    p = load_params()
    bundle = joblib.load(ROOT / "models" / "btts_yes_model.joblib")
    client = TheStatsAPIClient()

    mls = client.find_mls()
    current = mls.get("current_season_id")
    if not current:
        seasons = client.list_seasons(mls["id"])
        current = next((s["id"] for s in seasons if s.get("is_current")), seasons[0]["id"])

    upcoming = client.list_matches(mls["id"], current, status="scheduled")
    if not upcoming:
        print("No hay partidos programados proximamente.")
        return

    # Historial necesario para las features: dataset cacheado + resultados recientes
    hist = pd.read_json(ROOT / "data" / "raw" / "mls_dataset.jsonl", lines=True)
    new_rows = []
    for m in upcoming:
        new_rows.append({
            "match_id": m["id"], "season_id": m["season_id"], "kickoff": m["utc_date"],
            "home_team_id": m["home_team"]["id"], "home_team": m["home_team"]["name"],
            "away_team_id": m["away_team"]["id"], "away_team": m["away_team"]["name"],
            "home_goals": 0, "away_goals": 0,  # placeholder; target no se usa
            "btts_yes_odd": None, "home_xg": None, "away_xg": None,
        })
    df = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True)
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)

    feats = build_features(df, p)
    feats = feats[feats["match_id"].isin([m["id"] for m in upcoming])]

    cols = bundle["features"]
    X = feats[cols]
    proba = bundle["model"].predict_proba(X)[:, 1]

    out = feats[["match_id", "kickoff"]].copy()
    meta = {m["id"]: (m["home_team"]["name"], m["away_team"]["name"]) for m in upcoming}
    out["home_team"] = out["match_id"].map(lambda i: meta[i][0])
    out["away_team"] = out["match_id"].map(lambda i: meta[i][1])
    out["p_btts_yes"] = proba.round(4)
    out = out.sort_values("kickoff")
    out.to_csv(PRED / "latest.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
