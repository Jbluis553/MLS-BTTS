"""Construye el dataset de entrenamiento con rolling windows SIN data leakage.

Toda feature de un partido usa UNICAMENTE partidos finalizados con fecha < kickoff.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def load_params() -> dict:
    with open(ROOT / "config" / "params.yaml") as f:
        return yaml.safe_load(f)


def _team_history(df: pd.DataFrame) -> dict:
    """team_id -> DataFrame cronologico de sus partidos (como local y visitante)."""
    hist = {}
    for tid in pd.concat([df["home_team_id"], df["away_team_id"]]).unique():
        home = df[df["home_team_id"] == tid][["kickoff", "home_goals", "away_goals"]].copy()
        home.columns = ["kickoff", "gf", "ga"]
        away = df[df["away_team_id"] == tid][["kickoff", "away_goals", "home_goals"]].copy()
        away.columns = ["kickoff", "gf", "ga"]
        h = pd.concat([home, away]).sort_values("kickoff").reset_index(drop=True)
        h["btts"] = ((h["gf"] > 0) & (h["ga"] > 0)).astype(int)
        hist[tid] = h
    return hist


def build_features(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.sort_values("kickoff").reset_index(drop=True)
    hist = _team_history(df)
    n_ha, n_any = p["lookback_home_away"], p["lookback_any"]

    def team_feats(tid, kickoff, side, n=n_ha):
        h = hist[tid]
        prev = h[h["kickoff"] < kickoff]
        # ultimos N del mismo lado
        if side == "home":
            same = df[(df["home_team_id"] == tid) & (df["kickoff"] < kickoff)]
            gf_s = same["home_goals"].tail(n).mean()
            ga_s = same["away_goals"].tail(n).mean()
        else:
            same = df[(df["away_team_id"] == tid) & (df["kickoff"] < kickoff)]
            gf_s = same["away_goals"].tail(n).mean()
            ga_s = same["home_goals"].tail(n).mean()
        last = prev.tail(n_any)
        rest = None
        if len(prev):
            rest = (pd.Timestamp(kickoff) - pd.Timestamp(prev["kickoff"].iloc[-1])).days
        return {
            f"{side}_gf_same": gf_s,
            f"{side}_ga_same": ga_s,
            f"{side}_btts_rate": last["btts"].mean(),
            f"{side}_gf_avg": prev["gf"].tail(n_any).mean(),
            f"{side}_ga_avg": prev["ga"].tail(n_any).mean(),
            f"{side}_rest_days": rest,
        }

    rows = []
    for _, m in df.iterrows():
        f = {}
        f.update(team_feats(m["home_team_id"], m["kickoff"], "home"))
        f.update(team_feats(m["away_team_id"], m["kickoff"], "away"))
        f["btts_yes"] = int(m["home_goals"] > 0 and m["away_goals"] > 0)
        f["kickoff"] = m["kickoff"]
        f["match_id"] = m["match_id"]
        f["season_id"] = m["season_id"]
        if p.get("use_odds_feature", True):
            f["btts_yes_odd"] = m["btts_yes_odd"]
        rows.append(f)

    out = pd.DataFrame(rows)

    # xG rolling (donde exista)
    if {"home_xg", "away_xg"}.issubset(df.columns) and df[["home_xg", "away_xg"]].notna().any().any():
        xg = p.get("xg_lookback", 5)
        home_xg_avg, away_xg_avg = [], []
        for _, m in df.iterrows():
            hprev = df[(df["home_team_id"] == m["home_team_id"]) & (df["kickoff"] < m["kickoff"])]
            aprev = df[(df["away_team_id"] == m["away_team_id"]) & (df["kickoff"] < m["kickoff"])]
            home_xg_avg.append(hprev["home_xg"].dropna().tail(xg).mean())
            away_xg_avg.append(aprev["away_xg"].dropna().tail(xg).mean())
        out["home_xg_avg"] = home_xg_avg
        out["away_xg_avg"] = away_xg_avg

    return out


def main() -> None:
    p = load_params()
    raw = ROOT / "data" / "raw" / "mls_dataset.jsonl"
    df = pd.read_json(raw, lines=True)
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)

    if p.get("train_seasons", 0):
        last_seasons = sorted(df["season_id"].unique())[-int(p["train_seasons"]):]
        df = df[df["season_id"].isin(last_seasons)]

    feats = build_features(df, p)
    before = len(feats)
    feats = feats.dropna(subset=[c for c in feats.columns if c not in ("btts_yes_odd", "home_xg_avg", "away_xg_avg")])
    print(f"Filas: {before} -> {len(feats)} tras dropna (sin historia previa)")
    feats.to_csv(PROCESSED / "features.csv", index=False)
    print(f"Tasa BTTS-Yes global: {feats['btts_yes'].mean():.3f}")


if __name__ == "__main__":
    main()
