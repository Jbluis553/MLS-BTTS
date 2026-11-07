"""Descarga incremental de datos reales de la MLS desde TheStatsAPI.

Guarda data/raw/mls_dataset.jsonl con una fila por partido finalizado:
match_id, season_id, kickoff, equipos, goles, cuota BTTS-Yes (promedio) y xG si hay.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.thestatsapi_client import (TheStatsAPIClient, extract_btts_yes_odd,
                                    extract_xg)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
DATASET = RAW / "mls_dataset.jsonl"


def load_existing() -> dict:
    """match_id -> fila, para no re-descargar lo ya cacheado."""
    rows = {}
    if DATASET.exists():
        for line in DATASET.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["match_id"]] = r
    return rows


def main() -> None:
    client = TheStatsAPIClient()
    mls = client.find_mls()
    if not mls:
        raise SystemExit("La API no devolvio informacion de la competicion MLS.")

    competition_id = mls["id"]
    print(f"MLS: {mls['name']} ({competition_id})")

    seasons = client.list_seasons(competition_id)
    if not seasons:
        raise SystemExit("La API no devolvio temporadas para la MLS")

    cache = load_existing()
    new_rows = 0

    for season in seasons:
        season_id = season["id"]
        matches = client.list_matches(competition_id, season_id, status="finished")
        finished = [m for m in matches if m.get("status") == "finished"]
        for m in finished:
            match_id = m.get("id")
            if not match_id or match_id in cache:
                continue
            score = (m.get("score") or {}).get("regulation") or {}
            if score.get("home") is None or score.get("away") is None:
                continue  # sin marcador utilizable

            btts_yes_odd = None
            home_xg = away_xg = None

            if m.get("odds_available"):
                btts_yes_odd = extract_btts_yes_odd(client.match_odds(match_id))
            if m.get("xg_available"):
                home_xg, away_xg = extract_xg(client.match_stats(match_id))

            home_team = m.get("home_team") or {}
            away_team = m.get("away_team") or {}
            if not home_team.get("id") or not away_team.get("id"):
                continue

            row = {
                "match_id": match_id,
                "season_id": season_id,
                "season_name": season.get("name"),
                "kickoff": m.get("utc_date"),
                "matchday": m.get("matchday"),
                "home_team_id": home_team["id"],
                "home_team": home_team.get("name"),
                "away_team_id": away_team["id"],
                "away_team": away_team.get("name"),
                "home_goals": int(score["home"]),
                "away_goals": int(score["away"]),
                "btts_yes_odd": btts_yes_odd,
                "home_xg": home_xg,
                "away_xg": away_xg,
            }
            cache[match_id] = row
            new_rows += 1

    with DATASET.open("w") as f:
        for r in sorted(cache.values(), key=lambda x: x["kickoff"] or ""):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Partidos en dataset: {len(cache)}  |  nuevos: {new_rows}")


if __name__ == "__main__":
    main()
