"""Cliente minimo para TheStatsAPI con manejo de rate limits (429 / Retry-After)."""
import os
import time

import requests

BASE_URL = "https://api.thestatsapi.com/api"


class TheStatsAPIError(RuntimeError):
    pass


class TheStatsAPIClient:
    """Cliente REST. Toda llamada lleva `Authorization: Bearer <key>`."""

    def __init__(self, api_key: str | None = None, max_retries: int = 6, timeout: int = 30):
        self.api_key = api_key or os.environ["THESTATSAPI_KEY"]
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _get(self, path: str, **params):
        params = {k: v for k, v in params.items() if v is not None}
        for attempt in range(self.max_retries):
            resp = self.session.get(f"{BASE_URL}{path}", params=params, timeout=self.timeout)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "60"))
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        raise TheStatsAPIError(f"Rate limit persistente tras {self.max_retries} reintentos: {path}")

    # ------------------------------------------------------------------ #
    def find_mls(self) -> dict:
        """Devuelve la competicion MLS (id tipo comp_...)."""
        js = self._get("/football/competitions", search="Major League Soccer", per_page=20)
        data = (js or {}).get("data", [])
        if not data:
            raise TheStatsAPIError("MLS no encontrada en /football/competitions")
        return data[0]

    def list_seasons(self, competition_id: str) -> list[dict]:
        js = self._get(f"/football/competitions/{competition_id}/seasons")
        return (js or {}).get("data", [])

    def list_matches(self, competition_id: str, season_id: str,
                     status: str | None = None) -> list[dict]:
        """Partidos de una temporada, paginando con per_page=100."""
        out, page = [], 1
        while True:
            js = self._get("/football/matches",
                           competition_id=competition_id,
                           season_id=season_id,
                           status=status,
                           per_page=100,
                           page=page)
            if not js:
                break
            out.extend(js.get("data", []))
            if page >= js.get("meta", {}).get("total_pages", 1):
                break
            page += 1
        return out

    def match_odds(self, match_id: str) -> dict | None:
        js = self._get(f"/football/matches/{match_id}/odds")
        return (js or {}).get("data")

    def match_stats(self, match_id: str) -> dict | None:
        js = self._get(f"/football/matches/{match_id}/stats")
        return (js or {}).get("data")


# --------------------------------------------------------------------- #
def extract_btts_yes_odd(odds_data: dict | None) -> float | None:
    """Promedio de cuotas 'yes' del mercado both_teams_to_score entre bookmakers.

    Estructura esperada: data.bookmakers[].markets[] con name 'both_teams_to_score'
    y entradas yes/no (campo 'odd'). Parseo defensivo por si cambian nombres.
    """
    if not odds_data:
        return None
    bookmakers = odds_data.get("bookmakers") or []
    yes_odds = []
    for bm in bookmakers:
        for market in bm.get("markets", []):
            name = str(market.get("name", "")).lower()
            if "both_teams_to_score" in name or name == "btts":
                entries = market.get("entries") or market.get("selections") or []
                for e in entries:
                    sel = str(e.get("selection", e.get("name", ""))).lower()
                    if sel == "yes":
                        odd = e.get("odd") or e.get("odds")
                        if odd:
                            yes_odds.append(float(odd))
    if not yes_odds:
        return None
    return round(sum(yes_odds) / len(yes_odds), 4)


def _find_xg(node, side: str) -> float | None:
    """Busca xG del equipo local/visitante dentro del JSON de stats (parseo defensivo)."""
    if not isinstance(node, dict):
        return None
    team = node.get(side) or node.get(f"{side}_team") or {}
    if isinstance(team, dict):
        for key in ("expected_goals", "xg", "xG"):
            if key in team and team[key] is not None:
                return float(team[key])
    for key in (f"{side}_xg", f"{side}_expected_goals", f"xg_{side}"):
        if key in node and node[key] is not None:
            return float(node[key])
    return None


def extract_xg(stats_data: dict | None) -> tuple[float | None, float | None]:
    if not stats_data:
        return None, None
    data = stats_data.get("data", stats_data)
    return _find_xg(data, "home"), _find_xg(data, "away")
