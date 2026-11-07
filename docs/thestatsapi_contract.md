# Contrato TheStatsAPI usado por este repo

Fuente oficial: https://api.thestatsapi.com/llms.txt (v1.0.0)

- Base URL: `https://api.thestatsapi.com/api`
- Auth: `Authorization: Bearer THESTATSAPI_KEY` en cada request.
- Rate limits: headers `X-RateLimit-Remaining` (por minuto) y `X-Monthly-Quota-Remaining`;
  `429` + `Retry-After` al excederse (el cliente espera y reintenta).
- IDs con prefijo: `comp_` competicion, `sn_` temporada, `tm_` equipo, `mt_` partido.

## Endpoints utilizados

| Endpoint | Uso en el pipeline |
|---|---|
| `GET /football/competitions?search=Major League Soccer` | Localizar `competition_id` de la MLS |
| `GET /football/competitions/{competition_id}` | `current_season_id` y flags (`odds_available`, `xg_available`) |
| `GET /football/competitions/{competition_id}/seasons` | Lista de temporadas con `is_current` |
| `GET /football/matches?competition_id&season_id&status=finished&per_page=100&page=N` | Resultados reales (score.regulation) |
| `GET /football/matches/{match_id}/odds` | Mercado `both_teams_to_score` (yes/no) |
| `GET /football/matches/{match_id}/stats` | xG por equipo donde `xg_available` |

## Target

`btts_yes = (score.regulation.home > 0) & (score.regulation.away > 0)`

## Regla anti-datos-ficticios

`src/fetch_data.py` SOLO escribe filas cuyo marcador viene del campo `score.regulation`
del endpoint real. Si un partido no tiene marcador, se omite. No existe ningun
generador de datos sinteticos en el repo.
