# Toronto Tempo 2026: a performance data pipeline on public WNBA data

Independent project by **Hugo Leligny**. Not affiliated with the Toronto Tempo or the WNBA. Public data only.

**Reports:** [hugollel.github.io/toronto-tempo-2026](https://hugollel.github.io/toronto-tempo-2026)
- Part 1, *Availability wins*: travel, injuries and availability across the Tempo's first season.
- Part 2, *Building the 2027 roster*: four factors, shot zones, shot creation and player profiles to discuss.

## What the pipeline does

| Step | Script | What it does |
|---|---|---|
| 1 | `pipeline/01_ingest.py` | Downloads schedules, team box scores, player box scores and play-by-play (ESPN, via SportsDataverse) for 2024–2026 as parquet. Logs every file (rows, size, md5). |
| 2 | `pipeline/02_load_db.py` | Loads the files into a DuckDB database (`raw` schema). Handles schema changes between seasons (`union_by_name`) and reports them. |
| 3 | `pipeline/03_transform.py` | Builds the `ref`, `clean` and `mart` layers in SQL (see the data dictionary below). |
| 4 | `pipeline/04_quality_checks.py` | Runs 14 SQL quality checks (FAIL / WARN). Results are appended to `data/quality_report.csv`; the script stops with an error if a FAIL check fails. |
| 5 | `pipeline/05_analysis.py` | Expected-performance model (least squares with team-season, opponent-season and home effects), travel and time-zone effects, road trips. |
| 6 | `pipeline/06_lineups.py` | Rebuilds the five players on the court for every possession from substitutions (stints), then on/off. Checks stint points against official scores. |
| 7 | `pipeline/07_season.py` | Availability, injuries, minutes, monthly splits, duos and trios. |
| 8 | `pipeline/08_roster.py` | Four factors, shot zones and shot charts, shot creation, player profiles; exports the data for the interactive shot explorer. |

Run in order: `python pipeline/01_ingest.py` … `python pipeline/08_roster.py`, then `quarto render report/`.
Requirements: Python 3.11+, pandas, numpy, duckdb, matplotlib, pyarrow; Quarto for the reports.

## Data model

```
raw    files as downloaded (schedule, team_box, player_box, pbp)
ref    reference tables written by hand (arena cities: coordinates, time zone, country)
clean  one row per game / team-game / player-game, typed and deduplicated
mart   analysis-ready tables (load, travel, ratings, stints, on/off)
```

All tables share the same keys: `game_id`, `team_id` / `team_abbr`, `athlete_id`, `season`, `game_date`. Internal club data (monitoring, wellness, wearables) can join on **player + date** without manual work.

## Data dictionary

### `ref.venue_cities`
| Column | Definition |
|---|---|
| city, state, country | Arena location |
| lat, lon | Coordinates used for straight-line travel distance |
| tz_offset_et | Time difference with Toronto / Eastern Time, in hours |

### `clean.games` — one row per game
| Column | Definition |
|---|---|
| game_id, season, game_date, tipoff | Identifiers and date / time |
| game_type, phase | Regular season, playoffs, etc. |
| home_team_id, away_team_id | Teams |
| venue, venue_city, venue_state | Where the game was played (Toronto also played "home" games in Vancouver and Montreal) |
| periods | 4, or more with overtime |

### `clean.team_games` — one row per team per game
| Column | Definition |
|---|---|
| team_abbr, team_name, is_home | Team and side |
| points, opp_points, win | Result |
| fgm, fga, ftm, fta, oreb, dreb, reb, ast, stl, blk, tov, pf | Box-score totals |
| paint_points, fast_break_points | ESPN team stats |
| possessions | FGA − OREB + TOV + 0.44 × FTA |

### `clean.player_games` — one row per player per game
| Column | Definition |
|---|---|
| athlete_id, player_name, position | Player |
| minutes, fgm, fga, ftm, fta, oreb, dreb, reb, ast, stl, blk, tov, pf, plus_minus | As published by ESPN |

### `mart.team_game_load` — one row per team per game, with context
| Column | Definition |
|---|---|
| prev_game_date, days_rest, back_to_back | Rest since the previous game |
| travel_km | Straight-line distance from the previous game's arena |
| tz_shift_h | Time-zone change since the previous game (positive = flying east) |
| crossed_border | Canada ↔ USA since the previous game |
| opp_days_rest, rest_advantage | Opponent's rest, and the difference |
| pace | Possessions per 40 minutes |
| off_rating, def_rating, net_rating | Points scored / allowed / difference per 100 possessions |
| efg_pct | Effective FG%: (FGM + 0.5 × 3PM) / FGA |

### `mart.player_game_load`
Player minutes per game, minutes over the last 7 days, with the team's rest and travel context.

### `mart.team_season_load`
One row per team-season: games, back_to_backs, total_travel_km, border_crossings, time_zones_crossed, avg_net_rating.

### `mart.stints`, `mart.stint_players`, `mart.player_on_off`
| Table | Definition |
|---|---|
| stints | One row per stretch of play with the same ten players: start / end time, points for and against, possessions |
| stint_players | The five players of each team in each stint |
| player_on_off | For each player: net rating with her on the court vs off, minutes, possessions |

## Quality checks (`pipeline/04_quality_checks.py`)
| Level | Check |
|---|---|
| FAIL | games: one row per game_id |
| FAIL | team_games: one row per team per game |
| FAIL | player_games: one row per player per game |
| FAIL | every game has exactly 2 teams |
| FAIL | player_games: no missing ids |
| FAIL | schedule score = team box score |
| FAIL | sum of player points = team points |
| FAIL | rest days never negative |
| WARN | team minutes = 200 (+25 per overtime), ± 5 |
| WARN | possessions between 55 and 110 |
| WARN | every venue city has coordinates |
| WARN | every game has player stats |
| WARN | completed games missing from team box (this is how the missing Finals Game 7 was flagged) |
| WARN | duplicate rows removed from raw player box |

The lineup rebuild (`06_lineups.py`) adds its own check: points scored during the rebuilt stints must equal each game's official score.

## Limits
Public data shows associations, not causes. Absence reasons come from ESPN and can be incomplete. Travel distances are straight lines. See the "Limits" sections of both reports.
