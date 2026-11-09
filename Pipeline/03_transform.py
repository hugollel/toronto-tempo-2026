"""
03_transform.py — Clean, model and enrich the data (SQL in DuckDB)
==================================================================
Builds three layers on top of the raw tables:

  ref.*    reference data maintained by hand (venue city coordinates,
           time zone, country) - needed because the source has no GPS data
  clean.*  one tidy table per entity, with clear names and types:
             clean.games         1 row = 1 completed game (All-Star excluded)
             clean.teams         1 row = 1 team
             clean.team_games    1 row = 1 team in 1 game
             clean.player_games  1 row = 1 player who played in 1 game
  mart.*   analysis-ready tables (schedule load + performance):
             mart.team_game_load    rest days, back-to-backs, travel km,
                                    time zone shifts, border crossings,
                                    rolling 7-day load, net rating
             mart.player_game_load  player minutes, 7-day minutes,
                                    game score + team schedule context
             mart.team_season_load  season summary per team

Run from the project folder (after 01 and 02):
    python3 pipeline/03_transform.py
"""

from pathlib import Path

import duckdb

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_FILE = PROJECT_DIR / "data" / "wnba.duckdb"


# --- 1. Reference data --------------------------------------------------------
# Approximate arena coordinates for every city where a WNBA game was played
# 2024-2026. tz_offset_et = hours vs Eastern Time during the WNBA season
# (summer), e.g. Seattle = -3. Phoenix does not use daylight saving, so in
# summer it is on the same clock as the Pacific coast (-3).

REF_VENUE_CITIES = """
CREATE OR REPLACE TABLE ref.venue_cities AS
SELECT * FROM (VALUES
    ('Arlington',     'TX', 'US', 32.7357,  -97.1081, -1),
    ('Atlanta',       'GA', 'US', 33.7573,  -84.3963,  0),
    ('Baltimore',     'MD', 'US', 39.2884,  -76.6200,  0),
    ('Boston',        'MA', 'US', 42.3662,  -71.0621,  0),
    ('Brooklyn',      'NY', 'US', 40.6826,  -73.9754,  0),
    ('Chicago',       'IL', 'US', 41.8676,  -87.6170, -1),
    ('College Park',  'GA', 'US', 33.6537,  -84.4489,  0),
    ('Dallas',        'TX', 'US', 32.7905,  -96.8103, -1),
    ('Elmont',        'NY', 'US', 40.7170,  -73.7255,  0),
    ('Fairfax',       'VA', 'US', 38.8304,  -77.3075,  0),
    ('Hartford',      'CT', 'US', 41.7680,  -72.6766,  0),
    ('Indianapolis',  'IN', 'US', 39.7640,  -86.1555,  0),
    ('Las Vegas',     'NV', 'US', 36.0907, -115.1780, -3),
    ('Long Beach',    'CA', 'US', 33.7701, -118.1937, -3),
    ('Los Angeles',   'CA', 'US', 34.0430, -118.2673, -3),
    ('Minneapolis',   'MN', 'US', 44.9795,  -93.2761, -1),
    ('Montreal',      'PQ', 'CA', 45.4961,  -73.5693,  0),
    ('Phoenix',       'AZ', 'US', 33.4457, -112.0712, -3),
    ('Portland',      'OR', 'US', 45.5316, -122.6668, -3),
    ('San Francisco', 'CA', 'US', 37.7680, -122.3877, -3),
    ('San Jose',      'CA', 'US', 37.3327, -121.9010, -3),
    ('Seattle',       'WA', 'US', 47.6221, -122.3540, -3),
    ('Toronto',       'ON', 'CA', 43.6435,  -79.3791,  0),
    ('Uncasville',    'CT', 'US', 41.4911,  -72.0886,  0),
    ('Vancouver',     'BC', 'CA', 49.2778, -123.1088, -3),
    ('Washington',    'DC', 'US', 38.8690,  -76.9940,  0)
) AS t(city, state, country, lat, lon, tz_offset_et)
"""


# --- 2. Clean layer -----------------------------------------------------------

CLEAN_GAMES = """
CREATE OR REPLACE TABLE clean.games AS
SELECT
    game_id,
    season,
    season_type,
    CASE season_type WHEN 2 THEN 'regular' WHEN 3 THEN 'playoffs' END AS phase,
    type_abbreviation               AS game_type,
    game_date,
    game_date_time                  AS tipoff,
    home_id                         AS home_team_id,
    away_id                         AS away_team_id,
    home_score,
    away_score,
    venue_full_name                 AS venue,
    venue_address_city              AS venue_city,
    venue_address_state             AS venue_state,
    neutral_site,
    CAST(COALESCE(status_period, 4) AS INTEGER) AS periods,
    attendance
FROM raw.schedule
WHERE status_type_completed                -- played games only
  AND season_type IN (2, 3)                -- regular season + playoffs
  AND type_abbreviation <> 'ALLSTAR'       -- All-Star Game is not a real team game
QUALIFY ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY filename) = 1
"""

CLEAN_TEAMS = """
CREATE OR REPLACE TABLE clean.teams AS
SELECT
    team_id,
    arg_max(team_abbreviation, game_date) AS team_abbr,   -- most recent name wins
    arg_max(team_display_name, game_date) AS team_name
FROM raw.team_box
WHERE game_id IN (SELECT game_id FROM clean.games)
GROUP BY team_id
"""

CLEAN_TEAM_GAMES = """
CREATE OR REPLACE TABLE clean.team_games AS
SELECT
    tb.game_id,
    g.season,
    g.phase,
    g.game_date,
    g.tipoff,
    tb.team_id,
    tb.opponent_team_id,
    tb.team_home_away = 'home'                   AS is_home,
    tb.team_score                                AS points,
    tb.opponent_team_score                       AS opp_points,
    tb.team_winner                               AS win,
    tb.field_goals_made                          AS fgm,
    tb.field_goals_attempted                     AS fga,
    tb.three_point_field_goals_made              AS fg3m,
    tb.three_point_field_goals_attempted         AS fg3a,
    tb.free_throws_made                          AS ftm,
    tb.free_throws_attempted                     AS fta,
    tb.offensive_rebounds                        AS oreb,
    tb.defensive_rebounds                        AS dreb,
    tb.total_rebounds                            AS reb,
    tb.assists                                   AS ast,
    tb.steals                                    AS stl,
    tb.blocks                                    AS blk,
    tb.total_turnovers                           AS tov,
    tb.fouls                                     AS pf,
    TRY_CAST(tb.points_in_paint AS INTEGER)      AS paint_points,     -- stored as text in the source
    TRY_CAST(tb.fast_break_points AS INTEGER)    AS fast_break_points,
    -- Standard possession estimate
    tb.field_goals_attempted - tb.offensive_rebounds + tb.total_turnovers
        + 0.44 * tb.free_throws_attempted        AS possessions
FROM raw.team_box tb
JOIN clean.games g ON g.game_id = tb.game_id
QUALIFY ROW_NUMBER() OVER (PARTITION BY tb.game_id, tb.team_id ORDER BY tb.filename) = 1
"""

CLEAN_PLAYER_GAMES = """
CREATE OR REPLACE TABLE clean.player_games AS
SELECT
    pb.game_id,
    g.season,
    g.phase,
    g.game_date,
    pb.athlete_id,
    pb.athlete_display_name                      AS player_name,
    pb.athlete_position_abbreviation             AS position,
    pb.team_id,
    pb.opponent_team_id,
    pb.starter,
    pb.minutes,
    pb.points,
    pb.field_goals_made                          AS fgm,
    pb.field_goals_attempted                     AS fga,
    pb.three_point_field_goals_made              AS fg3m,
    pb.three_point_field_goals_attempted         AS fg3a,
    pb.free_throws_made                          AS ftm,
    pb.free_throws_attempted                     AS fta,
    pb.offensive_rebounds                        AS oreb,
    pb.defensive_rebounds                        AS dreb,
    pb.rebounds                                  AS reb,
    pb.assists                                   AS ast,
    pb.steals                                    AS stl,
    pb.blocks                                    AS blk,
    pb.turnovers                                 AS tov,
    pb.fouls                                     AS pf,
    -- stored as text like '-15.0' in the source
    CAST(TRY_CAST(pb.plus_minus AS DOUBLE) AS INTEGER) AS plus_minus
FROM raw.player_box pb
JOIN clean.games g ON g.game_id = pb.game_id
WHERE NOT COALESCE(pb.did_not_play, false)
  AND COALESCE(pb.minutes, 0) > 0
QUALIFY ROW_NUMBER() OVER (PARTITION BY pb.game_id, pb.athlete_id ORDER BY pb.filename) = 1
"""


# --- 3. Mart layer ------------------------------------------------------------

MART_TEAM_GAME_LOAD = """
CREATE OR REPLACE TABLE mart.team_game_load AS
WITH base AS (
    SELECT
        tg.*,
        g.venue_city,
        g.venue_state,
        v.lat, v.lon, v.tz_offset_et, v.country
    FROM clean.team_games tg
    JOIN clean.games g       ON g.game_id = tg.game_id
    LEFT JOIN ref.venue_cities v
           ON v.city = g.venue_city AND v.state = g.venue_state
),
-- previous game of the same team, same season
seq AS (
    SELECT
        *,
        LAG(game_date)    OVER w AS prev_game_date,
        LAG(lat)          OVER w AS prev_lat,
        LAG(lon)          OVER w AS prev_lon,
        LAG(tz_offset_et) OVER w AS prev_tz,
        LAG(country)      OVER w AS prev_country
    FROM base
    WINDOW w AS (PARTITION BY team_id, season ORDER BY tipoff)
),
feat AS (
    SELECT
        *,
        DATE_DIFF('day', prev_game_date, game_date) - 1 AS days_rest,     -- 0 = back-to-back
        DATE_DIFF('day', prev_game_date, game_date) = 1 AS back_to_back,
        -- great-circle distance between the previous venue and this one (km)
        2 * 6371 * ASIN(SQRT(
              POW(SIN(RADIANS(lat - prev_lat) / 2), 2)
            + COS(RADIANS(prev_lat)) * COS(RADIANS(lat))
            * POW(SIN(RADIANS(lon - prev_lon) / 2), 2)
        ))                                              AS travel_km,
        tz_offset_et - prev_tz                          AS tz_shift_h,
        country <> prev_country                         AS crossed_border
    FROM seq
),
-- rolling 7-day load (this game + the 6 days before)
roll AS (
    SELECT
        *,
        COUNT(*)                     OVER r AS games_last_7d,
        COALESCE(SUM(travel_km) OVER r, 0) AS km_last_7d
    FROM feat
    WINDOW r AS (
        PARTITION BY team_id, season ORDER BY game_date
        RANGE BETWEEN INTERVAL 6 DAY PRECEDING AND CURRENT ROW
    )
)
SELECT
    r.game_id,
    r.season,
    r.phase,
    r.game_date,
    r.team_id,
    t.team_abbr,
    r.opponent_team_id,
    ot.team_abbr                                AS opponent_abbr,
    r.is_home,
    r.venue_city,
    -- schedule load
    r.days_rest,
    r.back_to_back,
    ROUND(r.travel_km)                          AS travel_km,
    r.tz_shift_h,
    r.crossed_border,
    r.games_last_7d,
    ROUND(r.km_last_7d)                         AS km_last_7d,
    o.days_rest                                 AS opp_days_rest,
    r.days_rest - o.days_rest                   AS rest_advantage,
    -- performance
    r.points,
    r.opp_points,
    r.win,
    ROUND((r.possessions + o.possessions) / 2, 1)                              AS pace,
    ROUND(100.0 * r.points     / ((r.possessions + o.possessions) / 2), 1)     AS off_rating,
    ROUND(100.0 * r.opp_points / ((r.possessions + o.possessions) / 2), 1)     AS def_rating,
    ROUND(100.0 * (r.points - r.opp_points)
                 / ((r.possessions + o.possessions) / 2), 1)                   AS net_rating,
    ROUND((r.fgm + 0.5 * r.fg3m) / NULLIF(r.fga, 0), 3)                        AS efg_pct
FROM roll r
JOIN roll o            ON o.game_id = r.game_id AND o.team_id = r.opponent_team_id
LEFT JOIN clean.teams t  ON t.team_id  = r.team_id
LEFT JOIN clean.teams ot ON ot.team_id = r.opponent_team_id
"""

MART_PLAYER_GAME_LOAD = """
CREATE OR REPLACE TABLE mart.player_game_load AS
SELECT
    pg.*,
    tl.team_abbr,
    tl.days_rest       AS team_days_rest,
    tl.back_to_back,
    tl.travel_km,
    tl.tz_shift_h,
    tl.crossed_border,
    tl.games_last_7d   AS team_games_last_7d,
    SUM(pg.minutes) OVER (
        PARTITION BY pg.athlete_id, pg.season ORDER BY pg.game_date
        RANGE BETWEEN INTERVAL 6 DAY PRECEDING AND CURRENT ROW
    )                  AS minutes_last_7d,
    -- Hollinger game score: one-number summary of a box score line
    ROUND(pg.points + 0.4 * pg.fgm - 0.7 * pg.fga - 0.4 * (pg.fta - pg.ftm)
          + 0.7 * pg.oreb + 0.3 * pg.dreb + pg.stl + 0.7 * pg.ast
          + 0.7 * pg.blk - 0.4 * pg.pf - pg.tov, 1) AS game_score
FROM clean.player_games pg
JOIN mart.team_game_load tl ON tl.game_id = pg.game_id AND tl.team_id = pg.team_id
"""

MART_TEAM_SEASON_LOAD = """
CREATE OR REPLACE TABLE mart.team_season_load AS
SELECT
    season,
    team_abbr,
    COUNT(*)                               AS games,
    SUM(CAST(back_to_back AS INTEGER))     AS back_to_backs,
    ROUND(SUM(travel_km))                  AS total_travel_km,
    SUM(CAST(crossed_border AS INTEGER))   AS border_crossings,
    SUM(ABS(tz_shift_h))                   AS time_zones_crossed,
    ROUND(AVG(net_rating), 1)              AS avg_net_rating
FROM mart.team_game_load
WHERE phase = 'regular'                    -- same number of games for everyone
GROUP BY season, team_abbr
"""


STEPS = [
    ("ref.venue_cities",       REF_VENUE_CITIES),
    ("clean.games",            CLEAN_GAMES),
    ("clean.teams",            CLEAN_TEAMS),
    ("clean.team_games",       CLEAN_TEAM_GAMES),
    ("clean.player_games",     CLEAN_PLAYER_GAMES),
    ("mart.team_game_load",    MART_TEAM_GAME_LOAD),
    ("mart.player_game_load",  MART_PLAYER_GAME_LOAD),
    ("mart.team_season_load",  MART_TEAM_SEASON_LOAD),
]


def main() -> None:
    con = duckdb.connect(str(DB_FILE))
    for schema in ("ref", "clean", "mart"):
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    for table, sql in STEPS:
        con.execute(sql)
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"OK  {table:<24} {n:>6} rows")

    # First look at the question: who travels the most?
    print("\n2026 regular season - schedule load (top 5 by travel + Toronto):")
    df = con.execute("""
        SELECT team_abbr, games, back_to_backs, total_travel_km,
               border_crossings, time_zones_crossed, avg_net_rating
        FROM mart.team_season_load
        WHERE season = 2026
        QUALIFY ROW_NUMBER() OVER (ORDER BY total_travel_km DESC) <= 5
             OR team_abbr = 'TOR'
        ORDER BY total_travel_km DESC
    """).df()
    print(df.to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()
