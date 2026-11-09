"""
04_quality_checks.py — Data validation
======================================
Runs a set of automatic checks on the database. Each check is a SQL query
that counts problem rows: 0 means the check passes.

  FAIL  = the data cannot be trusted (broken keys, scores that don't match)
  WARN  = worth a look, but not blocking (unusual values, unmapped cities)

Results are printed and appended to data/quality_report.csv, so the history
of every run is kept. The script exits with an error code if any FAIL check
fails, so it can stop an automated pipeline before bad data reaches users.

Run from the project folder (after 03):
    python3 pipeline/04_quality_checks.py
"""

from datetime import datetime, timezone
from pathlib import Path
import sys

import duckdb
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_FILE = PROJECT_DIR / "data" / "wnba.duckdb"
REPORT_FILE = PROJECT_DIR / "data" / "quality_report.csv"

# (severity, name, SQL returning the number of problem rows)
CHECKS = [
    # --- Keys: one row per entity ------------------------------------------
    ("FAIL", "games: one row per game_id", """
        SELECT COUNT(*) - COUNT(DISTINCT game_id) FROM clean.games
    """),
    ("FAIL", "team_games: one row per team per game", """
        SELECT COUNT(*) FROM (
            SELECT game_id, team_id FROM clean.team_games
            GROUP BY game_id, team_id HAVING COUNT(*) > 1
        )
    """),
    ("FAIL", "player_games: one row per player per game", """
        SELECT COUNT(*) FROM (
            SELECT game_id, athlete_id FROM clean.player_games
            GROUP BY game_id, athlete_id HAVING COUNT(*) > 1
        )
    """),
    ("FAIL", "every game has exactly 2 teams", """
        SELECT COUNT(*) FROM (
            SELECT g.game_id, COUNT(tg.team_id) AS n_teams
            FROM clean.games g
            LEFT JOIN clean.team_games tg ON tg.game_id = g.game_id
            GROUP BY g.game_id
        ) WHERE n_teams <> 2
    """),
    ("FAIL", "player_games: no missing ids", """
        SELECT COUNT(*) FROM clean.player_games
        WHERE athlete_id IS NULL OR team_id IS NULL OR game_id IS NULL
    """),

    # --- Reconciliation between sources -----------------------------------
    ("FAIL", "schedule score = team box score", """
        SELECT COUNT(*)
        FROM clean.games g
        JOIN clean.team_games tg ON tg.game_id = g.game_id
        WHERE tg.points <> CASE WHEN tg.team_id = g.home_team_id
                                THEN g.home_score ELSE g.away_score END
    """),
    ("FAIL", "sum of player points = team points", """
        SELECT COUNT(*) FROM (
            SELECT tg.game_id, tg.team_id, tg.points,
                   COALESCE(SUM(pg.points), 0) AS player_points
            FROM clean.team_games tg
            LEFT JOIN clean.player_games pg
                   ON pg.game_id = tg.game_id AND pg.team_id = tg.team_id
            GROUP BY tg.game_id, tg.team_id, tg.points
        ) WHERE player_points <> points
    """),

    # --- Plausibility ------------------------------------------------------
    ("WARN", "team minutes = 200 (+25 per OT), +/- 5", """
        SELECT COUNT(*) FROM (
            SELECT pg.game_id, pg.team_id, g.periods, SUM(pg.minutes) AS minutes
            FROM clean.player_games pg
            JOIN clean.games g ON g.game_id = pg.game_id
            GROUP BY pg.game_id, pg.team_id, g.periods
        ) WHERE ABS(minutes - (200 + 25 * (periods - 4))) > 5
    """),
    ("WARN", "possessions between 55 and 110", """
        SELECT COUNT(*) FROM clean.team_games
        WHERE possessions NOT BETWEEN 55 AND 110
    """),
    ("FAIL", "rest days never negative", """
        SELECT COUNT(*) FROM mart.team_game_load WHERE days_rest < 0
    """),

    # --- Completeness ------------------------------------------------------
    ("WARN", "every venue city has coordinates", """
        SELECT COUNT(*)
        FROM clean.games g
        LEFT JOIN ref.venue_cities v
               ON v.city = g.venue_city AND v.state = g.venue_state
        WHERE v.city IS NULL
    """),
    ("WARN", "every game has player stats", """
        SELECT COUNT(*) FROM clean.games g
        WHERE NOT EXISTS (
            SELECT 1 FROM clean.player_games pg WHERE pg.game_id = g.game_id
        )
    """),
    ("WARN", "completed games missing from team box", """
        SELECT COUNT(*) FROM raw.schedule s
        WHERE s.status_type_completed
          AND s.season_type IN (2, 3)
          AND s.type_abbreviation <> 'ALLSTAR'
          AND s.game_id NOT IN (SELECT game_id FROM raw.team_box)
    """),
    ("WARN", "duplicate rows removed from raw player box", """
        SELECT COUNT(*) - (SELECT COUNT(*) FROM (
            SELECT DISTINCT game_id, athlete_id FROM raw.player_box
        )) FROM raw.player_box
    """),
]


def main() -> None:
    con = duckdb.connect(str(DB_FILE), read_only=True)
    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results = []

    for severity, name, sql in CHECKS:
        try:
            n = con.execute(sql).fetchone()[0]
            status = "PASS" if n == 0 else severity
        except Exception as e:  # a broken check must be visible, not silent
            n, status = None, "ERROR"
            print(f"   error in '{name}': {e}")
        results.append({"run_at": run_at, "check": name, "severity": severity,
                        "status": status, "problem_rows": n})
        print(f"{status:<5}  {name:<46} {'' if n is None else n}")

    con.close()

    report = pd.DataFrame(results)
    report.to_csv(REPORT_FILE, mode="a", header=not REPORT_FILE.exists(), index=False)

    n_fail = report["status"].isin(["FAIL", "ERROR"]).sum()
    n_warn = (report["status"] == "WARN").sum()
    print(f"\n{len(report) - n_fail - n_warn} passed, {n_warn} warnings, {n_fail} failures"
          f"  ->  {REPORT_FILE.relative_to(PROJECT_DIR)}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
