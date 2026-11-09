"""
06_lineups.py — Who is on the court? (lineups, stints, on/off)
==============================================================
The play-by-play lists every event (shots, rebounds, fouls...) and every
substitution, but never says directly which 5 players are on the court.
This script rebuilds it, then measures each player's on/off impact.

How the 5 on court are rebuilt, period by period:
  1. A player is "on court at the start of the period" if the first thing
     we see her do in that period is anything except entering the game
     (she shoots, rebounds, gets subbed OUT...).
  2. A player who plays a whole period without any recorded action is
     invisible. If fewer than 5 starters are found, we fill with the players
     who were on court at the end of the previous period (or the box score
     starters for the 1st period).
  3. Then we walk through the events: each substitution swaps one player.

Each stretch of time with the same 10 players is a "stint". For every stint
we count seconds, points and possessions for both teams.

Quality checks (printed at the end):
  - share of stints with exactly 5 players per team
  - rebuilt minutes vs official box score minutes, per player
  - rebuilt points vs official final score, per game

Tables written to the database:
  mart.stints          1 row = 1 stint for 1 team (lineup, time, points, possessions)
  mart.stint_players   1 row = 1 player on court in 1 stint
  mart.player_on_off   on/off summary per player (team net rating with / without her)

Run from the project folder (after 01, 02 with play-by-play loaded):
    python3 pipeline/06_lineups.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_FILE = PROJECT_DIR / "data" / "wnba.duckdb"
TAB_DIR = PROJECT_DIR / "outputs" / "tables"

SEASONS = [2026]
MIN_MINUTES_ON_OFF = 200      # only report on/off for players with enough minutes

# Events that say nothing about who is on the court (bench techs, reviews...)
IGNORE_FOR_PRESENCE = ("Technical", "ejection", "Timeout", "Challenge", "Review",
                       "End Period", "End Game", "Delay")


# =============================================================================
# 1. Prepare events
# =============================================================================

def prepare_events(pbp: pd.DataFrame) -> pd.DataFrame:
    e = pbp.copy()
    e = e.sort_values(["game_id", "game_play_number"]).reset_index(drop=True)
    e["type_text"] = e["type_text"].fillna("")
    period = e["period_number"].astype(int)
    period_len = np.where(period <= 4, 600, 300)
    period_start = np.where(period <= 4, (period - 1) * 600, 2400 + (period - 5) * 300)
    clock = e["clock_minutes"].fillna(0).astype(float) * 60 + e["clock_seconds"].fillna(0).astype(float)
    e["elapsed_s"] = period_start + period_len - clock

    # points scored on this event, by side
    g = e.groupby("game_id")
    e["home_pts"] = (e["home_score"] - g["home_score"].shift().fillna(0))
    e["away_pts"] = (e["away_score"] - g["away_score"].shift().fillna(0))

    t = e["type_text"]
    is_ft = t.str.contains("Free Throw")
    e["fga"] = (e["shooting_play"].astype(str).str.lower().eq("true") & ~is_ft).astype(int)
    e["fta"] = is_ft.astype(int)
    e["oreb"] = (t == "Offensive Rebound").astype(int)
    e["tov"] = (t.str.contains("Turnover") & (t != "No Turnover")).astype(int)
    e["is_sub"] = t == "Substitution"
    e["presence_ok"] = ~t.str.contains("|".join(IGNORE_FOR_PRESENCE), case=False)
    return e


# =============================================================================
# 2. Rebuild lineups and stints for one game
# =============================================================================

def _period_starters(ev, team_players, previous_end, box_starters, period):
    """Players on court at the start of a period for one team."""
    first_seen = {}   # player -> "in" (entered by sub) or "on" (already on court)
    for row in ev.itertuples(index=False):
        if row.is_sub:
            pin, pout = row.athlete_id_1, row.athlete_id_2
            if pout in team_players and pout not in first_seen:
                first_seen[pout] = "on"
            if pin in team_players and pin not in first_seen:
                first_seen[pin] = "in"
        elif row.presence_ok:
            for pid in (row.athlete_id_1, row.athlete_id_2, row.athlete_id_3):
                if pid in team_players and pid not in first_seen:
                    first_seen[pid] = "on"
    starters = [p for p, how in first_seen.items() if how == "on"]
    fallback = box_starters if period == 1 else previous_end
    for p in fallback:
        if len(starters) >= 5:
            break
        if p not in starters and first_seen.get(p) != "in":
            starters.append(p)
    return starters[:5] if len(starters) > 5 else starters


def build_game_stints(ev: pd.DataFrame, roster: dict, box_starters: dict,
                      home_id: int, away_id: int) -> pd.DataFrame:
    """roster: team_id -> set of athlete ids. Returns one row per stint per team."""
    rows = []
    end_lineup = {home_id: [], away_id: []}
    for period, pev in ev.groupby("period_number", sort=True):
        lineup = {}
        for tid in (home_id, away_id):
            lineup[tid] = _period_starters(pev, roster[tid], end_lineup[tid],
                                           box_starters[tid], period)
        period_len = 600 if period <= 4 else 300
        period_start = (period - 1) * 600 if period <= 4 else 2400 + (period - 5) * 300
        t_prev = period_start
        acc = None

        def open_stint(t0):
            return {"start_s": t0, "home": tuple(sorted(lineup[home_id])),
                    "away": tuple(sorted(lineup[away_id])),
                    "home_pts": 0.0, "away_pts": 0.0,
                    "home_fga": 0, "away_fga": 0, "home_fta": 0, "away_fta": 0,
                    "home_oreb": 0, "away_oreb": 0, "home_tov": 0, "away_tov": 0}

        acc = open_stint(t_prev)
        for row in pev.itertuples(index=False):
            side = "home" if row.team_id == home_id else "away" if row.team_id == away_id else None
            acc["home_pts"] += row.home_pts
            acc["away_pts"] += row.away_pts
            if side:
                for stat in ("fga", "fta", "oreb", "tov"):
                    acc[f"{side}_{stat}"] += getattr(row, stat)
            if row.is_sub:
                tid = home_id if row.athlete_id_1 in roster[home_id] else \
                      away_id if row.athlete_id_1 in roster[away_id] else None
                if tid is None:
                    continue
                pin, pout = row.athlete_id_1, row.athlete_id_2
                # close the current stint at the substitution time
                acc["end_s"] = row.elapsed_s
                rows.append(acc)
                if pout in lineup[tid]:
                    lineup[tid].remove(pout)
                if pin not in lineup[tid]:
                    lineup[tid].append(pin)
                acc = open_stint(row.elapsed_s)
        acc["end_s"] = period_start + period_len
        rows.append(acc)
        end_lineup = {k: list(v) for k, v in lineup.items()}

    out = []
    for i, s in enumerate(rows):
        secs = s["end_s"] - s["start_s"]
        for side, opp, tid, oid in (("home", "away", home_id, away_id),
                                    ("away", "home", away_id, home_id)):
            out.append({
                "stint_no": i, "team_id": tid, "opponent_team_id": oid,
                "start_s": s["start_s"], "end_s": s["end_s"], "seconds": secs,
                "lineup": "-".join(str(int(p)) for p in s[side]),
                "n_players": len(s[side]),
                "opp_lineup": "-".join(str(int(p)) for p in s[opp]),
                "pts_for": s[f"{side}_pts"], "pts_against": s[f"{opp}_pts"],
                "poss_for": s[f"{side}_fga"] - s[f"{side}_oreb"] + s[f"{side}_tov"] + 0.44 * s[f"{side}_fta"],
                "poss_against": s[f"{opp}_fga"] - s[f"{opp}_oreb"] + s[f"{opp}_tov"] + 0.44 * s[f"{opp}_fta"],
            })
    return pd.DataFrame(out)


def build_stints(pbp: pd.DataFrame, player_box: pd.DataFrame) -> pd.DataFrame:
    ev = prepare_events(pbp)
    pb = player_box[player_box["minutes"].fillna(0) > 0]
    all_stints = []
    for game_id, gev in ev.groupby("game_id", sort=False):
        gpb = pb[pb["game_id"] == game_id]
        if gpb.empty:
            continue
        home_id = int(gev["home_team_id"].iloc[0])
        away_id = int(gev["away_team_id"].iloc[0])
        roster = {tid: set(gpb.loc[gpb["team_id"] == tid, "athlete_id"]) for tid in (home_id, away_id)}
        starters = {tid: list(gpb.loc[(gpb["team_id"] == tid) & gpb["starter"].fillna(False).astype(bool),
                                       "athlete_id"]) for tid in (home_id, away_id)}
        st = build_game_stints(gev, roster, starters, home_id, away_id)
        st.insert(0, "game_id", game_id)
        st.insert(1, "season", int(gev["season"].iloc[0]))
        st.insert(2, "game_date", gev["game_date"].iloc[0])
        all_stints.append(st)
    stints = pd.concat(all_stints, ignore_index=True)
    # Zero-second stints are kept on purpose: free throws taken between two
    # substitutions at the same clock time belong to them.
    stints = stints.reset_index(drop=True)
    stints.insert(0, "stint_id", np.arange(len(stints)))
    return stints


# =============================================================================
# 3. Player tables and on/off
# =============================================================================

def explode_players(stints: pd.DataFrame) -> pd.DataFrame:
    sp = stints[["stint_id", "game_id", "team_id", "lineup"]].copy()
    sp["athlete_id"] = sp["lineup"].str.split("-")
    sp = sp.explode("athlete_id").dropna(subset=["athlete_id"])
    sp = sp[sp["athlete_id"] != ""]
    sp["athlete_id"] = sp["athlete_id"].astype(int)
    return sp[["stint_id", "game_id", "team_id", "athlete_id"]]


def on_off(stints: pd.DataFrame, stint_players: pd.DataFrame,
           player_box: pd.DataFrame) -> pd.DataFrame:
    names = (player_box.sort_values("game_date")
             .groupby("athlete_id")["athlete_display_name"].last())
    rows = []
    for (team_id, season), ts in stints.groupby(["team_id", "season"]):
        sp = stint_players[stint_players["stint_id"].isin(ts["stint_id"])]
        games_played = sp.groupby("athlete_id")["game_id"].nunique()
        for pid, pstints in sp.groupby("athlete_id"):
            on_ids = set(pstints["stint_id"])
            # "off" = only games she played in, so absences don't count as "off"
            in_games = ts[ts["game_id"].isin(pstints["game_id"])]
            on = in_games[in_games["stint_id"].isin(on_ids)]
            off = in_games[~in_games["stint_id"].isin(on_ids)]

            def net(d):
                poss = (d["poss_for"].sum() + d["poss_against"].sum()) / 2
                return 100 * (d["pts_for"].sum() - d["pts_against"].sum()) / poss if poss > 0 else np.nan, poss

            net_on, poss_on = net(on)
            net_off, poss_off = net(off)
            rows.append({"season": season, "team_id": team_id, "athlete_id": pid,
                         "player_name": names.get(pid, str(pid)),
                         "games": int(games_played.get(pid, 0)),
                         "minutes_on": round(on["seconds"].sum() / 60, 1),
                         "possessions_on": round(poss_on), "possessions_off": round(poss_off),
                         "net_on": round(net_on, 1), "net_off": round(net_off, 1),
                         "on_off_diff": round(net_on - net_off, 1)})
    return pd.DataFrame(rows)


# =============================================================================
# 4. Quality checks
# =============================================================================

def quality_checks(stints, stint_players, player_box, pbp) -> pd.DataFrame:
    checks = []
    w = stints["seconds"]
    share5 = (w * (stints["n_players"] == 5)).sum() / w.sum()
    checks.append(("time with exactly 5 players on court (%)", round(100 * share5, 2)))

    rebuilt_min = (stint_players.merge(stints[["stint_id", "seconds"]], on="stint_id")
                   .groupby(["game_id", "athlete_id"])["seconds"].sum() / 60).rename("rebuilt")
    official = player_box.set_index(["game_id", "athlete_id"])["minutes"].rename("official")
    cmp = pd.concat([rebuilt_min, official], axis=1, join="inner").dropna()
    diff = (cmp["rebuilt"] - cmp["official"]).abs()
    checks.append(("player-games compared", len(cmp)))
    checks.append(("median minutes gap vs box score", round(diff.median(), 2)))
    checks.append(("player-games within 2 min of box score (%)", round(100 * (diff <= 2).mean(), 1)))

    pts = stints.groupby(["game_id", "team_id"])["pts_for"].sum()
    final = (pbp.sort_values("game_play_number").groupby("game_id")
             .agg(home_team_id=("home_team_id", "first"), away_team_id=("away_team_id", "first"),
                  home_score=("home_score", "last"), away_score=("away_score", "last")))
    fin = pd.concat([
        final.reset_index().rename(columns={"home_team_id": "team_id", "home_score": "score"})[["game_id", "team_id", "score"]],
        final.reset_index().rename(columns={"away_team_id": "team_id", "away_score": "score"})[["game_id", "team_id", "score"]],
    ]).set_index(["game_id", "team_id"])["score"]
    j = pd.concat([pts, fin], axis=1, join="inner")
    checks.append(("team-games where stint points = final score (%)",
                   round(100 * (j.iloc[:, 0] == j.iloc[:, 1]).mean(), 1)))
    return pd.DataFrame(checks, columns=["check", "value"])


# =============================================================================
# 5. Main
# =============================================================================

def run(pbp, player_box):
    pbp = pbp[pbp["season"].isin(SEASONS)]
    player_box = player_box[player_box["season"].isin(SEASONS)]
    stints = build_stints(pbp, player_box)
    stint_players = explode_players(stints)
    onoff = on_off(stints, stint_players, player_box)
    qc = quality_checks(stints, stint_players, player_box, pbp)
    return stints, stint_players, onoff, qc


def main() -> None:
    import duckdb
    con = duckdb.connect(str(DB_FILE))
    seasons = ", ".join(str(s) for s in SEASONS)
    pbp = con.execute(f"SELECT * FROM raw.pbp WHERE season IN ({seasons})").df()
    player_box = con.execute(f"""
        SELECT game_id, season, game_date, athlete_id, athlete_display_name,
               team_id, team_abbreviation, minutes, starter
        FROM raw.player_box
        WHERE season IN ({seasons}) AND NOT COALESCE(did_not_play, false)
    """).df()

    # database nullable types -> plain floats / objects (safe comparisons)
    for df in (pbp, player_box):
        for col in df.columns:
            if str(df[col].dtype) in ("Int8", "Int16", "Int32", "Int64", "Float64"):
                df[col] = df[col].astype(float)
            elif str(df[col].dtype) == "boolean":
                df[col] = df[col].astype(object).where(df[col].notna(), np.nan)

    print(f"Rebuilding lineups from {len(pbp):,} events...")
    stints, stint_players, onoff, qc = run(pbp, player_box)

    con.execute("CREATE SCHEMA IF NOT EXISTS mart")
    for name, df in (("stints", stints), ("stint_players", stint_players), ("player_on_off", onoff)):
        con.register("tmp_df", df)
        con.execute(f"CREATE OR REPLACE TABLE mart.{name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
        print(f"OK  mart.{name:<15} {len(df):>7} rows")

    print("\nQuality checks:")
    print(qc.to_string(index=False))

    tor = con.execute("SELECT team_id FROM clean.teams WHERE team_abbr = 'TOR'").fetchone()
    con.close()

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    qc.to_csv(TAB_DIR / "lineup_quality_checks.csv", index=False)
    onoff.to_csv(TAB_DIR / "player_on_off.csv", index=False)
    if tor:
        t = onoff[(onoff["team_id"] == tor[0]) & (onoff["minutes_on"] >= MIN_MINUTES_ON_OFF)]
        print(f"\nToronto - on/off (players with {MIN_MINUTES_ON_OFF}+ minutes):")
        print(t.sort_values("on_off_diff", ascending=False)
              [["player_name", "games", "minutes_on", "net_on", "net_off", "on_off_diff"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
