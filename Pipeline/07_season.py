"""
07_season.py — Toronto Tempo 2026: what went wrong, player by player
====================================================================
Builds the season story from the tables made in 03, 05 and 06:

  - availability of every player, game by game (played / injured / coach's
    decision / not on the roster)
  - games lost to injury: Toronto vs the rest of the league
  - minutes per game for each rotation player, with injury absences
  - on/off: team net rating with each player on vs off the court
  - month by month: results, opponent strength, travel, injuries
    -> which factor changed when the season broke down?

Outputs:
  outputs/figures/06_ ... 10_*.png
  outputs/tables/tor_*.csv, league_injury_games.csv

Run from the project folder (after 03, 05 and 06):
    python3 pipeline/07_season.py
"""

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_FILE = PROJECT_DIR / "data" / "wnba.duckdb"
FIG_DIR = PROJECT_DIR / "outputs" / "figures"
TAB_DIR = PROJECT_DIR / "outputs" / "tables"

FOCUS_TEAM = "TOR"
SEASON = 2026
ROTATION_MIN = 15          # rotation player = 15+ minutes per game when playing
N_PLAYERS_GRID = 14
N_PLAYERS_SMALL_MULTIPLES = 9
MIN_MINUTES_ON_OFF = 200
MIN_MINUTES_DUO = 150       # duos must have shared the court this long
MIN_MINUTES_TRIO = 100
N_COMBOS_SHOWN = 5

# reuse style + expected-performance model from 05_analysis.py
_spec = importlib.util.spec_from_file_location("analysis05", Path(__file__).parent / "05_analysis.py")
a05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a05)
from style import NAVY, INK_2, BOREALIS, BORDEAUX, SOFT, LIGHT, BG, header, source, hgrid, save

# Turning point shown on the availability chart (game number + label)
SPLIT_GAME = 16
SPLIT_LABEL = "Sykes' foot injury"


# =============================================================================
# 1. Data
# =============================================================================

def load_from_db():
    import duckdb
    con = duckdb.connect(str(DB_FILE), read_only=True)
    team, _ = a05.load_from_db()
    player_box = con.execute(f"""
        SELECT game_id, season, season_type, game_date, athlete_id, athlete_display_name, athlete_jersey,
               team_id, team_abbreviation, minutes, starter, did_not_play, reason,
               points, field_goals_made, field_goals_attempted, free_throws_made,
               free_throws_attempted, offensive_rebounds, defensive_rebounds,
               steals, assists, blocks, fouls, turnovers
        FROM raw.player_box
        WHERE season = {SEASON} AND season_type = 2
    """).df()
    onoff = con.execute(f"SELECT * FROM mart.player_on_off WHERE season = {SEASON}").df()
    stints = con.execute(f"SELECT * FROM mart.stints WHERE season = {SEASON}").df()
    con.close()
    return team, player_box, onoff, stints


def player_status(pb: pd.DataFrame) -> pd.Series:
    played = (~pb["did_not_play"].fillna(False).astype(bool)) & (pb["minutes"].fillna(0) > 0)
    coach = pb["reason"].fillna("").str.upper().str.contains("COACH")
    return pd.Series(np.select([played, coach], ["Played", "Coach's decision"], "Injured / ill"),
                     index=pb.index)


# The public feed can lag behind: on 2026-11-05 it still missed Game 7 of the
# Finals (Golden State beat Atlanta 68-63 on Oct 31, 2026 - AJC). If a Finals
# series is not decided in the data, this manual value is used and flagged.
CHAMPION_OVERRIDE = {2026: "GS"}


def playoff_results(team: pd.DataFrame, season: int) -> pd.DataFrame:
    """Furthest playoff round reached by each team, from the playoff games."""
    po = team[(team["season"] == season) & (team["phase"] == "playoffs")]
    rows, flagged = [], False
    for abbr, g in po.groupby("team_abbr"):
        series = g.groupby("opponent_abbr").agg(wins=("win", lambda s: int(s.astype(bool).sum())),
                                                games=("game_id", "count"),
                                                last=("game_date", "max"))
        n_rounds = len(series)
        final = series.sort_values("last").iloc[-1]
        if n_rounds == 1:
            result = "1st round"
        elif n_rounds == 2:
            result = "Semifinal"
        elif final["wins"] == 4:
            result = "Champion"
        elif final["games"] - final["wins"] == 4:
            result = "Finalist"
        else:  # Finals not decided in the data -> manual value, flagged below
            result = "Champion" if abbr == CHAMPION_OVERRIDE.get(season) else "Finalist"
            flagged = True
        rows.append({"team_abbr": abbr, "playoffs": result})
    if flagged:
        print(f"   ! Finals not complete in the data - manual champion used: {CHAMPION_OVERRIDE.get(season)}")
    return pd.DataFrame(rows)


# =============================================================================
# 2. Tables
# =============================================================================

def build_tables(team, pb):
    reg = team[(team["season"] == SEASON) & (team["phase"] == "regular")
               & (team["game_type"] != "CC")].reset_index(drop=True)
    reg["net_vs_expected"] = a05.add_vs_expected(reg, "net_rating")

    pb = pb.copy()
    pb["status"] = player_status(pb)
    played = pb[pb["status"] == "Played"]
    avg_min = played.groupby(["team_id", "athlete_id"])["minutes"].mean().rename("avg_minutes")
    pb = pb.join(avg_min, on=["team_id", "athlete_id"])
    pb["rotation"] = pb["avg_minutes"].fillna(0) >= ROTATION_MIN

    # --- league: rotation games lost to injury ---
    inj = pb[(pb["status"] == "Injured / ill") & pb["rotation"]]
    abbr = reg.drop_duplicates("team_id").set_index("team_id")["team_abbr"]
    league = (inj.groupby("team_id")
              .agg(player_games_lost=("game_id", "count"), minutes_lost=("avg_minutes", "sum"))
              .reindex(abbr.index, fill_value=0).reset_index())
    league["team_abbr"] = league["team_id"].map(abbr)
    league["minutes_lost"] = league["minutes_lost"].round()
    st = a05.standings(reg, SEASON)[["team_abbr", "wins", "losses", "rank"]]
    league = league.merge(st, on="team_abbr", how="left")
    league = league.merge(playoff_results(team, SEASON), on="team_abbr", how="left")
    league["playoffs"] = league["playoffs"].fillna("No playoffs")
    league = league.sort_values("player_games_lost", ascending=False)

    # --- Toronto game list ---
    tor = reg[reg["team_abbr"] == FOCUS_TEAM].sort_values("game_date").reset_index(drop=True)
    tor["game_no"] = tor.index + 1
    tor["month"] = pd.to_datetime(tor["game_date"]).dt.month
    team_strength = reg.groupby("team_abbr")["net_rating"].mean()
    tor["opp_strength"] = tor["opponent_abbr"].map(team_strength)
    inj_count = inj.groupby(["game_id", "team_id"]).size().rename("rotation_injured")
    tor = tor.join(inj_count, on=["game_id", "team_id"])
    tor["rotation_injured"] = tor["rotation_injured"].fillna(0).astype(int)

    tor_id = tor["team_id"].iloc[0]
    tpb = pb[pb["team_id"] == tor_id].merge(tor[["game_id", "game_no"]], on="game_id")
    tpb["game_score"] = (tpb["points"] + 0.4 * tpb["field_goals_made"] - 0.7 * tpb["field_goals_attempted"]
                         - 0.4 * (tpb["free_throws_attempted"] - tpb["free_throws_made"])
                         + 0.7 * tpb["offensive_rebounds"] + 0.3 * tpb["defensive_rebounds"]
                         + tpb["steals"] + 0.7 * tpb["assists"] + 0.7 * tpb["blocks"]
                         - 0.4 * tpb["fouls"] - tpb["turnovers"])

    # --- monthly summary ---
    month_names = {5: "May", 6: "June", 7: "July", 8: "August", 9: "September"}
    monthly = (tor.groupby("month")
               .agg(games=("game_id", "count"),
                    wins=("win", lambda s: int(s.astype(bool).sum())),
                    net_rating=("net_rating", "mean"),
                    opponent_strength=("opp_strength", "mean"),
                    home_share=("is_home", "mean"),
                    travel_km_per_game=("travel_km", "mean"),
                    back_to_backs=("back_to_back", lambda s: int(s.fillna(False).astype(bool).sum())),
                    rotation_injured_per_game=("rotation_injured", "mean"))
               .round(2).reset_index())
    monthly["month"] = monthly["month"].map(month_names)

    # --- with / without each key player (injury absences only) ---
    rows = []
    key = (tpb[tpb["status"] == "Played"].groupby("athlete_display_name")["minutes"].sum()
           .sort_values(ascending=False).head(N_PLAYERS_SMALL_MULTIPLES).index)
    for name in key:
        p = tpb[tpb["athlete_display_name"] == name]
        g_with = p.loc[p["status"] == "Played", "game_id"]
        g_inj = p.loc[p["status"] == "Injured / ill", "game_id"]
        w, wo = tor[tor["game_id"].isin(g_with)], tor[tor["game_id"].isin(g_inj)]
        rows.append({"player": name, "games_played": len(w), "games_missed_injured": len(wo),
                     "record_with": f"{int(w['win'].sum())}-{len(w) - int(w['win'].sum())}",
                     "record_without": f"{int(wo['win'].sum())}-{len(wo) - int(wo['win'].sum())}" if len(wo) else "",
                     "net_with": round(w["net_rating"].mean(), 1),
                     "net_without": round(wo["net_rating"].mean(), 1) if len(wo) else np.nan})
    with_without = pd.DataFrame(rows)

    # --- player x month: minutes and game score per 36 ---
    pm = tpb[tpb["status"] == "Played"].merge(tor[["game_id", "month"]], on="game_id")
    player_month = (pm.groupby(["athlete_display_name", "month"])
                    .apply(lambda d: pd.Series({
                        "games": len(d), "minutes_avg": d["minutes"].mean(),
                        "game_score_per36": 36 * d["game_score"].sum() / d["minutes"].sum()}),
                           include_groups=False)
                    .round(1).reset_index())

    return tor, tpb, league, monthly, with_without, player_month


# =============================================================================
# 3. Figures
# =============================================================================

def _top_players(tpb, n):
    return list(tpb[tpb["status"] == "Played"].groupby("athlete_display_name")["minutes"].sum()
                .sort_values(ascending=False).head(n).index)


def fig_availability(tor, tpb, path):
    order = _top_players(tpb, N_PLAYERS_GRID)
    fig = plt.figure(figsize=(13, 10))
    header(fig, "The season broke when the roster did",
           f"Toronto Tempo {SEASON} · 5-game rolling net rating (top) and who was available, game by game (bottom)")
    ax = fig.add_axes([0.17, 0.60, 0.79, 0.25])
    ax2 = fig.add_axes([0.17, 0.11, 0.79, 0.44], sharex=ax)

    # --- top: rolling net rating ---
    roll = tor["net_rating"].rolling(5, min_periods=3).mean()
    ax.axhline(0, color=SOFT, lw=1.2)
    ax.fill_between(tor["game_no"], roll, 0, where=roll < 0, color=BORDEAUX, alpha=0.12, interpolate=True)
    ax.fill_between(tor["game_no"], roll, 0, where=roll >= 0, color=BOREALIS, alpha=0.15, interpolate=True)
    ax.plot(tor["game_no"], roll, color=NAVY, lw=2.5)
    win = tor["win"].astype(bool).values
    ax.scatter(tor["game_no"][win], np.full(win.sum(), 24), s=30, color=BOREALIS, zorder=3)
    ax.scatter(tor["game_no"][~win], np.full((~win).sum(), 24), s=30, facecolor=BG,
               edgecolor=SOFT, lw=1.2, zorder=3)
    ax.text(0.1, 24, "W / L  ", ha="right", va="center", fontsize=8, color=INK_2)
    ax.set_ylim(-34, 29)
    ax.set_yticks([-30, -15, 0, 15], ["-30", "-15", "0", "+15"])
    hgrid(ax, [-30, -15, 15])
    ax.set_ylabel("Net rating\n(5-game avg)", fontsize=9)
    ax.tick_params(labelbottom=False)

    # annotate the turning point: first long absence of a top-3 player by minutes, if any
    first_half = tor[tor["game_no"] <= SPLIT_GAME]
    second_half = tor[tor["game_no"] > SPLIT_GAME]
    rec = lambda d: f"{int(d['win'].sum())}-{len(d) - int(d['win'].sum())}"
    ax.axvline(SPLIT_GAME + 0.5, color=BORDEAUX, lw=1, ls=(0, (3, 3)))
    ax.text(SPLIT_GAME / 2 + 0.5, 15, f"{rec(first_half)} in the first {SPLIT_GAME} games",
            ha="center", fontsize=10, fontweight="bold", color=NAVY)
    ax.text((SPLIT_GAME + len(tor)) / 2 + 0.5, -31, f"{rec(second_half)} after {SPLIT_LABEL}",
            ha="center", fontsize=10, fontweight="bold", color=BORDEAUX)

    # --- bottom: availability dots ---
    for i, name in enumerate(order):
        p = tpb[tpb["athlete_display_name"] == name]
        ax2.axhline(i, color=LIGHT, lw=7, zorder=0, solid_capstyle="round")
        for status, kw in [("Played", dict(s=40, color=BOREALIS)),
                           ("Injured / ill", dict(s=44, marker="X", color=BORDEAUX)),
                           ("Coach's decision", dict(s=20, color=SOFT))]:
            q = p[p["status"] == status]
            ax2.scatter(q["game_no"], np.full(len(q), i), zorder=3, **kw)
    ax2.set_yticks(range(len(order)), order, fontsize=10, color=NAVY)
    ax2.set_ylim(len(order) - 0.5, -0.7)
    ax2.set_xlim(0.3, len(tor) + 0.7)
    months = tor.groupby("month")["game_no"].min()
    ax2.set_xticks(months.values, [pd.Timestamp(2026, m, 1).strftime("%b") for m in months.index])
    for g in months.values[1:]:
        ax.axvline(g - 0.5, color=LIGHT, lw=1, zorder=0)
        ax2.axvline(g - 0.5, color=LIGHT, lw=1, zorder=0)
    ax2.scatter([], [], s=40, color=BOREALIS, label="Played")
    ax2.scatter([], [], s=44, marker="X", color=BORDEAUX, label="Out (injury / illness)")
    ax2.scatter([], [], s=20, color=SOFT, label="Coach's decision")
    ax2.legend(loc="upper left", bbox_to_anchor=(0, -0.06), ncol=3, fontsize=9)
    source(fig, f"Top {len(order)} players by minutes. Empty = not on the roster (signing, trade, national team). "
                "Source: ESPN box scores via SportsDataverse.")
    save(fig, path)


def fig_league_injuries(league, path):
    d = league.sort_values("player_games_lost").reset_index(drop=True)
    fig = plt.figure(figsize=(11, 8))
    header(fig, "No team lost more to injury",
           f"Games missed through injury or illness by rotation players (15+ min/game), {SEASON} "
           "· with each team's record and playoff result")
    ax = fig.add_axes([0.25, 0.08, 0.66, 0.74])
    is_focus = d["team_abbr"] == FOCUS_TEAM
    ax.barh(range(len(d)), d["player_games_lost"],
            color=np.where(is_focus, BORDEAUX, SOFT), height=0.62)
    for y, r in d.iterrows():
        focus = r["team_abbr"] == FOCUS_TEAM
        extra = f"   ≈ {r['minutes_lost']:,.0f} min lost" if focus else ""
        ax.text(r["player_games_lost"] + 1.5, y, f"{r['player_games_lost']}{extra}", va="center",
                fontsize=10, color=BORDEAUX if focus else INK_2,
                fontweight="bold" if focus else "normal")
    labels = [f"{t}   {int(w)}-{int(l)} · {po}" for t, w, l, po in
              zip(d["team_abbr"], d["wins"], d["losses"], d["playoffs"])]
    ax.set_yticks(range(len(d)), labels, fontsize=10)
    for lbl, f, po in zip(ax.get_yticklabels(), is_focus, d["playoffs"]):
        lbl.set_color(BORDEAUX if f else BOREALIS if po == "Champion" else
                      NAVY if po != "No playoffs" else INK_2)
        lbl.set_fontweight("bold" if (f or po == "Champion") else "normal")
    fig.text(0.24, 0.835, "team · record · playoffs", fontsize=8.5, color=INK_2, ha="right")
    champ = d[d["playoffs"] == "Champion"]
    if len(champ):
        y = champ.index[0]
        ax.text(champ["player_games_lost"].iloc[0] + 9, y, "<- the champion lost the fewest games to injury",
                fontsize=9.5, color=BOREALIS, va="center", fontweight="bold")
    med = d["player_games_lost"].median()
    ax.vlines(med, 0.5, len(d) - 0.5, color=INK_2, lw=1, ls=(0, (3, 3)))
    ax.text(med, len(d) - 0.4, f"league median: {med:.0f}", fontsize=9, color=INK_2,
            va="bottom", ha="center")
    ax.set_xlim(0, d["player_games_lost"].max() * 1.45)
    ax.set_ylim(-0.6, len(d) + 0.2)
    ax.set_xticks([])
    source(fig, "ESPN absence reasons; coach's decisions & national-team absences excluded. "
                "Finals Game 7 added manually.")
    save(fig, path)


def fig_player_minutes(tor, tpb, path):
    names = _top_players(tpb, N_PLAYERS_SMALL_MULTIPLES)
    n = len(names)
    fig = plt.figure(figsize=(13, 2 + 1.35 * n))
    top, bottom = 1 - 1.55 / fig.get_figheight(), 0.75 / fig.get_figheight()
    header(fig, "Player by player: who carried the minutes, who was missing",
           f"Toronto Tempo {SEASON} · minutes per game (line), season average (dashed) "
           "and games missed through injury or illness (✕)")
    row_h = (top - bottom) / n
    months = tor.groupby("month")["game_no"].agg(["min", "max"])
    month_lbl = {m: pd.Timestamp(2026, m, 1).strftime("%B") for m in months.index}
    for i, name in enumerate(names):
        ax = fig.add_axes([0.29, top - (i + 1) * row_h + row_h * 0.12, 0.68, row_h * 0.78])
        p = tpb[tpb["athlete_display_name"] == name].set_index("game_no")
        line = p["minutes"].where(p["status"] == "Played").reindex(range(1, len(tor) + 1))
        inj = p[p["status"] == "Injured / ill"]
        played = p[p["status"] == "Played"]
        # alternate month bands so dates are easy to follow
        for k, (m, r) in enumerate(months.iterrows()):
            if k % 2 == 0:
                ax.axvspan(r["min"] - 0.5, r["max"] + 0.5, color=LIGHT, zorder=0, lw=0)
        avg = played["minutes"].mean()
        ax.axhline(avg, color=INK_2, lw=1, ls=(0, (2, 3)), zorder=1)
        ax.plot(line.index, line.values, color=BOREALIS, lw=2.2, marker="o", ms=3.5, zorder=3)
        ax.scatter(inj.index, np.full(len(inj), 4), marker="X", s=40, color=BORDEAUX, zorder=3)
        ax.set_ylim(0, 44)
        ax.set_xlim(0.5, len(tor) + 0.5)
        ax.set_yticks([0, 20, 40])
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xticks([])
        if i == 0 or i == n - 1:  # month names above the first row and below the last
            for m, r in months.iterrows():
                ax.text((r["min"] + r["max"]) / 2, 47 if i == 0 else -9, month_lbl[m],
                        ha="center", va="bottom" if i == 0 else "top", fontsize=10, color=INK_2)
        # left label: jersey badge + name + key numbers
        y_mid = top - (i + 0.5) * row_h
        jersey = p["athlete_jersey"].dropna()
        num = f"{int(float(jersey.iloc[0]))}" if len(jersey) else ""
        fig.text(0.055, y_mid, num, ha="center", va="center", fontsize=11, fontweight="bold",
                 color=BG, bbox=dict(boxstyle="circle,pad=0.45", fc=NAVY, ec="none"))
        fig.text(0.085, y_mid + row_h * 0.12, name, fontsize=11.5, fontweight="bold", color=NAVY, va="center")
        missed = f" · {len(inj)} missed" if len(inj) else " · never missed"
        fig.text(0.085, y_mid - row_h * 0.18, f"{len(played)} games · {avg:.0f} min avg{missed}",
                 fontsize=9, color=BORDEAUX if len(inj) >= 10 else INK_2, va="center")
    source(fig, "Shaded bands = alternate months. Number = jersey. Source: ESPN box scores via SportsDataverse.")
    save(fig, path)


def fig_on_off(onoff, tor_id, path):
    d = onoff[(onoff["team_id"] == tor_id) & (onoff["minutes_on"] >= MIN_MINUTES_ON_OFF)]
    d = d.sort_values("on_off_diff").reset_index(drop=True)
    fig = plt.figure(figsize=(11, 7.5))
    header(fig, "Who made the team better on the court?",
           f"Toronto Tempo {SEASON} · team net rating with the player on court minus off court "
           f"(per 100 possessions, {MIN_MINUTES_ON_OFF}+ min)")
    ax = fig.add_axes([0.19, 0.09, 0.62, 0.74])
    colors = [BOREALIS if v > 0 else BORDEAUX for v in d["on_off_diff"]]
    ax.barh(d["player_name"], d["on_off_diff"], color=colors, height=0.62)
    ax.axvline(0, color=NAVY, lw=1)
    lim = d["on_off_diff"].abs().max() * 1.35
    ax.set_xlim(-lim, lim)
    for y, v in enumerate(d["on_off_diff"]):
        ax.text(v + (0.4 if v >= 0 else -0.4), y, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=10, fontweight="bold",
                color=BOREALIS if v >= 0 else BORDEAUX)
    # side table: on / off / minutes
    for y, (on, off, m) in enumerate(zip(d["net_on"], d["net_off"], d["minutes_on"])):
        fig.text(0.83, ax.transData.transform((0, y))[1] / fig.dpi / fig.get_figheight(),
                 f"{on:+.1f}   {off:+.1f}   {m:,.0f}", fontsize=9, color=INK_2, va="center")
    fig.text(0.83, 0.85, "on      off     min", fontsize=8.5, color=INK_2, fontweight="bold")
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=10, colors=NAVY)
    source(fig, "Rebuilt from play-by-play; 'off' = only games she played. One season is noisy: read as a direction.")
    save(fig, path)


def fig_monthly(monthly, path):
    panels = [("net_rating", "Net rating", "{:+.1f}"),
              ("rotation_injured_per_game", "Rotation players out injured, per game", "{:.1f}"),
              ("opponent_strength", "Opponent strength (their avg net rating)", "{:+.1f}"),
              ("travel_km_per_game", "Travel per game (km)", "{:,.0f}")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.80, bottom=0.12, hspace=0.6, wspace=0.15)
    header(fig, "What changed when the season broke down?",
           "Toronto Tempo 2026, month by month · injuries rose, travel did not explain it")
    x = np.arange(len(monthly))
    broke = monthly["net_rating"] < -10
    for ax, (col, title, fmt) in zip(axes.flat, panels):
        vals = monthly[col]
        ax.bar(x, vals, color=[BORDEAUX if b else BOREALIS for b in broke], width=0.6)
        ax.axhline(0, color=SOFT, lw=1.2)
        for xi, v in zip(x, vals):
            ax.annotate(fmt.format(v), (xi, v), xytext=(0, 5 if v >= 0 else -14),
                        textcoords="offset points", ha="center", fontsize=10, color=NAVY,
                        fontweight="bold")
        ax.set_xticks(x, [f"{m[:3]}\n{w}-{g - w}" for m, w, g in
                          zip(monthly["month"], monthly["wins"], monthly["games"])])
        ax.set_title(title)
        ax.set_yticks([])
        lo, hi = min(0, vals.min()), max(0, vals.max())
        pad = (hi - lo) * 0.2 or 1
        ax.set_ylim(lo - pad, hi + pad)
    fig.text(0.04, 0.86, "■ months when the team collapsed (net rating below -10)",
             color=BORDEAUX, fontsize=9)
    source(fig, "Labels under bars = record. Source: ESPN via SportsDataverse.")
    save(fig, path)


def combos(stints, team_id, names, k, min_minutes):
    """Net rating for every group of k players who shared the court (duos, trios)."""
    import itertools
    t = stints[stints["team_id"] == team_id]
    rows = []
    for r in t.itertuples(index=False):
        ids = sorted(int(x) for x in str(r.lineup).split("-") if x)
        for c in itertools.combinations(ids, k):
            rows.append((c, r.seconds, r.pts_for, r.pts_against, r.poss_for, r.poss_against))
    d = (pd.DataFrame(rows, columns=["combo", "seconds", "pts_for", "pts_against", "poss_for", "poss_against"])
         .groupby("combo").sum())
    d["minutes"] = d["seconds"] / 60
    d = d[d["minutes"] >= min_minutes]
    d["net_rating"] = 100 * (d["pts_for"] - d["pts_against"]) / ((d["poss_for"] + d["poss_against"]) / 2)
    d["players"] = [" + ".join(names.get(i, str(i)).split()[-1] for i in c) for c in d.index]
    return (d.reset_index(drop=True)[["players", "minutes", "net_rating"]]
            .sort_values("net_rating", ascending=False).round(1).reset_index(drop=True))


def fig_combos(duos, trios, path, team_avg=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.80, bottom=0.08, wspace=0.75)
    header(fig, "Some combinations worked, others never did",
           f"Toronto Tempo {SEASON} · team net rating when these players shared the court "
           f"(best {N_COMBOS_SHOWN} and worst {N_COMBOS_SHOWN})")
    for ax, (d, title, mm) in zip(axes, [(duos, "Duos", MIN_MINUTES_DUO), (trios, "Trios", MIN_MINUTES_TRIO)]):
        sel = pd.concat([d.head(N_COMBOS_SHOWN), d.tail(N_COMBOS_SHOWN)]).iloc[::-1].reset_index(drop=True)
        y = np.arange(len(sel)) + np.where(np.arange(len(sel)) >= N_COMBOS_SHOWN, 0.8, 0)
        colors = [BOREALIS if v >= 0 else BORDEAUX for v in sel["net_rating"]]
        ax.barh(y, sel["net_rating"], color=colors, height=0.7)
        ax.axvline(0, color=NAVY, lw=1)
        for yy, v, m in zip(y, sel["net_rating"], sel["minutes"]):
            ax.text(v + (1 if v >= 0 else -1), yy, f"{v:+.1f}  ({m:.0f} min)" if v >= 0 else f"({m:.0f} min)  {v:+.1f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=9,
                    color=BOREALIS if v >= 0 else BORDEAUX, fontweight="bold")
        ax.set_yticks(y, sel["players"], fontsize=9.5, color=NAVY)
        lim = sel["net_rating"].abs().max() * 2.5
        if team_avg is not None:
            ax.axvline(team_avg, color=INK_2, lw=1, ls=(0, (3, 3)))
            ax.text(team_avg, y.max() + 0.9, f"team avg {team_avg:+.1f}", ha="center",
                    fontsize=8.5, color=INK_2)
        ax.set_xlim(-lim, lim)
        ax.set_xticks([])
        ax.set_title(f"{title} ({mm}+ min together)", pad=26)
    source(fig, "Rebuilt from play-by-play. Small samples: a direction for lineup decisions, not a verdict.")
    save(fig, path)


# =============================================================================
# 4. Main
# =============================================================================

def run(team, player_box, onoff, stints=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    tor, tpb, league, monthly, with_without, player_month = build_tables(team, player_box)
    tor_id = tor["team_id"].iloc[0]

    fig_availability(tor, tpb, FIG_DIR / f"06_{FOCUS_TEAM}_availability.png")
    fig_league_injuries(league, FIG_DIR / "07_league_injury_games.png")
    fig_player_minutes(tor, tpb, FIG_DIR / f"08_{FOCUS_TEAM}_player_minutes.png")
    fig_on_off(onoff, tor_id, FIG_DIR / f"09_{FOCUS_TEAM}_on_off.png")
    fig_monthly(monthly, FIG_DIR / f"10_{FOCUS_TEAM}_monthly.png")

    if stints is not None:
        names = player_box.drop_duplicates("athlete_id").set_index("athlete_id")["athlete_display_name"]
        duos = combos(stints, tor_id, names, 2, MIN_MINUTES_DUO)
        trios = combos(stints, tor_id, names, 3, MIN_MINUTES_TRIO)
        duos.to_csv(TAB_DIR / f"{FOCUS_TEAM.lower()}_duos.csv", index=False)
        trios.to_csv(TAB_DIR / f"{FOCUS_TEAM.lower()}_trios.csv", index=False)
        fig_combos(duos, trios, FIG_DIR / f"11_{FOCUS_TEAM}_combinations.png",
                   team_avg=round(tor["net_rating"].mean(), 1))
        print(f"\n{FOCUS_TEAM} duos (best and worst):")
        print(pd.concat([duos.head(N_COMBOS_SHOWN), duos.tail(N_COMBOS_SHOWN)]).to_string(index=False))
        print(f"\n{FOCUS_TEAM} trios (best and worst):")
        print(pd.concat([trios.head(N_COMBOS_SHOWN), trios.tail(N_COMBOS_SHOWN)]).to_string(index=False))

    league.to_csv(TAB_DIR / "league_injury_games.csv", index=False)
    monthly.to_csv(TAB_DIR / f"{FOCUS_TEAM.lower()}_monthly.csv", index=False)
    with_without.to_csv(TAB_DIR / f"{FOCUS_TEAM.lower()}_with_without.csv", index=False)
    player_month.to_csv(TAB_DIR / f"{FOCUS_TEAM.lower()}_player_month.csv", index=False)
    tpb[["game_no", "game_date", "athlete_display_name", "status", "reason", "minutes"]] \
        .to_csv(TAB_DIR / f"{FOCUS_TEAM.lower()}_availability.csv", index=False)

    print("Rotation games lost to injury / illness, by team:")
    print(league[["team_abbr", "player_games_lost", "minutes_lost"]].to_string(index=False))
    print(f"\n{FOCUS_TEAM} month by month:")
    print(monthly.to_string(index=False))
    print(f"\n{FOCUS_TEAM} with / without (injury absences only):")
    print(with_without.to_string(index=False))
    print(f"\nFigures: {FIG_DIR.relative_to(PROJECT_DIR)}   Tables: {TAB_DIR.relative_to(PROJECT_DIR)}")


def main() -> None:
    run(*load_from_db())


if __name__ == "__main__":
    main()
