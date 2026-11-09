"""
05_analysis.py — Does schedule load affect performance?
=======================================================
Uses the mart tables built in 03 to answer three questions:

  1. League-wide (2024-2026 regular seasons): do back-to-backs, long trips,
     time zone changes and border crossings lower team performance?
  2. How does the Toronto Tempo's 2026 schedule compare with the league?
  3. Which Tempo games were the most demanding (schedule load index)?

Performance is measured against expectation: a team's net rating in a game
is compared with what we would expect given its own level, the opponent's
level and home court. A negative "vs expected" value means the team played
below its usual level in that situation.

Outputs:
  outputs/figures/*.png          charts for the report
  outputs/tables/*.csv           numbers behind the charts

Run from the project folder (after 03):
    python3 pipeline/05_analysis.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write image files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_FILE = PROJECT_DIR / "data" / "wnba.duckdb"
FIG_DIR = PROJECT_DIR / "outputs" / "figures"
TAB_DIR = PROJECT_DIR / "outputs" / "tables"

FOCUS_TEAM = "TOR"
FOCUS_SEASON = 2026

# --- Shared style (colors, fonts, title layout) ---
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (NAVY, INK_2, BOREALIS, BORDEAUX, SOFT, LIGHT, BG, LOAD_COLORS,
                   header, source, hgrid, save)
from matplotlib.patches import Patch

# names kept for scripts that import them from this file
INK, MUTED, GRID, SURFACE, ACCENT, NEUTRAL = NAVY, INK_2, LIGHT, BG, BOREALIS, SOFT
STATUS = LOAD_COLORS


# =============================================================================
# 1. Load data
# =============================================================================

def load_from_db():
    import duckdb
    con = duckdb.connect(str(DB_FILE), read_only=True)
    team = con.execute("""
        SELECT t.*, v.lat, v.lon, v.tz_offset_et AS venue_tz
        FROM mart.team_game_load t
        LEFT JOIN ref.venue_cities v ON v.city = t.venue_city
    """).df()
    games = con.execute("SELECT game_id, game_type FROM clean.games").df()
    player = con.execute("SELECT * FROM mart.player_game_load").df()
    con.close()
    team = team.merge(games, on="game_id", how="left")
    player = player.merge(games, on="game_id", how="left")
    # Missing values from the database (first game of a season has no
    # "previous game") -> use plain floats / objects so comparisons don't fail
    for df in (team, player):
        for col in df.columns:
            if str(df[col].dtype) in ("Int8", "Int16", "Int32", "Int64", "Float64"):
                df[col] = df[col].astype(float)
            elif str(df[col].dtype) == "boolean":
                df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    return team, player


# =============================================================================
# 2. Expected performance model
# =============================================================================

def add_vs_expected(df: pd.DataFrame, metric: str) -> pd.Series:
    """
    metric = team level + opponent level + home court  (least squares)
    Team and opponent levels are estimated per season, so a team's level
    in 2024 does not leak into 2026. Returns actual - expected.
    """
    team_key = df["team_abbr"] + "_" + df["season"].astype(str)
    opp_key = df["opponent_abbr"] + "_" + df["season"].astype(str)
    team_dum = pd.get_dummies(team_key, prefix="t", dtype=float)
    opp_dum = pd.get_dummies(opp_key, prefix="o", dtype=float)
    X = pd.concat([team_dum, opp_dum], axis=1)
    X["home"] = df["is_home"].astype(float).values
    y = df[metric].astype(float).values
    ok = ~np.isnan(y)
    beta, *_ = np.linalg.lstsq(X.values[ok], y[ok], rcond=None)
    expected = X.values @ beta
    return pd.Series(y - expected, index=df.index)


# =============================================================================
# 3. Schedule load index (the decision rule)
# =============================================================================

def load_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple, explainable points system - one point per stressor:
      back-to-back                     +2
      only 1 day of rest               +1
      trip of 2,000 km or more         +1
      time zone change of 2h or more   +1
      eastward time zone change        +1  (harder to adapt to than westward)
      border crossing                  +1
      4+ games in the last 7 days      +1
    Travel stressors are ignored after 4+ days of rest (reset).
    Level:  0-1 Standard   2-3 Adjusted   4+ Priority
    """
    # After 4+ days off, the team has had time to go home and adapt:
    # travel, time zone and border stressors no longer count.
    travel_counts = (df["days_rest"].fillna(99) < 4)
    pts_travel = (
        (df["travel_km"].fillna(0) >= 2000).astype(int)
        + (df["tz_shift_h"].abs().fillna(0) >= 2).astype(int)
        + (df["tz_shift_h"].fillna(0) > 0).astype(int)
        + (df["crossed_border"].fillna(False).astype(bool)).astype(int)
    ) * travel_counts.astype(int)
    pts = (
        2 * (df["days_rest"] == 0).astype(int)
        + (df["days_rest"] == 1).astype(int)
        + pts_travel
        + (df["games_last_7d"].fillna(0) >= 4).astype(int)
    )
    level = pd.cut(pts, [-1, 1, 3, 99], labels=["Standard", "Adjusted", "Priority"])
    return pd.DataFrame({"load_points": pts, "load_level": level.astype(str)}, index=df.index)


# =============================================================================
# 4. Group comparisons
# =============================================================================

def categorize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    recent = df["days_rest"] < 4   # travel factors only matter if the trip is recent
    out["Rest days"] = pd.cut(df["days_rest"], [-1, 0, 1, 2, 99],
                              labels=["0 (B2B)", "1", "2", "3+"]).astype(str)
    out["Trip since last game"] = pd.cut(df["travel_km"], [-1, 0, 999, 2499, 99999],
                                         labels=["No trip", "< 1,000 km",
                                                 "1,000-2,500 km", "2,500+ km"]).astype(str).where(recent)
    tz = df["tz_shift_h"]
    out["Time zone change"] = np.select([tz < 0, tz == 0, tz > 0],
                                        ["Westward", "None", "Eastward"], default="nan")
    out["Time zone change"] = out["Time zone change"].where(recent)
    ra = df["rest_advantage"].clip(-2, 2)
    out["Rest vs opponent"] = np.select([ra < 0, ra == 0, ra > 0],
                                        ["Less rest", "Same", "More rest"], default="nan")
    out["Border crossing"] = np.where(df["crossed_border"].isna(), "nan",
                                      np.where(df["crossed_border"].fillna(False).astype(bool),
                                               "Crossed", "No"))
    out["Border crossing"] = out["Border crossing"].where(recent)
    out["Load level"] = df["load_level"]
    return out.replace("nan", np.nan)


ORDER = {
    "Rest days": ["0 (B2B)", "1", "2", "3+"],
    "Trip since last game": ["No trip", "< 1,000 km", "1,000-2,500 km", "2,500+ km"],
    "Time zone change": ["Westward", "None", "Eastward"],
    "Rest vs opponent": ["Less rest", "Same", "More rest"],
    "Border crossing": ["No", "Crossed"],
    "Load level": ["Standard", "Adjusted", "Priority"],
}


def group_effects(df: pd.DataFrame, cats: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for factor, levels in ORDER.items():
        for lvl in levels:
            vals = df.loc[cats[factor] == lvl, metric].dropna()
            n = len(vals)
            mean = vals.mean() if n else np.nan
            se = vals.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            rows.append({"factor": factor, "level": lvl, "n_games": n,
                         "mean": round(mean, 2),
                         "ci95_low": round(mean - 1.96 * se, 2),
                         "ci95_high": round(mean + 1.96 * se, 2)})
    return pd.DataFrame(rows)


# =============================================================================
# 5. Figures
# =============================================================================

def month_ticks(ax, dates, positions):
    """Put one tick per month at the first game of that month."""
    d = pd.Series(pd.to_datetime(dates).values, index=positions)
    first = d.groupby(d.dt.month).apply(lambda s: s.index.min())
    ax.set_xticks(first.values, [pd.Timestamp(2026, m, 1).strftime("%b") for m in first.index])
    for x in first.values[1:]:
        ax.axvline(x - 0.5, color=LIGHT, lw=1, zorder=0)


def fig_league_effects(eff: pd.DataFrame, path: Path, metric_label: str,
                       title: str = "The schedule matters, but only a little"):
    factors = ["Rest days", "Trip since last game", "Time zone change", "Rest vs opponent"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.10, hspace=0.55, wspace=0.12)
    header(fig, title,
           f"WNBA regular seasons 2024-2026 · {metric_label} vs expected, by situation "
           "(team level, opponent and home court accounted for)")
    lo, hi = eff["ci95_low"].min(), eff["ci95_high"].max()
    for ax, factor in zip(axes.flat, factors):
        d = eff[eff["factor"] == factor].reset_index(drop=True)
        x = np.arange(len(d))
        ax.axhline(0, color=SOFT, lw=1.2)
        ax.vlines(x, d["ci95_low"], d["ci95_high"], color=SOFT, lw=5, capstyle="round")
        colors = [BOREALIS if m >= 0 else BORDEAUX for m in d["mean"]]
        ax.scatter(x, d["mean"], s=110, color=colors, zorder=3, edgecolor=BG, lw=2)
        for xi, m in zip(x, d["mean"]):
            ax.annotate(f"{m:+.1f}", (xi, m), xytext=(12, 0), textcoords="offset points",
                        va="center", fontsize=10, fontweight="bold",
                        color=BOREALIS if m >= 0 else BORDEAUX)
        ax.set_xticks(x, [f"{l}\n{n} team-games" for l, n in zip(d["level"], d["n_games"])], fontsize=9)
        ax.set_title(factor)
        ax.set_xlim(-0.5, len(d) - 0.4)
        ax.set_ylim(lo - 0.5, hi + 0.5)
        hgrid(ax, [-4, -2, 2])
    source(fig, "Dot = average · bar = 95% confidence interval. Source: ESPN via SportsDataverse.")
    save(fig, path)


def fig_season_load(season_tab: pd.DataFrame, path: Path):
    d = season_tab.sort_values("border_crossings")
    colors = [BORDEAUX if t == FOCUS_TEAM else SOFT for t in d["team_abbr"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.08, wspace=0.15)
    header(fig, "Toronto doesn't travel the most - it crosses the border the most",
           f"{FOCUS_SEASON} regular season, by team")
    for ax, (col, title, scale, fmt) in zip(axes, [
            ("border_crossings", "Border crossings", 1, "{:.0f}"),
            ("total_travel_km", "Distance travelled (1,000 km)", 1000, "{:.0f}")]):
        vals = d[col] / scale
        ax.barh(d["team_abbr"], vals, color=colors, height=0.65)
        for y, (t, v) in enumerate(zip(d["team_abbr"], vals)):
            ax.text(v + vals.max() * 0.015, y, fmt.format(v), va="center", fontsize=9,
                    color=BORDEAUX if t == FOCUS_TEAM else INK_2,
                    fontweight="bold" if t == FOCUS_TEAM else "normal")
        ax.set_title(title)
        ax.set_xticks([])
        ax.tick_params(axis="y", colors=NAVY)
    source(fig, "Distances = straight line between arenas. Source: ESPN via SportsDataverse.")
    save(fig, path)


def fig_focus_timeline(d: pd.DataFrame, path: Path):
    d = d.sort_values("game_date").reset_index(drop=True)
    d["n"] = d.index + 1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.6]})
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.10, hspace=0.35)
    header(fig, "Heavy schedule days were not the main story",
           f"Toronto Tempo {FOCUS_SEASON} · schedule load per game (top) and net rating (bottom)")
    ax1.bar(d["n"], d["load_points"], color=[LOAD_COLORS[l] for l in d["load_level"]], width=0.7)
    ax1.set_title("Schedule load index")
    hgrid(ax1, [2, 4])
    ax1.set_yticks([0, 2, 4])
    ax1.legend(handles=[Patch(color=c, label=l) for l, c in LOAD_COLORS.items()],
               loc="upper right", ncol=3, fontsize=9, bbox_to_anchor=(1, 1.35))
    ax2.axhline(0, color=SOFT, lw=1.2)
    ax2.bar(d["n"], d["net_rating"],
            color=[BOREALIS if w else SOFT for w in d["win"].astype(bool)], width=0.7)
    ax2.set_title("Net rating per game (blue = win)")
    hgrid(ax2, [-30, -15, 15, 30])
    for ax in (ax1, ax2):
        month_ticks(ax, d["game_date"], d["n"])
    source(fig, "Load points: back-to-back +2, 1 day rest +1, trip 2,000+ km +1, time change 2h+ +1, "
                "eastward +1, border +1, 4+ games in 7 days +1.")
    save(fig, path)


def fig_focus_map(d: pd.DataFrame, path: Path):
    d = d.sort_values("game_date").dropna(subset=["lat", "lon"])
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_axes([0.04, 0.07, 0.92, 0.75])
    header(fig, f"Toronto's {FOCUS_SEASON} road map",
           f"Every trip of the regular season · dot size = games played there · "
           f"{int(d['crossed_border'].fillna(False).astype(bool).sum())} border crossings")
    ax.plot(d["lon"], d["lat"], color=SOFT, lw=1, alpha=0.9, zorder=1)
    cities = d.groupby("venue_city").agg(lat=("lat", "first"), lon=("lon", "first"),
                                         games=("game_id", "count")).reset_index()
    canada = cities["venue_city"].isin(["Toronto", "Vancouver", "Montreal"])
    ax.scatter(cities["lon"], cities["lat"], s=40 + cities["games"] * 18,
               color=np.where(canada, BORDEAUX, BOREALIS), edgecolor=BG, linewidth=2, zorder=2)
    for _, c in cities.iterrows():
        dy = -13 if c["venue_city"] == "College Park" else 7
        ax.annotate(c["venue_city"], (c["lon"], c["lat"]), xytext=(7, dy),
                    textcoords="offset points", fontsize=9, color=NAVY)
    ax.scatter([], [], s=80, color=BORDEAUX, label="Canada")
    ax.scatter([], [], s=80, color=BOREALIS, label="United States")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_aspect(1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    source(fig, "Straight lines between consecutive game venues. Source: ESPN via SportsDataverse.")
    save(fig, path)


def timezone_effects(reg: pd.DataFrame) -> pd.DataFrame:
    """Performance vs expected by time zone change on arrival (only recent trips)."""
    d = reg[(reg["days_rest"] < 4) & reg["tz_shift_h"].notna()]
    out = (d.groupby("tz_shift_h")["net_vs_expected"]
           .agg(mean="mean", se="sem", n_games="count").reset_index())
    out["ci95_low"] = out["mean"] - 1.96 * out["se"]
    out["ci95_high"] = out["mean"] + 1.96 * out["se"]
    return out.round(2)


def _plane(ax, x, y, color, facing_left=False, size=0.42):
    """Small plane icon (mirrored, not upside down, when flying west)."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D
    path = TextPath((0, 0), "\u2708", size=1, prop=FontProperties(family="DejaVu Sans"))
    ext = path.get_extents()
    sx = -1 if facing_left else 1
    tr = (Affine2D().translate(-(ext.x0 + ext.x1) / 2, -(ext.y0 + ext.y1) / 2)
          .scale(sx * size, size * 2.2).translate(x, y))
    ax.add_patch(PathPatch(tr.transform_path(path), color=color, lw=0, zorder=5))


def fig_timezone_effect(tz: pd.DataFrame, path: Path):
    fig = plt.figure(figsize=(12, 8.5))
    header(fig, "Flying east costs more than flying west",
           "All WNBA teams, 2024-2026 · net rating vs expected, by time zone change since the previous game")
    ax = fig.add_axes([0.08, 0.33, 0.88, 0.50])
    x = tz["tz_shift_h"].values
    ax.axhline(0, color=SOFT, lw=1.2)
    ax.vlines(x, tz["ci95_low"], tz["ci95_high"], color=SOFT, lw=6, capstyle="round")
    east_drop = [(h > 0 and m < 0) for h, m in zip(x, tz["mean"])]
    ax.plot(x[x >= 0], tz["mean"][x >= 0], color=BORDEAUX, lw=1.5, ls=(0, (3, 3)), zorder=2)
    ax.scatter(x, tz["mean"], s=150, color=[BORDEAUX if e else BOREALIS for e in east_drop],
               zorder=3, edgecolor=BG, lw=2)
    for h, m, e in zip(x, tz["mean"], east_drop):
        ax.annotate(f"{m:+.1f}", (h, m), xytext=(0, 16), textcoords="offset points", ha="center",
                    fontsize=11, fontweight="bold", color=BORDEAUX if e else BOREALIS,
                    bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none"), zorder=4)
    labels = {-3: "3h west", -2: "2h west", -1: "1h west", 0: "same zone",
              1: "1h east", 2: "2h east", 3: "3h east"}
    examples = {-3: "e.g. Toronto → Seattle", -2: "e.g. Chicago → Seattle", -1: "e.g. Toronto → Chicago",
                0: "e.g. Toronto → New York", 1: "e.g. Chicago → Toronto", 2: "e.g. Seattle → Chicago",
                3: "e.g. Seattle → Toronto"}
    ax.set_xticks(x, [f"{labels.get(int(h), h)}\n{examples.get(int(h), '')}\n{n} team-games"
                      for h, n in zip(x, tz["n_games"])], fontsize=9)
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.set_ylabel("Net rating vs expected")
    hgrid(ax, [-6, -4, -2, 2, 4])
    if (tz["tz_shift_h"] == 3).any():
        top = tz.loc[tz["tz_shift_h"] == 3, "ci95_high"].iloc[0]
        ax.annotate("Every West Coast trip ends\nwith a 3h-east flight home", xy=(3, top + 0.2),
                    xytext=(3, top + 2.0), fontsize=10, color=BORDEAUX, ha="right",
                    arrowprops=dict(arrowstyle="-", color=BORDEAUX, lw=1))

    # --- travel band: which way the plane flies ---
    band = fig.add_axes([0.08, 0.075, 0.88, 0.09], sharex=ax)
    band.set_ylim(-1, 1)
    band.axis("off")
    band.annotate("", xy=(x.min() - 0.2, 0), xytext=(-0.25, 0),
                  arrowprops=dict(arrowstyle="-|>", color=BOREALIS, lw=2.5, mutation_scale=18))
    band.annotate("", xy=(x.max() + 0.2, 0), xytext=(0.25, 0),
                  arrowprops=dict(arrowstyle="-|>", color=BORDEAUX, lw=2.5, mutation_scale=18))
    _plane(band, -1.5, 0.55, BOREALIS, facing_left=True)
    _plane(band, 1.5, 0.55, BORDEAUX)
    band.text(0, 0, "●", color=NAVY, fontsize=12, ha="center", va="center")
    band.text(-1.5, -0.95, "FLYING WEST · the body clock gains hours (easier)", ha="center",
              fontsize=9.5, color=BOREALIS, fontweight="bold")
    band.text(1.5, -0.95, "FLYING EAST · the body clock loses hours (harder)", ha="center",
              fontsize=9.5, color=BORDEAUX, fontweight="bold")
    source(fig, "Games within 3 days of the previous one. Dot = average, bar = 95% CI: consistent trend, "
                "each point alone uncertain.")
    save(fig, path)


def road_trips(focus: pd.DataFrame) -> pd.DataFrame:
    """Every stretch of consecutive games played away from Toronto."""
    d = focus.sort_values("game_date").reset_index(drop=True)
    away = d["venue_city"] != "Toronto"
    d["trip_id"] = (away != away.shift()).cumsum()
    d = d[away]
    # time difference of the furthest stop vs Toronto (venue_tz: hours vs Eastern)
    out = (d.groupby("trip_id")
           .agg(start=("game_date", "min"), end=("game_date", "max"),
                cities=("venue_city", lambda s: " · ".join(dict.fromkeys(s))),
                games=("game_id", "count"),
                wins=("win", lambda s: int(s.astype(bool).sum())),
                km=("travel_km", "sum"),
                furthest_tz=("venue_tz", lambda s: s.loc[s.abs().idxmax()]),
                net_rating=("net_rating", "mean"))
           .reset_index(drop=True))
    out["record"] = out["wins"].astype(str) + "-" + (out["games"] - out["wins"]).astype(str)
    return out.round(1)


def fig_road_trips(trips: pd.DataFrame, path: Path):
    d = trips.iloc[::-1].reset_index(drop=True)
    fig = plt.figure(figsize=(13, 8.5))
    header(fig, "Toronto's road trips, one by one",
           f"Toronto Tempo {FOCUS_SEASON} · every stretch away from Toronto: average net rating, record, "
           "distance and time difference")
    ax = fig.add_axes([0.40, 0.08, 0.36, 0.74])
    colors = [BOREALIS if v >= 0 else BORDEAUX for v in d["net_rating"]]
    ax.barh(range(len(d)), d["net_rating"], color=colors, height=0.6)
    ax.axvline(0, color=NAVY, lw=1)
    lim = d["net_rating"].abs().max() * 1.3
    ax.set_xlim(-lim, lim)
    for y, v in enumerate(d["net_rating"]):
        ax.text(v + (0.8 if v >= 0 else -0.8), y, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=10, fontweight="bold",
                color=BOREALIS if v >= 0 else BORDEAUX)
    def period(a, b):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if a == b:
            return f"{a:%b} {a.day}"
        return f"{a:%b} {a.day}-{b.day}" if a.month == b.month else f"{a:%b} {a.day} - {b:%b} {b.day}"
    labels = [f"{period(s, e)}   {c}" for s, e, c in zip(d["start"], d["end"], d["cities"])]
    ax.set_yticks(range(len(d)), labels, fontsize=9.5, color=NAVY)
    ax.set_xticks([])
    ax.set_ylim(-0.6, len(d) - 0.4)
    # right-hand table
    tx = [0.79, 0.86, 0.93]
    for xx, h in zip(tx, ["record", "km", "time diff."]):
        fig.text(xx, 0.835, h, fontsize=9, color=INK_2, fontweight="bold", ha="center")
    for y, r in d.iterrows():
        yy = ax.transData.transform((0, y))[1] / fig.dpi / fig.get_figheight()
        tzt = "same" if r["furthest_tz"] == 0 else f"{int(abs(r['furthest_tz']))}h " + \
              ("west" if r["furthest_tz"] < 0 else "east")
        for xx, val in zip(tx, [r["record"], f"{r['km']:,.0f}", tzt]):
            fig.text(xx, yy, val, fontsize=9.5, color=NAVY, ha="center", va="center")
    fig.text(0.40, 0.84, "Average net rating on the trip", fontsize=9, color=INK_2, fontweight="bold")
    source(fig, "Includes 'home' games played outside Toronto (Vancouver, Montreal). "
                "km = straight line between arenas, including the flight out.")
    save(fig, path)


def standings(reg: pd.DataFrame, season: int) -> pd.DataFrame:
    d = reg[reg["season"] == season]
    st = (d.groupby("team_abbr")
          .agg(games=("game_id", "count"), wins=("win", lambda s: int(s.astype(bool).sum())),
               net_rating=("net_rating", "mean"))
          .reset_index())
    st["losses"] = st["games"] - st["wins"]
    st = st.sort_values(["wins", "net_rating"], ascending=False).reset_index(drop=True)
    st["rank"] = np.arange(1, len(st) + 1)
    return st


# =============================================================================
# 6. Main
# =============================================================================

def run(team: pd.DataFrame, player: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    # Regular season only, Commissioner's Cup final excluded (not a normal schedule game)
    reg = team[(team["phase"] == "regular") & (team["game_type"] != "CC")].copy()
    reg = reg.reset_index(drop=True)

    reg["net_vs_expected"] = add_vs_expected(reg, "net_rating")
    reg["efg_vs_expected"] = add_vs_expected(reg, "efg_pct") * 100  # in % points
    reg = pd.concat([reg, load_index(reg)], axis=1)
    cats = categorize(reg)

    # --- 1. League-wide effects ---
    eff_net = group_effects(reg, cats, "net_vs_expected")
    eff_efg = group_effects(reg, cats, "efg_vs_expected")
    eff_net.to_csv(TAB_DIR / "league_effects_net_rating.csv", index=False)
    eff_efg.to_csv(TAB_DIR / "league_effects_efg.csv", index=False)
    # Detailed 4-panel views kept as an appendix
    fig_league_effects(eff_net, FIG_DIR / "A1_league_effects_net_rating.png", "net rating")
    fig_league_effects(eff_efg, FIG_DIR / "A2_league_effects_efg.png", "eFG% (% points)",
                       title="Shooting efficiency barely moves with the schedule")

    # Main view: time zone direction (dose-response)
    tz = timezone_effects(reg)
    tz.to_csv(TAB_DIR / "league_timezone_effect.csv", index=False)
    fig_timezone_effect(tz, FIG_DIR / "01_timezone_effect.png")
    print("\nNet rating vs expected by time zone change:")
    print(tz.to_string(index=False))

    print("League-wide, net rating vs expected (points per 100 possessions):")
    print(eff_net.to_string(index=False))

    # --- 2. Season load comparison ---
    season_tab = (reg[reg["season"] == FOCUS_SEASON]
                  .groupby("team_abbr")
                  .agg(games=("game_id", "count"),
                       back_to_backs=("back_to_back", lambda s: int(s.fillna(False).astype(bool).sum())),
                       total_travel_km=("travel_km", "sum"),
                       border_crossings=("crossed_border", lambda s: int(s.fillna(False).astype(bool).sum())),
                       time_zones_crossed=("tz_shift_h", lambda s: s.abs().sum()),
                       priority_games=("load_level", lambda s: int((s == "Priority").sum())),
                       net_rating=("net_rating", "mean"))
                  .round(1).reset_index())
    season_tab.to_csv(TAB_DIR / f"season_load_{FOCUS_SEASON}.csv", index=False)
    fig_season_load(season_tab, FIG_DIR / f"A3_season_load_{FOCUS_SEASON}.png")
    standings(reg, FOCUS_SEASON).to_csv(TAB_DIR / f"standings_{FOCUS_SEASON}.csv", index=False)
    print(f"\n{FOCUS_SEASON} schedule load by team:")
    print(season_tab.sort_values("border_crossings", ascending=False).to_string(index=False))

    # --- 3. Focus team ---
    focus = reg[(reg["team_abbr"] == FOCUS_TEAM) & (reg["season"] == FOCUS_SEASON)].copy()
    fig_focus_timeline(focus, FIG_DIR / f"A4_{FOCUS_TEAM}_{FOCUS_SEASON}_timeline.png")
    fig_focus_map(focus, FIG_DIR / f"A5_{FOCUS_TEAM}_{FOCUS_SEASON}_travel_map.png")
    trips = road_trips(focus)
    trips.to_csv(TAB_DIR / f"{FOCUS_TEAM}_{FOCUS_SEASON}_road_trips.csv", index=False)
    fig_road_trips(trips, FIG_DIR / f"02_{FOCUS_TEAM}_{FOCUS_SEASON}_road_trips.png")
    print(f"\n{FOCUS_TEAM} {FOCUS_SEASON} road trips:")
    print(trips.to_string(index=False))
    cols = ["game_date", "is_home", "opponent_abbr", "venue_city", "days_rest", "travel_km",
            "tz_shift_h", "crossed_border", "games_last_7d", "load_points", "load_level",
            "net_rating", "net_vs_expected", "win"]
    focus[cols].sort_values("game_date").to_csv(
        TAB_DIR / f"{FOCUS_TEAM}_{FOCUS_SEASON}_games.csv", index=False)

    lvl = focus.groupby("load_level").agg(games=("game_id", "count"),
                                          net_vs_expected=("net_vs_expected", "mean"),
                                          wins=("win", "sum")).round(1)
    print(f"\n{FOCUS_TEAM} {FOCUS_SEASON} by load level:")
    print(lvl.to_string())
    print(f"\n{FOCUS_TEAM} {FOCUS_SEASON} - 5 most demanding games:")
    print(focus.sort_values(["load_points", "travel_km"], ascending=False)[cols].head(5)
          .to_string(index=False))

    # --- 4. Player level: efficiency on back-to-backs (players with 20+ games) ---
    p = player[(player["phase"] == "regular") & (player["game_type"] != "CC")].copy()
    p = p[p["minutes"] >= 10]
    p["gs_per36"] = p["game_score"] / p["minutes"] * 36
    p["n_games"] = p.groupby(["athlete_id", "season"])["game_id"].transform("count")
    p = p[p["n_games"] >= 20]
    p["gs_vs_own_avg"] = p["gs_per36"] - p.groupby(["athlete_id", "season"])["gs_per36"].transform("mean")
    p["min_vs_own_avg"] = p["minutes"] - p.groupby(["athlete_id", "season"])["minutes"].transform("mean")
    b2b = p["team_days_rest"] == 0
    player_tab = pd.DataFrame({
        "situation": ["Back-to-back", "Not back-to-back"],
        "player_games": [b2b.sum(), (~b2b).sum()],
        "game_score_per36_vs_own_avg": [p.loc[b2b, "gs_vs_own_avg"].mean(),
                                        p.loc[~b2b, "gs_vs_own_avg"].mean()],
        "minutes_vs_own_avg": [p.loc[b2b, "min_vs_own_avg"].mean(),
                               p.loc[~b2b, "min_vs_own_avg"].mean()],
    }).round(2)
    player_tab.to_csv(TAB_DIR / "player_back_to_back.csv", index=False)
    print("\nPlayers (20+ games, 10+ min), compared with their own season average:")
    print(player_tab.to_string(index=False))

    print(f"\nFigures: {FIG_DIR.relative_to(PROJECT_DIR)}   Tables: {TAB_DIR.relative_to(PROJECT_DIR)}")


def main() -> None:
    team, player = load_from_db()
    run(team, player)


if __name__ == "__main__":
    main()
