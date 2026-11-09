"""
08_roster.py — Building the roster: what does Toronto need?
===========================================================
Compares the Tempo's 2026 team profile with the playoff teams, then turns
the biggest gaps into a recruiting profile.

  1. Four factors (offense and defense): shooting, turnovers, rebounding,
     free throws - the four things that explain most wins.
  2. Shot profile from play-by-play coordinates: where Toronto shoots, and
     where opponents shoot against Toronto (rim, paint, mid-range, 3s).
  3. Shot charts for Toronto's main players.
  4. Recruiting profile: league players who match the biggest gap,
     with a durability score (share of team games played, 2024-2026).

Outputs:
  outputs/figures/12_ ... 15_*.png
  outputs/tables/four_factors_2026.csv, shot_zones_2026.csv, recruit_candidates_2026.csv

Run from the project folder (after 03 and 06, play-by-play loaded):
    python3 pipeline/08_roster.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import NAVY, INK_2, BOREALIS, BORDEAUX, SOFT, LIGHT, BG, header, source, hgrid, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc, Rectangle, Circle  # noqa: E402

_spec = importlib.util.spec_from_file_location("season07", Path(__file__).parent / "07_season.py")
s07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s07)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_FILE = PROJECT_DIR / "data" / "wnba.duckdb"
FIG_DIR = PROJECT_DIR / "outputs" / "figures"
TAB_DIR = PROJECT_DIR / "outputs" / "tables"

FOCUS_TEAM = "TOR"
SEASON = 2026
HOOP_X = 41.75                 # hoop distance from mid-court along the length (feet)
N_SHOT_CHARTS = 8
MIN_MINUTES_CANDIDATE = 400
N_CANDIDATES = 12

ZONE_ORDER = ["Rim", "Paint", "Mid-range", "Corner 3", "Above-break 3"]


# =============================================================================
# 1. Data
# =============================================================================

def load_from_db():
    import duckdb
    con = duckdb.connect(str(DB_FILE), read_only=True)
    pbp = con.execute(f"""
        SELECT game_id, season, season_type, team_id, home_team_id, away_team_id,
               type_text, text, shooting_play, scoring_play, score_value,
               coordinate_x, coordinate_y, athlete_id_1, athlete_id_2, period_number
        FROM raw.pbp WHERE season = {SEASON} AND season_type = 2
    """).df()
    team_games = con.execute("SELECT * FROM clean.team_games").df()
    player_games = con.execute("SELECT * FROM clean.player_games").df()
    games = con.execute("SELECT game_id, game_type FROM clean.games").df()
    teams = con.execute("SELECT * FROM clean.teams").df()
    team, player_box, _, _ = s07.load_from_db()
    con.close()
    return pbp, team_games, player_games, games, teams, team, player_box


# =============================================================================
# 2. Team profile: four factors
# =============================================================================

def four_factors(team_games: pd.DataFrame, games: pd.DataFrame, abbr: pd.Series) -> pd.DataFrame:
    tg = team_games.merge(games, on="game_id")
    tg = tg[(tg["season"] == SEASON) & (tg["phase"] == "regular") & (tg["game_type"] != "CC")]
    opp = tg[["game_id", "team_id", "fgm", "fga", "fg3m", "fta", "ftm", "oreb", "dreb", "tov"]] \
        .rename(columns=lambda c: c if c == "game_id" else f"opp_{c}") \
        .rename(columns={"opp_team_id": "opponent_team_id"})
    tg = tg.merge(opp, on=["game_id", "opponent_team_id"])
    a = tg.groupby("team_id").sum(numeric_only=True)
    ff = pd.DataFrame({
        "eFG%": (a["fgm"] + 0.5 * a["fg3m"]) / a["fga"],
        "TOV%": a["tov"] / (a["fga"] + 0.44 * a["fta"] + a["tov"]),
        "OREB%": a["oreb"] / (a["oreb"] + a["opp_dreb"]),
        "FT rate": a["ftm"] / a["fga"],
        "Opp eFG%": (a["opp_fgm"] + 0.5 * a["opp_fg3m"]) / a["opp_fga"],
        "Opp TOV%": a["opp_tov"] / (a["opp_fga"] + 0.44 * a["opp_fta"] + a["opp_tov"]),
        "DREB%": a["dreb"] / (a["dreb"] + a["opp_oreb"]),
        "Opp FT rate": a["opp_ftm"] / a["opp_fga"],
    })
    ff.index = ff.index.map(abbr)
    return ff


# higher is better for these, lower is better for the others
HIGHER_IS_BETTER = {"eFG%": True, "TOV%": False, "OREB%": True, "FT rate": True,
                    "Opp eFG%": False, "Opp TOV%": True, "DREB%": True, "Opp FT rate": False}


# =============================================================================
# 3. Shots: zones from coordinates
# =============================================================================

def prepare_shots(pbp: pd.DataFrame, abbr: pd.Series) -> pd.DataFrame:
    s = pbp[pbp["shooting_play"].astype(str).str.lower().eq("true")
            & ~pbp["type_text"].fillna("").str.contains("Free Throw")].copy()
    s = s.dropna(subset=["coordinate_x", "coordinate_y"])
    s["made"] = s["scoring_play"].astype(str).str.lower().eq("true")
    s["three"] = s["text"].fillna("").str.contains("three point", case=False)
    s["points"] = np.where(s["made"], np.where(s["three"], 3, 2), 0)
    # court frame: baseline at v=0, hoop at (0, 5.25), u = left/right
    s["u"] = s["coordinate_y"]
    s["v"] = 47 - s["coordinate_x"].abs()
    s["dist"] = np.sqrt((s["coordinate_x"].abs() - HOOP_X) ** 2 + s["coordinate_y"] ** 2)
    corner = s["three"] & (s["coordinate_y"].abs() >= 21)
    s["zone"] = np.select([s["three"] & corner, s["three"], s["dist"] <= 4, s["dist"] <= 14],
                          ["Corner 3", "Above-break 3", "Rim", "Paint"], "Mid-range")
    s["creation"] = s["type_text"].fillna("").map(shot_creation)
    s["quarter"] = np.where(s["period_number"] <= 4, "Q" + s["period_number"].astype(int).astype(str), "OT") \
        if "period_number" in s else "Q?"
    s["assisted"] = s["made"] & s["text"].fillna("").str.contains("assist", case=False)
    s["defending_team_id"] = np.where(s["team_id"] == s["home_team_id"], s["away_team_id"], s["home_team_id"])
    s["team"] = s["team_id"].map(abbr)
    s["defending_team"] = s["defending_team_id"].map(abbr)
    return s[s["team"].notna() & s["defending_team"].notna()]


CREATION_ORDER = ["Drive", "Cut / putback", "Off the dribble / post", "Spot-up jumper", "Other finish"]
CREATION_COLORS = {"Drive": "#3f8fd2", "Cut / putback": "#1b2a4a", "Off the dribble / post": "#7d2240",
                   "Spot-up jumper": "#e0a526", "Other finish": "#c9cfdb"}


def shot_creation(t: str) -> str:
    """How the shot was created, from ESPN's shot type."""
    if "Putback" in t or "Tip" in t or "Cutting" in t or "Alley Oop" in t:
        return "Cut / putback"
    if "Driving" in t:
        return "Drive"
    if any(k in t for k in ("Pullup", "Step Back", "Hook", "Turnaround", "Fade")):
        return "Off the dribble / post"
    if "Jump Shot" in t:
        return "Spot-up jumper"
    return "Other finish"


def zone_table(shots, side, playoff_teams):
    col = "team" if side == "offense" else "defending_team"
    grp = np.where(shots[col] == FOCUS_TEAM, FOCUS_TEAM,
                   np.where(shots[col].isin(playoff_teams), "Playoff teams", "Other teams"))
    z = (shots.assign(group=grp).groupby(["group", "zone"])
         .agg(shots=("points", "size"), points_per_shot=("points", "mean"), fg=("made", "mean")).reset_index())
    z["share"] = z["shots"] / z.groupby("group")["shots"].transform("sum")
    z["side"] = side
    return z


# =============================================================================
# 4. Recruiting: candidates + durability
# =============================================================================

def candidates(player_games, games, team_games, names_pos):
    pg = player_games.merge(games, on="game_id")
    pg = pg[(pg["phase"] == "regular") & (pg["game_type"] != "CC")]
    tg = team_games.merge(games, on="game_id")
    tg = tg[(tg["phase"] == "regular") & (tg["game_type"] != "CC")]
    team_games_n = tg.groupby(["team_id", "season"])["game_id"].nunique()

    # durability: games played / games of her team(s), over the seasons she appears in
    played = pg.groupby(["athlete_id", "season", "team_id"])["game_id"].nunique().rename("played").reset_index()
    played["team_games"] = played.set_index(["team_id", "season"]).index.map(team_games_n)
    # a player traded mid-season: count the season once, with her total games and a full season
    dur = (played.groupby(["athlete_id", "season"])
           .agg(played=("played", "sum"), team_games=("team_games", "max")).reset_index())
    dur["played"] = dur[["played", "team_games"]].min(axis=1)
    durability = dur.groupby("athlete_id").apply(
        lambda d: d["played"].sum() / d["team_games"].sum(), include_groups=False).rename("availability")

    cur = pg[pg["season"] == SEASON]
    p = cur.groupby("athlete_id").agg(
        player=("player_name", "last"), team_id=("team_id", "last"), games=("game_id", "nunique"),
        minutes=("minutes", "sum"), dreb=("dreb", "sum"), oreb=("oreb", "sum"), blk=("blk", "sum"),
        stl=("stl", "sum"), pf=("pf", "sum"), pts=("points", "sum"),
        fga=("fga", "sum"), fta=("fta", "sum"), fg3a=("fg3a", "sum"), fg3m=("fg3m", "sum"))
    p = p[p["minutes"] >= MIN_MINUTES_CANDIDATE].copy()
    per36 = 36 / p["minutes"]
    p["dreb_36"] = p["dreb"] * per36
    p["blk_36"] = p["blk"] * per36
    p["stl_36"] = p["stl"] * per36
    p["ts_pct"] = p["pts"] / (2 * (p["fga"] + 0.44 * p["fta"]))
    p = p.join(durability).join(names_pos)
    # defensive interior score: percentile of DREB/36 and BLK/36 (equal weight)
    p["def_interior_score"] = (p["dreb_36"].rank(pct=True) + p["blk_36"].rank(pct=True)) / 2 * 100
    return p.round(3)


# =============================================================================
# 5. Figures
# =============================================================================

PLAIN_NAMES = {
    "eFG%": ("Shooting", "eFG%"),
    "TOV%": ("Taking care of the ball", "turnovers per 100 poss."),
    "OREB%": ("Offensive rebounding", "OREB%"),
    "FT rate": ("Getting to the line", "free throws made per 100 shots"),
    "Opp eFG%": ("Stopping opponents' shots", "opp. eFG%"),
    "Opp TOV%": ("Forcing turnovers", "opp. turnovers per 100 poss."),
    "DREB%": ("Defensive rebounding", "DREB%"),
    "Opp FT rate": ("Not fouling", "opp. free throws made per 100 shots"),
}


def fig_four_factors(ff, playoff_teams, path):
    """One bar per factor: Toronto's gap to the playoff teams, in league standard deviations."""
    rows = []
    for m, hib in HIGHER_IS_BETTER.items():
        sign = 1 if hib else -1
        tor, po = ff.loc[FOCUS_TEAM, m], ff.loc[playoff_teams, m].mean()
        rows.append({"metric": m, "side": "Defense" if m.startswith("Opp") or m == "DREB%" else "Offense",
                     "gap": sign * (tor - po) / ff[m].std(), "tor": tor * 100, "po": po * 100,
                     "rank": int((sign * ff[m] > sign * tor).sum() + 1)})
    d = pd.DataFrame(rows)
    d["order"] = d["side"].map({"Offense": 0, "Defense": 1})
    d = d.sort_values(["order", "gap"], ascending=[True, True]).reset_index(drop=True)

    fig = plt.figure(figsize=(11, 7.2))
    header(fig, "Toronto's biggest gaps are on defense",
           "2026 · Toronto vs the playoff teams on the four factors · bar to the left = worse, in league standard deviations")
    ax = fig.add_axes([0.30, 0.10, 0.38, 0.72])
    y = []
    pos = 0
    for i, r in d.iterrows():
        if i > 0 and r["side"] != d.loc[i - 1, "side"]:
            pos += 0.9
        y.append(pos)
        pos += 1
    y = np.array(y)[::-1] if False else np.array(y)
    y = y.max() - y
    colors = [BORDEAUX if g <= -1.5 else ("#c98a9c" if g < 0 else BOREALIS) for g in d["gap"]]
    ax.barh(y, d["gap"], color=colors, height=0.66)
    ax.axvline(0, color=NAVY, lw=1)
    ax.set_xlim(-2.7, 1.0)
    ax.set_ylim(-0.6, y.max() + 0.6)
    ax.set_yticks([])
    ax.set_xticks([-2, -1, 0, 1], ["−2 SD", "−1 SD", "playoff\nlevel", "+1 SD"], fontsize=9)
    for yy, (_, r) in zip(y, d.iterrows()):
        name, unit = PLAIN_NAMES[r["metric"]]
        fy = fig.transFigure.inverted().transform(ax.transData.transform((0, yy)))[1]
        fig.text(0.04, fy + 0.012, name, fontsize=11, fontweight="bold", color=NAVY, va="center")
        fig.text(0.04, fy - 0.016, unit, fontsize=8.5, color=INK_2, va="center")
        ax.text(r["gap"] + (-0.06 if r["gap"] < 0 else 0.06), yy, f"{r['gap']:+.1f}",
                ha="right" if r["gap"] < 0 else "left", va="center", fontsize=10, fontweight="bold",
                color=BORDEAUX if r["gap"] < 0 else BOREALIS)
        fig.text(0.72, fy, f"{r['tor']:.1f}", fontsize=10.5, fontweight="bold", color=BORDEAUX, va="center", ha="center")
        fig.text(0.81, fy, f"{r['po']:.1f}", fontsize=10.5, color=BOREALIS, va="center", ha="center")
        rk = r["rank"]
        fig.text(0.90, fy, f"{rk}{'st' if rk == 1 else 'nd' if rk == 2 else 'rd' if rk == 3 else 'th'}",
                 fontsize=10.5, color=NAVY, va="center", ha="center")
    top = fig.transFigure.inverted().transform(ax.transData.transform((0, y.max() + 0.5)))[1]
    for xx, h in [(0.72, "Toronto"), (0.81, "Playoff\nteams"), (0.90, "Rank\n(of 15)")]:
        fig.text(xx, top + 0.01, h, fontsize=8.5, color=INK_2, fontweight="bold", ha="center", va="bottom")
    split = (y[d["side"] == "Offense"].min() + y[d["side"] == "Defense"].max()) / 2
    ax.axhline(split, color=LIGHT, lw=1)
    for side in ["Offense", "Defense"]:
        yy = y[d["side"] == side].max()
        fy = fig.transFigure.inverted().transform(ax.transData.transform((0, yy + 0.62)))[1]
        fig.text(0.04, fy, side.upper(), fontsize=9, color=INK_2, fontweight="bold")
    source(fig, "SD = spread across the 15 teams. Playoff teams = average of the 8. eFG% = shooting % with 3s worth 1.5 shots.")
    save(fig, path)


def _colored_line(fig, x, y, parts, fontsize=12):
    """Write a line made of (text, color) pieces, one after the other."""
    renderer = fig.canvas.get_renderer()
    for txt, col in parts:
        t = fig.text(x, y, txt, fontsize=fontsize, fontweight="bold", color=col, va="bottom")
        bb = t.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        x = bb.x1


def fig_shot_zones(zones, path):
    """Paired bars: where shots were taken, and how well they went in, Toronto vs playoff teams."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.80, bottom=0.07, wspace=0.10, hspace=0.24)
    header(fig, "Opponents got to the rim against Toronto, and scored there",
           f"{SEASON} · by zone: share of shot attempts (made or missed), then FG% · Toronto vs the 8 playoff teams")
    y = np.arange(len(ZONE_ORDER))[::-1].astype(float)
    h = 0.36
    rows = [("share", "Where the shots were taken (share of attempts)", 42, "{:.0f}%"),
            ("fg", "How often they went in (FG%)", 85, "{:.0f}%")]
    for r, (metric, row_title, xmax, fmt) in enumerate(rows):
        for c, side in enumerate(["offense", "defense"]):
            ax = axes[r, c]
            z = zones[zones["side"] == side]
            po = z[z["group"] == "Playoff teams"].set_index("zone").reindex(ZONE_ORDER)[metric] * 100
            to = z[z["group"] == FOCUS_TEAM].set_index("zone").reindex(ZONE_ORDER)[metric] * 100
            if side == "defense":
                ax.axhspan(y[0] - 0.5, y[0] + 0.5, color=LIGHT, zorder=0)
            ax.barh(y + h / 2, to, height=h, color=BORDEAUX, zorder=2)
            ax.barh(y - h / 2, po, height=h, color=BOREALIS, zorder=2)
            for yy, a_, b_ in zip(y, po, to):
                ax.text(b_ + xmax * 0.015, yy + h / 2, fmt.format(b_), va="center", fontsize=9.5,
                        color=BORDEAUX, fontweight="bold")
                ax.text(a_ + xmax * 0.015, yy - h / 2, fmt.format(a_), va="center", fontsize=9.5,
                        color=BOREALIS, fontweight="bold")
            if r == 0:
                ax.text(xmax * 0.015, y[0] + h / 2, "Toronto", va="center", fontsize=8.5, color=BG,
                        fontweight="bold", zorder=3)
                ax.text(xmax * 0.015, y[0] - h / 2, "Playoff teams", va="center", fontsize=8.5,
                        color=BG, fontweight="bold", zorder=3)
            ax.set_xlim(0, xmax)
            ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
            ax.set_xticks([])
            ax.set_title(row_title, fontsize=10, color=INK_2, fontweight="normal", pad=6)
        axes[r, 0].set_yticks(y, ZONE_ORDER, fontsize=11, color=NAVY)
    # column titles, with the colours spelled out
    top = axes[0, 0].get_position().y1 + 0.035
    _colored_line(fig, axes[0, 0].get_position().x0, top,
                  [("Shots taken: ", NAVY), ("Toronto", BORDEAUX), (" vs ", NAVY), ("playoff teams", BOREALIS)])
    _colored_line(fig, axes[0, 1].get_position().x0, top,
                  [("Shots allowed: ", NAVY), ("Toronto", BORDEAUX), (" vs ", NAVY), ("playoff teams", BOREALIS)])
    source(fig, "Rim = within 4 ft of the basket, paint = 4-14 ft, mid-range = 2-pointers beyond 14 ft. From ESPN shot coordinates.")
    save(fig, path)


def draw_half_court(ax, color=None, lw=1):
    c = color or INK_2
    ax.add_patch(Rectangle((-25, 0), 50, 47, fill=False, lw=lw, ec=c))
    ax.add_patch(Rectangle((-8, 0), 16, 19, fill=False, lw=lw, ec=c))
    ax.add_patch(Circle((0, 5.25), 0.75, fill=False, lw=lw, ec=c))
    r = 22.15
    yc = 5.25 + np.sqrt(r ** 2 - 22 ** 2)
    ax.plot([-22, -22], [0, yc], color=c, lw=lw)
    ax.plot([22, 22], [0, yc], color=c, lw=lw)
    th = np.degrees(np.arccos(22 / r))
    ax.add_patch(Arc((0, 5.25), 2 * r, 2 * r, theta1=th, theta2=180 - th, lw=lw, ec=c))
    ax.set_xlim(-25.5, 25.5)
    ax.set_ylim(-0.5, 34)
    ax.set_aspect("equal")
    ax.axis("off")


def _court_zone_grid(step=0.25):
    """Zone of every point of the half-court, with the same rules as the shots."""
    u, v = np.meshgrid(np.arange(-25, 25 + step, step), np.arange(0, 34 + step, step))
    dist = np.sqrt(u ** 2 + (v - 5.25) ** 2)
    three = (dist > 22.15) | ((np.abs(u) >= 22) & (v <= 5.25 + np.sqrt(22.15 ** 2 - 22 ** 2)))
    zone = np.select([three & (np.abs(u) >= 21), three, dist <= 4, dist <= 14],
                     [3, 4, 0, 1], 2)            # index in ZONE_ORDER
    return u, v, zone


ZONE_LABEL_XY = {"Rim": (0, 3.2), "Paint": (0, 13.5), "Mid-range": (0, 22.5),
                 "Corner 3": (23.5, 3.5), "Above-break 3": (0, 30.5)}


def fig_shot_charts(shots, tor_names, jerseys, path):
    from matplotlib.colors import LinearSegmentedColormap
    t = shots[shots["team"] == FOCUS_TEAM]
    league = shots.groupby("zone")["points"].mean()
    top = t.groupby("athlete_id_1").size().sort_values(ascending=False).head(N_SHOT_CHARTS).index
    cmap = LinearSegmentedColormap.from_list("eff", [BORDEAUX, "#c98a9c", "#cfd5df", "#8fbbe6", BOREALIS])
    u, v, zgrid = _court_zone_grid()

    fig, axes = plt.subplots(2, 4, figsize=(13, 9))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.79, bottom=0.12, hspace=0.42, wspace=0.06)
    header(fig, "Where each player scores efficiently",
           f"Toronto Tempo {SEASON} · each zone coloured by points per shot vs the league average in that zone "
           "· two numbers per zone (see key below)")
    for ax, pid in zip(axes.flat, top):
        p = t[t["athlete_id_1"] == pid]
        stats = p.groupby("zone").agg(n=("points", "size"), pps=("points", "mean"), fg=("made", "mean"))
        diff = np.full(zgrid.shape, np.nan)
        for i, z in enumerate(ZONE_ORDER):
            if z in stats.index and stats.loc[z, "n"] >= 8:
                diff[zgrid == i] = stats.loc[z, "pps"] - league[z]
        ax.imshow(np.ma.masked_invalid(diff), origin="lower", extent=(-25, 25, 0, 34), cmap=cmap,
                  vmin=-0.4, vmax=0.4, interpolation="nearest", zorder=0)
        ax.imshow(np.ma.masked_where(~np.isnan(diff), np.ones_like(diff)), origin="lower",
                  extent=(-25, 25, 0, 34), cmap=LinearSegmentedColormap.from_list("g", [LIGHT, LIGHT]),
                  interpolation="nearest", zorder=0)
        draw_half_court(ax, color=INK_2, lw=0.8)
        for z, (x, y) in ZONE_LABEL_XY.items():
            if z not in stats.index:
                continue
            share = stats.loc[z, "n"] / len(p) * 100
            txt = f"{share:.0f}%\n{stats.loc[z, 'fg'] * 100:.0f}%"
            for xx in ([-x, x] if z == "Corner 3" else [x]):
                ax.text(xx, y, txt, ha="center", va="center", fontsize=7.5, color=NAVY, fontweight="bold",
                        linespacing=1.1, bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.75))
        name = tor_names.get(pid, str(pid))
        num = jerseys.get(pid)
        ax.set_title(name, fontsize=11, pad=8, loc="left", x=0.13)
        if num is not None:
            ax.text(-23.2, 36.6, f"{int(num)}", ha="center", va="center", fontsize=9.5, fontweight="bold",
                    color=BG, bbox=dict(boxstyle="circle,pad=0.35", fc=NAVY, ec="none"), clip_on=False)
        efg = (p["made"].sum() + 0.5 * (p["made"] & p["three"]).sum()) / len(p)
        ax.text(0, -1.5, f"{len(p)} shots · eFG {efg * 100:.0f}%", ha="center", va="top",
                fontsize=8.5, color=INK_2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-0.4, 0.4))
    cax = fig.add_axes([0.35, 0.085, 0.30, 0.013])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[-0.4, 0, 0.4])
    cb.ax.set_xticklabels(["less efficient", "league average", "more efficient"],
                          fontsize=8, color=INK_2)
    cb.outline.set_visible(False)
    fig.text(0.13, 0.083, "37%\n38%", ha="center", va="center", fontsize=8, color=NAVY, fontweight="bold",
             linespacing=1.1, bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec=SOFT))
    fig.text(0.15, 0.092, "share of her shots taken in the zone", fontsize=8, color=INK_2, va="center")
    fig.text(0.15, 0.074, "% of those shots made (FG%)", fontsize=8, color=INK_2, va="center")
    source(fig, "Pale = fewer than 8 shots in the zone. Points per shot from ESPN shot coordinates. "
                "Number = jersey.")
    save(fig, path)


def fig_shot_creation(shots, tor_names, jerseys, path):
    t = shots[shots["team"] == FOCUS_TEAM]
    top = list(t.groupby("athlete_id_1").size().sort_values(ascending=False).head(N_SHOT_CHARTS).index)
    mix = (t[t["athlete_id_1"].isin(top)].groupby(["athlete_id_1", "creation"]).size()
           .unstack(fill_value=0).reindex(columns=CREATION_ORDER, fill_value=0))
    mix = mix.div(mix.sum(axis=1), axis=0).reindex(top)
    eff = (t.groupby(["athlete_id_1", "creation"])["points"].mean().unstack().reindex(top))
    assisted = (t[t["made"]].groupby("athlete_id_1")["assisted"].mean().reindex(top))

    fig = plt.figure(figsize=(13, 7.5))
    header(fig, "How each player creates her shots",
           f"Toronto Tempo {SEASON} · shot mix by type (bars, with points per shot) and share of made shots that were assisted")
    ax = fig.add_axes([0.22, 0.14, 0.62, 0.66])
    y = np.arange(len(top))[::-1]
    left = np.zeros(len(top))
    for cat in CREATION_ORDER:
        w = mix[cat].values * 100
        ax.barh(y, w, left=left, color=CREATION_COLORS[cat], height=0.62, label=cat,
                edgecolor=BG, linewidth=1.5)
        for yy, l, ww, pid in zip(y, left, w, top):
            if ww >= 9:
                pps = eff.loc[pid, cat]
                dark = cat in ("Cut / putback", "Off the dribble / post", "Drive")
                ax.text(l + ww / 2, yy, f"{ww:.0f}%\n{pps:.2f}", ha="center", va="center", fontsize=7.5,
                        color=BG if dark else NAVY, linespacing=1.05)
        left += w
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    ax.set_yticks(y, [""] * len(y))
    for yy, pid in zip(y, top):
        fig_y = ax.transData.transform((0, yy))[1] / fig.dpi / fig.get_figheight()
        num = jerseys.get(pid)
        if num is not None:
            fig.text(0.035, fig_y, f"{int(num)}", ha="center", va="center", fontsize=9.5, fontweight="bold",
                     color=BG, bbox=dict(boxstyle="circle,pad=0.35", fc=NAVY, ec="none"))
        fig.text(0.06, fig_y, tor_names.get(pid, str(pid)), va="center", fontsize=10.5, color=NAVY,
                 fontweight="bold")
        fig.text(0.87, fig_y, f"{assisted[pid] * 100:.0f}%", va="center", fontsize=11, color=NAVY,
                 fontweight="bold")
    fig.text(0.87, 0.82, "assisted\nmakes", fontsize=8.5, color=INK_2, fontweight="bold", va="bottom")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=5, fontsize=9, handlelength=1.2)
    source(fig, "Shot types as labelled by ESPN. Spot-up = jump shot not off the dribble (mostly catch-and-shoot). "
                "Inside bars: share of her shots / points per shot.")
    save(fig, path)


def export_shots_js(shots, names, jerseys, path):
    """Toronto's shots + league benchmarks, as a small JS file for the interactive shot explorer."""
    import json
    t = shots[shots["team"] == FOCUS_TEAM]
    rows = [[names.get(pid, str(pid)), round(float(u), 1), round(float(v), 1), int(m), c, z, q, int(th)]
            for pid, u, v, m, c, z, q, th in zip(t["athlete_id_1"], t["u"], t["v"], t["made"], t["creation"],
                                                    t["zone"], t["quarter"], t["three"])]
    lg = (shots.groupby(["creation", "quarter", "zone"])
          .agg(n=("points", "size"), made=("made", "sum"), pts=("points", "sum")).reset_index())
    league = [[r.creation, r.quarter, r.zone, int(r.n), int(r.made), int(r.pts)] for r in lg.itertuples()]
    order = list(t.groupby("athlete_id_1").size().sort_values(ascending=False).index)
    players = [[names.get(pid, str(pid)), int(jerseys[pid]) if pid in jerseys else None] for pid in order]
    data = {"cols": ["player", "u", "v", "made", "creation", "zone", "quarter", "three"],
            "shots": rows, "league": league, "players": players, "season": SEASON}
    path.write_text("window.SHOTS_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")


def creation_table(shots, player_games, min_minutes=200):
    """Points each player creates per 36 min: her own unassisted baskets + the baskets she assists."""
    made = shots[shots["made"]]
    own = made[~made["assisted"]].groupby("athlete_id_1")["points"].agg(["size", "sum"])
    own.columns = ["unassisted_fgm", "own_created_pts"]
    ast = made[made["assisted"]].dropna(subset=["athlete_id_2"]).copy()
    ast["athlete_id_2"] = ast["athlete_id_2"].astype(int)
    ast = ast.groupby("athlete_id_2")["points"].agg(["size", "sum"])
    ast.columns = ["assists", "assist_pts"]
    cur = player_games[(player_games["season"] == SEASON) & (player_games["phase"] == "regular")]
    p = cur.groupby("athlete_id").agg(player=("player_name", "last"), team_id=("team_id", "last"),
                                      minutes=("minutes", "sum"), tov=("tov", "sum"))
    d = p.join(own).join(ast).fillna(0)
    d = d[d["minutes"] >= min_minutes].copy()
    d.index = d.index.astype(int)
    k = 36 / d["minutes"]
    d["own_pts_36"] = d["own_created_pts"] * k
    d["assist_pts_36"] = d["assist_pts"] * k
    d["created_pts_36"] = d["own_pts_36"] + d["assist_pts_36"]
    d["ast_36"] = d["assists"] * k
    d["tov_36"] = d["tov"] * k
    return d


def fig_creation(cre, tor_id, jerseys, tpb_missed, by_period, path):
    d = cre[cre["team_id"] == tor_id].sort_values("created_pts_36").tail(10)
    fig = plt.figure(figsize=(13, 8))
    header(fig, "Four players created Toronto's offense, and three of them got hurt",
           "Toronto Tempo 2026 · left: what each player creates · right: what the team created as Sykes and Rice dropped out")
    ax = fig.add_axes([0.20, 0.12, 0.42, 0.68])
    y = np.arange(len(d))
    ax.barh(y, d["own_pts_36"], color=BORDEAUX, height=0.62, label="Own unassisted baskets")
    ax.barh(y, d["assist_pts_36"], left=d["own_pts_36"], color=BOREALIS, height=0.62,
            label="Baskets she assisted")
    ax.set_yticks(y, [""] * len(y))
    ax.set_xticks([])
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_xlim(0, d["created_pts_36"].max() * 1.6)
    for yy, (pid, r) in zip(y, d.iterrows()):
        miss = tpb_missed.get(pid, 0)
        ax.text(r["created_pts_36"] + 0.3, yy, f"{r['created_pts_36']:.1f}" + (f"   {miss} games missed" if miss else ""),
                va="center", fontsize=9, color=BORDEAUX if miss >= 10 else NAVY,
                fontweight="bold" if r["created_pts_36"] >= 15 else "normal")
        num = jerseys.get(pid)
        fy = fig.transFigure.inverted().transform(ax.transData.transform((0, yy)))[1]
        if num is not None:
            fig.text(0.03, fy, f"{int(num)}", ha="center", va="center", fontsize=9, fontweight="bold", color=BG,
                     bbox=dict(boxstyle="circle,pad=0.35", fc=NAVY, ec="none"))
        fig.text(0.055, fy, r["player"], va="center", fontsize=10, color=NAVY, fontweight="bold")
    fig.text(0.03, 0.83, "Points created per 36 minutes", fontsize=11, fontweight="bold", color=NAVY)
    ax.legend(loc="lower right", fontsize=9)

    ax2 = fig.add_axes([0.70, 0.20, 0.27, 0.55])
    x = np.arange(len(by_period))
    ax2.bar(x, by_period["unassisted_fgm_pg"], color=[NAVY] + [SOFT] * (len(by_period) - 1), width=0.6)
    for xi, v in zip(x, by_period["unassisted_fgm_pg"]):
        ax2.text(xi, v + 0.2, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold", color=NAVY)
    ax2.set_xticks(x, by_period["label"], fontsize=8.5)
    ax2.set_yticks([])
    ax2.set_ylim(0, by_period["unassisted_fgm_pg"].max() * 1.25)
    ax2.set_title("Toronto baskets scored without an assist\n(per game, by stretch of the season)", fontsize=10.5)
    source(fig, "Assists and unassisted baskets from ESPN play-by-play. Games missed = through injury or illness, out of 44.")
    save(fig, path)


def creators(cre, availability):
    d = cre.join(availability)
    d["creator_score"] = (d["created_pts_36"].rank(pct=True) * 2 + (-d["tov_36"]).rank(pct=True)) / 3 * 100
    return d


def fig_candidates(cand, tor_id, path):
    fig = plt.figure(figsize=(12, 8))
    header(fig, "A profile worth adding: a durable interior defender",
           f"WNBA {SEASON}, players with {MIN_MINUTES_CANDIDATE}+ minutes · defensive rebounds and blocks per 36 min "
           "· bigger dot = more games played 2024-2026")
    ax = fig.add_axes([0.08, 0.10, 0.88, 0.72])
    size = 15 + 180 * cand["availability"].fillna(0) ** 3
    ax.scatter(cand["dreb_36"], cand["blk_36"], s=size, color=SOFT, alpha=0.8, lw=0)
    tor = cand[cand["team_id"] == tor_id]
    ax.scatter(tor["dreb_36"], tor["blk_36"], s=size[tor.index], color=BORDEAUX, lw=0, label="Toronto players")
    best = cand[(cand["team_id"] != tor_id) & (cand["availability"] >= 0.85)] \
        .sort_values("def_interior_score", ascending=False).head(N_CANDIDATES)
    ax.scatter(best["dreb_36"], best["blk_36"], s=size[best.index], color=BOREALIS, lw=0,
               label=f"Best {N_CANDIDATES} on rebounds + blocks, among players available 85%+ of games")
    for _, r in pd.concat([best, tor]).iterrows():
        ax.annotate(r["player"].split()[-1], (r["dreb_36"], r["blk_36"]), xytext=(6, 4),
                    textcoords="offset points", fontsize=8,
                    color=BORDEAUX if r["team_id"] == tor_id else NAVY)
    ax.set_xlabel("Defensive rebounds per 36 min")
    ax.set_ylabel("Blocks per 36 min")
    hgrid(ax, [1, 2, 3])
    ax.text(0.99, 0.97, "top right = protects the rim\nand finishes the defensive possession", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=INK_2, style="italic")
    ax.legend(loc="upper left", fontsize=9)
    source(fig, "Availability = share of her team's regular-season games played, 2024-2026. "
                "Contract status not checked: a profile, not a shopping list.")
    save(fig, path)


# =============================================================================
# 6. Main
# =============================================================================

def run(pbp, team_games, player_games, games, teams, team, player_box=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    abbr = teams.set_index("team_id")["team_abbr"]

    playoffs = s07.playoff_results(team, SEASON)
    playoff_teams = list(playoffs["team_abbr"])

    ff = four_factors(team_games, games, abbr)
    ff.round(4).to_csv(TAB_DIR / f"four_factors_{SEASON}.csv")
    fig_four_factors(ff, playoff_teams, FIG_DIR / "12_four_factors.png")

    shots = prepare_shots(pbp, abbr)
    zones = pd.concat([zone_table(shots, "offense", playoff_teams),
                       zone_table(shots, "defense", playoff_teams)])
    zones.round(3).to_csv(TAB_DIR / f"shot_zones_{SEASON}.csv", index=False)
    fig_shot_zones(zones, FIG_DIR / "13_shot_zones.png")

    names = player_games.sort_values("game_date").groupby("athlete_id")["player_name"].last()
    jerseys = {}
    if player_box is not None and "athlete_jersey" in player_box:
        tor_rows = player_box[player_box["team_id"].map(abbr) == FOCUS_TEAM]
        jerseys = (tor_rows.dropna(subset=["athlete_jersey"]).sort_values("game_date")
                   .drop_duplicates("athlete_id", keep="last")
                   .set_index("athlete_id")["athlete_jersey"].astype(float).to_dict())
    fig_shot_charts(shots, names.to_dict(), jerseys, FIG_DIR / f"14_{FOCUS_TEAM}_shot_charts.png")
    fig_shot_creation(shots, names.to_dict(), jerseys, FIG_DIR / f"14b_{FOCUS_TEAM}_shot_creation.png")
    export_shots_js(shots, names.to_dict(), jerseys, TAB_DIR / f"{FOCUS_TEAM.lower()}_shots_{SEASON}.js")

    names_pos = player_games.groupby("athlete_id")["position"].last().rename("position")
    cand = candidates(player_games, games, team_games, names_pos)
    tor_id = teams.loc[teams["team_abbr"] == FOCUS_TEAM, "team_id"].iloc[0]
    cand["team"] = cand["team_id"].map(abbr)
    cand.sort_values("def_interior_score", ascending=False).to_csv(
        TAB_DIR / f"recruit_candidates_{SEASON}.csv")
    fig_candidates(cand, tor_id, FIG_DIR / "15_recruit_profile.png")

    # --- shot creation: who creates for the team, and what happened when they were out ---
    cre = creation_table(shots, player_games)
    missed = {}
    if player_box is not None:
        pb = player_box[(player_box["team_id"] == tor_id) & (player_box["season_type"] == 2)].copy()
        st = s07.player_status(pb)
        missed = pb[st == "Injured / ill"].groupby("athlete_id").size().to_dict()
    tor_games = (team[(team["team_abbr"] == FOCUS_TEAM) & (team["season"] == SEASON) & (team["phase"] == "regular")
                      & (team["game_type"] != "CC")].sort_values("game_date").reset_index(drop=True))
    tor_games["game_no"] = tor_games.index + 1
    made_t = shots[(shots["team"] == FOCUS_TEAM) & shots["made"]].merge(tor_games[["game_id", "game_no"]], on="game_id")
    bins = [0, 10, 16, 26, len(tor_games)]
    labels = ["Games 1-10\nSykes + Rice", "11-16\nRice out", "17-26\nboth out", f"27-{len(tor_games)}\nSykes out"]
    made_t["period"] = pd.cut(made_t["game_no"], bins, labels=labels)
    n_games = pd.cut(tor_games["game_no"], bins, labels=labels).value_counts().reindex(labels)
    by_period = pd.DataFrame({"label": labels,
                              "unassisted_fgm_pg": (made_t[~made_t["assisted"]].groupby("period", observed=False)
                                                    .size().reindex(labels) / n_games).values})
    by_period.round(2).to_csv(TAB_DIR / f"{FOCUS_TEAM.lower()}_creation_by_period.csv", index=False)
    cre.round(2).to_csv(TAB_DIR / f"creation_{SEASON}.csv")
    fig_creation(cre, tor_id, jerseys, missed, by_period, FIG_DIR / f"14c_{FOCUS_TEAM}_creators.png")

    cr = creators(cre, cand["availability"])
    cr["team"] = cr["team_id"].map(abbr)
    cr.sort_values("creator_score", ascending=False).round(3).to_csv(TAB_DIR / f"creator_candidates_{SEASON}.csv")
    print(f"\n{FOCUS_TEAM} creators (points created per 36):")
    print(cre[cre["team_id"] == tor_id].sort_values("created_pts_36", ascending=False)
          [["player", "minutes", "own_pts_36", "assist_pts_36", "created_pts_36", "ast_36"]].round(1).to_string())
    print("\nUnassisted baskets per game by period:")
    print(by_period.round(2).to_string(index=False))
    print(f"\nTop creators (85%+ available, not {FOCUS_TEAM}):")
    print(cr[(cr["team"] != FOCUS_TEAM) & (cr["availability"] >= 0.85)]
          .sort_values("creator_score", ascending=False)
          [["player", "team", "minutes", "created_pts_36", "ast_36", "tov_36", "availability"]]
          .head(N_CANDIDATES).round(2).to_string(index=False))

    pd.set_option("display.width", 200)
    print("Four factors (%), Toronto vs playoff teams:")
    show = pd.DataFrame({FOCUS_TEAM: ff.loc[FOCUS_TEAM], "Playoff teams": ff.loc[playoff_teams].mean(),
                         "League": ff.mean()}) * 100
    print(show.round(1).to_string())
    print("\nShot zones (share, points per shot):")
    print(zones.pivot_table(index=["side", "zone"], columns="group", values=["share", "points_per_shot"])
          .round(3).to_string())
    print(f"\nTop interior-defense profiles (85%+ available, not {FOCUS_TEAM}):")
    print(cand[(cand["team"] != FOCUS_TEAM) & (cand["availability"] >= 0.85)]
          .sort_values("def_interior_score", ascending=False)
          [["player", "team", "position", "minutes", "dreb_36", "blk_36", "availability"]]
          .head(N_CANDIDATES).round(2).to_string(index=False))
    print(f"\nFigures: {FIG_DIR.relative_to(PROJECT_DIR)}   Tables: {TAB_DIR.relative_to(PROJECT_DIR)}")


def main() -> None:
    run(*load_from_db())


if __name__ == "__main__":
    main()
