#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, json, time, requests, pandas as pd
from dotenv import load_dotenv, find_dotenv


# In[2]:


load_dotenv(find_dotenv())
BASE = os.getenv("APISPORTS_BASE_URL", "https://v3.football.api-sports.io")
API_KEY = os.getenv("APISPORTS_FOOTBALL_KEY")
assert API_KEY, "Missing APISPORT|S_FOOTBALL_KEY in your .env"
H = {"x-apisports-key": API_KEY}
TZ = "America/Chicago"
os.makedirs("cache", exist_ok=True)


# In[3]:


import hashlib, pathlib

CACHE_DIR = pathlib.Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def _cache_path(prefix: str, key) -> pathlib.Path:
    key_str = str(key)                     # <-- make sure it's a string
    h = hashlib.sha1(key_str.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{prefix}_{h}.json"


# In[4]:


def get_fixture(fid, *, force=False):
    p = _cache_path("fixture",fid)
    
    if(not force) and os.path.exists(p):
        return json.load(open(p, "r", encoding="utf-8"))
    js = requests.get(f"{BASE}/fixtures", headers=H, params={"id":fid}).json()["response"][0]
    json.dump(js, open(p, "w", encoding="utf-8"))
    return js


# In[5]:


def get_events(fid, *, force=False):
    p = _cache_path("events", fid)
    if (not force) and os.path.exists(p):
        return json.load(open(p, "r", encoding="utf-8"))
    js = requests.get(f"{BASE}/fixtures/events", headers=H, params={"fixture": fid}).json()["response"]
    json.dump(js, open(p, "w", encoding="utf-8"))
    return js


# In[6]:


def get_stats(fid, *, force=False):
    p = _cache_path("stats", fid)
    if (not force) and os.path.exists(p):
        return json.load(open(p, "r", encoding="utf-8"))

    r = requests.get(f"{BASE}/fixtures/statistics", headers=H, params={"fixture": fid})
    body = r.json()
    resp = body.get("response", [])
    json.dump(resp, open(p, "w", encoding="utf-8"))
    return resp


# In[7]:


def _to_pct(x):
    if x is None: return None
    s = str(x)
    if s.endswith("%"): s = s[:-1]
    try: return float(s)
    except: return None


# In[8]:


def find_fixtures_between_teams(team1, team2, season, league=39):
    """
    Return fixtures for team1 vs team2 (either home/away) in league+season.
    Uses robust substring matching so minor name differences don't break it.
    """
    r = requests.get(
        f"{BASE}/fixtures",
        headers=H,
        params={"league": league, "season": int(season)}
    )
    body = r.json()
    all_fx = body.get("response", [])

    t1 = team1.lower().strip()
    t2 = team2.lower().strip()

    matches = []
    for f in all_fx:
        h = (f["teams"]["home"]["name"] or "").lower()
        a = (f["teams"]["away"]["name"] or "").lower()

        # robust match: each input must appear in either home or away
        if ((t1 in h) or (t1 in a)) and ((t2 in h) or (t2 in a)):
            matches.append({
                "fid": f["fixture"]["id"],
                "date": f["fixture"]["date"],
                "home": f["teams"]["home"]["name"],
                "away": f["teams"]["away"]["name"],
                "home_goals": f["goals"]["home"],
                "away_goals": f["goals"]["away"],
                "league_id": f["league"]["id"],
                "league_name": f["league"]["name"],
            })

    matches.sort(key=lambda x: x["date"])
    return matches


# In[9]:


def stats_df_from(st_raw, force=False):
    if not isinstance(st_raw, list) or not st_raw:
        return pd.DataFrame()
    
    rows = []
    for s in st_raw:
        team_name = (s.get("team") or {}).get("name")
        for p in (s.get("statistics") or []):
            rows.append({
                "teams": team_name, 
                "type": p.get("type"),
                "value": p.get("value")
            }) 
        
    df = pd.DataFrame(rows)
    #if df.empty():
        #return df
    
    dfp = df.pivot_table(index="teams", columns="type", values="value", aggfunc="first").reset_index()
    return dfp
    
    
    


# In[10]:


def events_df_from(ev_raw):
    rows=[]
    
    for e in ev_raw:
        t = e.get("time") or {}
        
        rows.append({
            "minute": e.get("elapsed"),
            "team": (e.get("team") or {}).get("name"),
            "player": (e.get("player") or {}).get("name"),
            "assist": (e.get("assist") or {}).get("name"),
            "type": e.get("type"),
            "detail": e.get("detail"),
            "comments": e.get("comments")
        })
    return pd.DataFrame(rows)
    


# In[11]:


def stats_dict_from(st_raw, force=False):
    out = {}
    
    for s in (st_raw or []):
        team = (s.get("team") or {}).get("name")
        d = {}
        for p in (s.get("statistics") or []):
            d[p.get("type")] = p.get("value")
        if "Shots on Target" in d and "Shots on Goal" not in d:
            d["Shots on Goal"] = d["Shots on Target"]
        if "Ball Possession %" in d and "Ball Possession" not in d:
            d["Ball Possession"] = d["Ball Possession %"]
        out[team] = d
    return out


# In[12]:


import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def _to_number(x):
    """Return a float from values like '48%', '510 (86%)', 12, None."""
    if x is None: return None
    s = str(x)
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    if m: return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None

def _scale_pair(a, b, invert=False):
    """Scale two numbers to 0..100 (per-metric). If invert=True, lower is better."""
    A = 0.0 if a is None else float(a)
    B = 0.0 if b is None else float(b)
    mx = max(A, B, 1e-6)
    A, B = (A/mx*100.0, B/mx*100.0)
    if invert: return 100.0 - A, 100.0 - B
    return A, B

def get_team_colors(teamA, teamB):
    """
    Stable colors for teams:
    - Same two colors every time for these names.
    - Deterministic mapping: sorted names -> color indices.
    """
    palette = cm.get_cmap("tab10").colors  # 10 distinct colors
    # choose two distinct indices deterministically
    names_sorted = sorted([teamA, teamB])
    base_idx = (hash(names_sorted[0]) % 10)
    other_idx = (hash(names_sorted[1]) % 10)
    if other_idx == base_idx:
        other_idx = (other_idx + 1) % 10
    # map back to original order (teamA left color, teamB right color)
    assign = {
        names_sorted[0]: palette[base_idx],
        names_sorted[1]: palette[other_idx],
    }
    return assign[teamA], assign[teamB]


# In[13]:


def plot_h2h_splitbar(stats_map, metrics=None, invert=None, title=None, show_values=True):
    """
    stats_map: {team: {stat: value}} from stats_dict_from(st_raw)
    metrics:   list of stat names to compare (order shown top->bottom)
    invert:    set of metric names where lower-is-better (e.g., {'Fouls'})
    """
    teams = list(stats_map.keys())
    assert len(teams) == 2, "Need exactly two teams"
    A, B = teams[0], teams[1]
    colorA, colorB = get_team_colors(A, B)

    if metrics is None:
        metrics = ["Shots on Goal", "Total Shots", "Ball Possession", "Passes accurate", "Fouls"]
    invert = set() if invert is None else set(invert)

    Astats = stats_map[A]; Bstats = stats_map[B]
    labels, left_vals, right_vals, raw_pairs = [], [], [], []

    for m in metrics:
        a_raw = _to_number(Astats.get(m))
        b_raw = _to_number(Bstats.get(m))
        a_sc, b_sc = _scale_pair(a_raw, b_raw, invert=(m in invert))
        labels.append(m)
        # split bar: A goes left (negative), B goes right (positive), centered at 0
        left_vals.append(-a_sc)
        right_vals.append(+b_sc)
        raw_pairs.append((a_raw, b_raw))

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 3.6))

    ax.barh(y, left_vals,  height=0.4, color=colorA, label=A, align="center")
    ax.barh(y, right_vals, height=0.4, color=colorB, label=B, align="center")

    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(-100, 100)
    ax.set_xlabel("← " + A + "   vs   " + B + " →", fontsize=9)
    if title: ax.set_title(title, fontsize=11, pad=6)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9, frameon=False)

    if show_values:
        for yi, (lv, rv, (a_raw, b_raw)) in enumerate(zip(left_vals, right_vals, raw_pairs)):
            # numeric text near bar tips (show raw values for clarity)
            if a_raw is not None:
                ax.text(lv - 2, yi, str(a_raw), va="center", ha="right", fontsize=8, color=colorA)
            if b_raw is not None:
                ax.text(rv + 2, yi, str(b_raw), va="center", ha="left",  fontsize=8, color=colorB)

    fig.tight_layout()
    return fig


# In[14]:


import numpy as np
import matplotlib.pyplot as plt

def plot_combo_chart(stats_map, metrics=None, invert=None, title=None, show_values=True):
    teams = list(stats_map.keys())
    assert len(teams) == 2, "Need exactly two teams"
    A, B = teams
    dA, dB = stats_map[A], stats_map[B]

    if metrics is None:
        metrics = ["Shots on Goal", "Total Shots", "Ball Possession", "Passes accurate", "Fouls"]
    invert = set() if invert is None else set(invert)

    # parse raw, compute 0..100 scaled values per metric
    valsA_sc, valsB_sc, rawA, rawB = [], [], [], []
    for m in metrics:
        a_raw = dA.get(m); b_raw = dB.get(m)
        a_num = _to_number(a_raw); b_num = _to_number(b_raw)
        a_sc, b_sc = _scale_pair(a_num, b_num, invert=(m in invert))
        valsA_sc.append(a_sc); valsB_sc.append(b_sc)
        rawA.append(a_raw);   rawB.append(b_raw)

    N = len(metrics)

    # figure smaller & tidy
    fig = plt.figure(figsize=(9.0, 4.2))
    ax1 = plt.subplot(1, 2, 1, polar=True)  # radar
    ax2 = plt.subplot(1, 2, 2)              # split bars

    # ---- RADAR (left) ----
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    ang_c  = angles + [angles[0]]
    A_c    = valsA_sc + [valsA_sc[0]]
    B_c    = valsB_sc + [valsB_sc[0]]

    ax1.set_theta_offset(np.pi/2); ax1.set_theta_direction(-1)
    ax1.set_xticks(angles); ax1.set_xticklabels(metrics, fontsize=8)
    ax1.set_yticks([20, 40, 60, 80, 100]); ax1.set_yticklabels([], fontsize=6)

    colA, colB = "#d62728", "#bcbd22"
    ax1.plot(ang_c, A_c, color=colA, linewidth=1.4, label=A)
    ax1.fill(ang_c, A_c, color=colA, alpha=0.18)
    ax1.plot(ang_c, B_c, color=colB, linewidth=1.4, label=B)
    ax1.fill(ang_c, B_c, color=colB, alpha=0.18)
    if title: ax1.set_title(title, fontsize=10, pad=8)
    ax1.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.25, 1.12), frameon=False)

    # ---- HEAD-TO-HEAD SPLIT BARS (right) ----
    y = np.arange(N)
    left  = -np.array(valsA_sc)   # A to left
    right =  np.array(valsB_sc)   # B to right
    ax2.barh(y, left,  height=0.38, color=colA, label=A, align="center")
    ax2.barh(y, right, height=0.38, color=colB, label=B, align="center")
    ax2.axvline(0, color="k", linewidth=0.8)
    ax2.set_yticks(y); ax2.set_yticklabels(metrics, fontsize=8)
    ax2.set_xlim(-100, 100); ax2.grid(axis="x", linestyle=":", alpha=0.4)
    ax2.set_title("Head-to-head (scaled 0–100)", fontsize=10)
    ax2.legend(fontsize=8, loc="lower right", frameon=False)

    if show_values:
        def fmt(val, metric):
            s = str(val) if val is not None else ""
            return s if "%" in s or "Possession" in metric else s
        for yi, (lv, rv, ra, rb, m) in enumerate(zip(left, right, rawA, rawB, metrics)):
            if ra is not None: ax2.text(lv - 2, yi, fmt(ra, m), va="center", ha="right", fontsize=7, color=colA)
            if rb is not None: ax2.text(rv + 2, yi, fmt(rb, m), va="center", ha="left",  fontsize=7, color=colB)

    fig.tight_layout(pad=1.2)
    return fig


# In[15]:


from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
def make_stats_agent(stats_df):
    return create_pandas_dataframe_agent(
        llm,
        stats_df, 
        verbose=0,
        allow_dangerous_code=True,
        agent_type = AgentType.OPENAI_FUNCTIONS,
        max_iterations=5
    )
    
    


# In[16]:


from typing import Optional, List
from pydantic import BaseModel, Field

class H2HArgs(BaseModel):
    team1: str = Field(..., description="Home or away team, e.g., 'Chelsea'")
    team2: str = Field(..., description="Opponent team, e.g., 'Liverpool'")
    season: int = Field(..., description="Season year, e.g., 2023 (free plan usually 2021–2023)")
    league: int = Field(39, description="League ID (39 = EPL)")
    force: bool = Field(False, description="If true, ignore cache when calling the API")
    use_llm: bool = Field(True, description="If true, phrase the stat summary via LLM")
    save_dir: Optional[str] = Field(None, description="If set, save PNG charts into this folder")

def head_to_head_report(team1: str, team2: str, season: int, league: int = 39,
                        force: bool = False, use_llm: bool = True, save_dir: str = "charts") -> str:
    import os, matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    lines = []

    fixtures = find_fixtures_between_teams(team1, team2, season, league=league)
    if not fixtures:
        return f"No fixtures found for {team1} vs {team2} in season {season} (league {league})."

    lines = []

    for fx in fixtures:
        date = fx["date"][:10]
        home = fx["home"]
        away = fx["away"]
        score = f"{home} {fx['home_goals']}–{fx['away_goals']} {away}"

        # Header
        lines.append(f"### {date} — {score}\n")

        # ---------- STATS ----------
        st_raw = get_stats(fx["fid"], force=force)
        if st_raw:
            st_map = stats_dict_from(st_raw)

            def v(team, key):
                return st_map.get(team, {}).get(key, "—")

            # Markdown stats table
            lines.append("**Match Statistics**")
            lines.append("| Team | Shots on Goal | Total Shots | Possession |")
            lines.append("|------|----------------|-------------|------------|")
            lines.append(
                f"| {home} | {v(home, 'Shots on Goal')} | {v(home, 'Total Shots')} | "
                f"{v(home, 'Ball Possession')} |"
            )
            lines.append(
                f"| {away} | {v(away, 'Shots on Goal')} | {v(away, 'Total Shots')} | "
                f"{v(away, 'Ball Possession')} |"
            )
            lines.append("")  # blank line
        else:
            lines.append("Stats unavailable for this match.\n")

        # ---------- GOAL SCORERS ----------
        ev_raw = get_events(fx["fid"], force=force)
        if ev_raw:
            ev_df = events_df_from(ev_raw)
            goals_df = ev_df[ev_df["type"] == "Goal"]

            def scorers(team):
                tdf = goals_df[goals_df["team"] == team]
                if tdf.empty:
                    return "None"
                out = []
                for _, row in tdf.iterrows():
                    p = row.get("player") or "Unknown"
                    m = row.get("minute")
                    d = row.get("detail") or ""
                    tag = f"{m}'" if m else ""
                    if d and d.lower() not in ("normal goal",):
                        tag += f" ({d})"
                    out.append(f"{p} {tag}".strip())
                return "; ".join(out)

            lines.append("**Goal Scorers**")
            lines.append("| Team | Goal Scorers |")
            lines.append("|------|--------------|")
            lines.append(f"| {home} | {scorers(home)} |")
            lines.append(f"| {away} | {scorers(away)} |")
            lines.append("")
        else:
            lines.append("Goal data unavailable for this match.\n")

        lines.append("---\n")  # Divider between matches

    return "\n".join(lines)





