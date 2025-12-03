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
    key_str = str(key)                     
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

    fixtures = find_fixtures_between_teams(team1, team2, season, league=league)
    if not fixtures:
        return f"No fixtures found for {team1} vs {team2} in season {season} (league {league})."

    lines = []

    for fx in fixtures:
        date = fx["date"][:10]
        home = fx["home"]
        away = fx["away"]
        score = f"{home} {fx['home_goals']}–{fx['away_goals']} {away}"

        # ---------- HEADER ----------
        lines.append(f"### {date} — {score}\n")

        # ---------- STATS ----------
        st_raw = get_stats(fx["fid"], force=force)
        if st_raw:
            st_map = stats_dict_from(st_raw)  # {team: {metric: value}}
            home_stats = st_map.get(home, {}) or {}
            away_stats = st_map.get(away, {}) or {}

            def stat_val(d: dict, keys):
                """Try several possible keys for the same concept."""
                for k in keys:
                    v = d.get(k)
                    if v not in (None, "", "-"):
                        return v
                return "—"

            # (label, [possible keys in API])
            metric_specs = [
                ("Ball Possession %", ["Ball Possession", "Ball Possession %"]),
                ("Shots on Goal", ["Shots on Goal", "Shots on Target"]),
                ("Total Shots", ["Total Shots", "Shots total"]),
                ("Shots off Target", ["Shots off Target", "Shots off goal"]),
                ("Big Chances", ["Big Chances"]),
                ("Big Chances Missed", ["Big Chances Missed"]),
                ("Total Passes", ["Total passes", "Passes total"]),
                ("Passes Accurate", ["Passes accurate"]),
                ("Pass Accuracy %", ["Passes %"]),
                ("Dribbles Succeeded", ["Dribbles succeeded", "Dribbles succeeded "]),
                ("Tackles", ["Tackles"]),
                ("Interceptions", ["Interceptions"]),
                ("Clearances", ["Clearances"]),
                ("Duels Won", ["Duels won"]),
                ("Aerial Duels Won", ["Aerials won", "Aerial duels won"]),
                ("Fouls", ["Fouls"]),
                ("Yellow Cards", ["Yellow Cards", "Yellow cards"]),
                ("Red Cards", ["Red Cards", "Red cards"]),
                ("Saves", ["Saves"]),
            ]

            lines.append("**Match Statistics**")
            lines.append(f"| Metric | {home} | {away} |")
            lines.append("|--------|-------|-------|")

            for label, keys in metric_specs:
                hv = stat_val(home_stats, keys)
                av = stat_val(away_stats, keys)
                # skip metrics where both sides have nothing
                if hv == "—" and av == "—":
                    continue
                lines.append(f"| {label} | {hv} | {av} |")

            lines.append("")  # blank line after table
        else:
            lines.append("Stats unavailable for this match.\n")

        # ---------- GOAL SCORERS ----------
        ev_raw = get_events(fx["fid"], force=force)
        if ev_raw:
            ev_df = events_df_from(ev_raw)
            if not ev_df.empty:
                goals_df = ev_df[ev_df["type"] == "Goal"].copy()

                def scorers(team_name: str) -> str:
                    tdf = goals_df[goals_df["team"] == team_name]
                    if tdf.empty:
                        return "None"
                    out = []
                    for _, row in tdf.iterrows():
                        player = row.get("player") or "Unknown"
                        minute = row.get("minute")
                        detail = (row.get("detail") or "").strip()
                        tag = ""
                        if minute is not None:
                            tag = f"{minute}'"
                        if detail and detail.lower() not in ("normal goal",):
                            tag = f"{tag} ({detail})" if tag else f"({detail})"
                        out.append(f"{player} {tag}".strip())
                    return "; ".join(out)

                lines.append("**Goal Scorers**")
                lines.append("| Team | Goal Scorers |")
                lines.append("|------|--------------|")
                lines.append(f"| {home} | {scorers(home)} |")
                lines.append(f"| {away} | {scorers(away)} |")
                lines.append("")
            else:
                lines.append("Goal data unavailable for this match.\n")
        else:
            lines.append("Goal data unavailable for this match.\n")

        lines.append("---\n")  

    return "\n".join(lines)





