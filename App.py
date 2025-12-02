# app.py
import os, re
import streamlit as st

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate


from Football_Agent import head_to_head_report  


class H2HArgs(BaseModel):
    """Arguments for head_to_head_report."""
    team1: str = Field(..., description="Team name, e.g., 'Chelsea'")
    team2: str = Field(..., description="Opponent team name, e.g., 'Liverpool'")
    season: int = Field(..., description="Four-digit season year, e.g., 2022")
    league: int = Field(39, description="League ID (39 = EPL)")
    force: bool = Field(False, description="Bypass local cache if True")
    use_llm: bool = Field(True, description="Enable LLM summarization inside the tool")
    save_dir: str = Field("charts", description="Directory to save charts")

h2h_tool = StructuredTool.from_function(
    func=head_to_head_report,
    name="head_to_head_report",
    description=(
        "Generate the head-to-head report and save charts for two teams in a given season. "
        "Understands EPL (league=39). Include season year in the user's prompt."
    ),
    args_schema=H2HArgs,
)

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        (
        "You are a football statistics assistant.\n\n"
        "CRITICAL RULES:\n"
        "1. When calling the `head_to_head_report` tool, you MUST return the tool output EXACTLY as it is.\n"
        "2. NEVER rewrite, reformat, rename, or paraphrase match lines.\n"
        "3. The tool already outputs match lines in this required format:\n"
        "   YYYY-MM-DD — HomeTeam X–Y AwayTeam\n"
        "   (em dash between date and teams, en dash between score numbers)\n"
        "4. DO NOT convert dates to natural language (e.g., 'January 21, 2023').\n"
        "5. DO NOT add bullet points, numbering, or any extra words before the match lines.\n"
        "6. The match lines MUST remain at the top of the output, unchanged.\n\n"
        "7. Give Statistics of all the matches between them for that season. \n"
        "8. Give which player scored the goals. \n"
        "After the match lines from the tool output, you may write a short summary in plain English."
        ),
    ),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}"),  
])

agent_runnable = create_tool_calling_agent(
    llm=llm,
    tools=[h2h_tool],
    prompt=prompt_template,
)

agent = AgentExecutor(
    agent=agent_runnable,
    tools=[h2h_tool],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=6,
)

# ---------- Page & Theme ----------
st.set_page_config(page_title="SoccerIQ — Agentic Match Insights", layout="wide")
st.markdown("""
<style>
.main { padding-top: 1rem; }
html, body, [class*="css"]  { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Helvetica Neue", "Arial", "Noto Sans", sans-serif; }
h1 { letter-spacing: 0.2px; }
.small-hint { color: #6b7280; font-size: 0.9rem; margin-top: -0.25rem; }
.match-card {
  border-radius: 16px; padding: 16px 18px;
  background: linear-gradient(180deg, #0b1324 0%, #0e162b 100%);
  border: 1px solid rgba(255,255,255,0.08); color: #ecf2ff;
  box-shadow: 0 6px 18px rgba(0,0,0,0.25); margin-bottom: 12px;
}
.match-top { display:flex; align-items:center; justify-content:space-between; margin-bottom:6px; color:#93a5c8; font-size:0.9rem; }
.match-teams { display:flex; align-items:center; gap:10px; font-weight:600; font-size:1.05rem; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:0.8rem; background:#0f2533; border:1px solid rgba(255,255,255,0.08); color:#a8c7d8; margin-left:6px; }
.stat-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
.pill { background:#0e2230; border:1px solid rgba(255,255,255,0.08); color:#e7f0f6; border-radius:999px; padding:6px 10px; font-size:0.85rem; }
.pill b { color:#7fe1c3; }
.section-title { margin-top: 10px; margin-bottom: 8px; color:#aebad1; font-weight:600; font-size:0.95rem; }
hr { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 14px 0; }
[data-testid="stSidebar"] { background: radial-gradient(600px 300px at 0% 0%, #0f1a2c 0%, #0b1324 60%); }
</style>
""", unsafe_allow_html=True)

st.title("⚽ SoccerIQ — Agentic Match Insights")
st.markdown(
    '<div class="small-hint">Ask in plain English, e.g. '
    '<i>“Give me result and stats between Manchester United vs Liverpool, season 2022”</i></div>',
    unsafe_allow_html=True,
)

# ---------- Prompt ----------
user_query = st.text_input(
    "Prompt",
    value="Give me result and stats between Chelsea vs Liverpool, season 2022",
    placeholder="Give me result and stats between Team A vs Team B, season YYYY"
)

cols = st.columns([1, 1, 2])
with cols[0]:
    show_charts = st.toggle("Show charts", value=False, help="Turn on to render saved charts (optional).")
with cols[1]:
    compact = st.toggle("Compact mode", value=True, help="Smaller text and tighter spacing.")

# ---------- Run ----------
if st.button("Run Analysis", type="primary", use_container_width=True):
    with st.spinner("Agent is working..."):
        # RAW prompt → agent; the tool schema guides argument extraction
        resp = agent.invoke({"input": user_query})
        output_text = resp.get("output", "") if isinstance(resp, dict) else str(resp)

    # Friendly warning if agent claims no fixtures
    if "No fixtures found" in output_text:
        st.warning(
            "No fixtures were found. Double-check the season (e.g., 2022 for EPL 2022–23) and team names. "
            "You can also try phrasing like: “Team A vs Team B, season 2022”."
        )

    # ---------- Render: pretty match cards + stat pills ----------
    st.markdown("### Results")

    # ---------- Raw text (collapsible) ----------
    with st.expander("See full agent output", expanded=not compact):
        st.markdown(output_text)
