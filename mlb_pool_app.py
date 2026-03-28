"""
MLB 13-Run Pool Tracker -- Streamlit Dashboard
================================================
Interactive web dashboard for tracking the 13-run pool.

Run with:
    pip install streamlit requests plotly pandas
    streamlit run mlb_pool_app.py
"""

import json
import time
import random
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://statsapi.mlb.com/api/v1"
SCHEDULE_ENDPOINT = f"{BASE_URL}/schedule"
REQUEST_TIMEOUT = 15
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2

OPENING_DAY = "2026-03-26"
SEASON_GAMES = 162
TARGET_RUNS = list(range(0, 14))
MC_SIMULATIONS = 10_000

ALL_TEAMS = [
    "Arizona Diamondbacks", "Atlanta Braves", "Baltimore Orioles",
    "Boston Red Sox", "Chicago Cubs", "Chicago White Sox",
    "Cincinnati Reds", "Cleveland Guardians", "Colorado Rockies",
    "Detroit Tigers", "Houston Astros", "Kansas City Royals",
    "Los Angeles Angels", "Los Angeles Dodgers", "Miami Marlins",
    "Milwaukee Brewers", "Minnesota Twins", "New York Mets",
    "New York Yankees", "Oakland Athletics", "Philadelphia Phillies",
    "Pittsburgh Pirates", "San Diego Padres", "San Francisco Giants",
    "Seattle Mariners", "St. Louis Cardinals", "Tampa Bay Rays",
    "Texas Rangers", "Toronto Blue Jays", "Washington Nationals",
]

# City abbreviations (used internally, not for display)
SHORT_NAMES = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
}

# Team nicknames for chart labels (e.g. "Brewers" instead of "MIL")
TEAM_NICKNAMES = {
    "Arizona Diamondbacks": "D-backs", "Atlanta Braves": "Braves",
    "Baltimore Orioles": "Orioles", "Boston Red Sox": "Red Sox",
    "Chicago Cubs": "Cubs", "Chicago White Sox": "White Sox",
    "Cincinnati Reds": "Reds", "Cleveland Guardians": "Guardians",
    "Colorado Rockies": "Rockies", "Detroit Tigers": "Tigers",
    "Houston Astros": "Astros", "Kansas City Royals": "Royals",
    "Los Angeles Angels": "Angels", "Los Angeles Dodgers": "Dodgers",
    "Miami Marlins": "Marlins", "Milwaukee Brewers": "Brewers",
    "Minnesota Twins": "Twins", "New York Mets": "Mets",
    "New York Yankees": "Yankees", "Oakland Athletics": "Athletics",
    "Philadelphia Phillies": "Phillies", "Pittsburgh Pirates": "Pirates",
    "San Diego Padres": "Padres", "San Francisco Giants": "Giants",
    "Seattle Mariners": "Mariners", "St. Louis Cardinals": "Cardinals",
    "Tampa Bay Rays": "Rays", "Texas Rangers": "Rangers",
    "Toronto Blue Jays": "Blue Jays", "Washington Nationals": "Nationals",
}

HISTORICAL_RUN_FREQ = {
    0: 0.070, 1: 0.098, 2: 0.123, 3: 0.138, 4: 0.136,
    5: 0.118, 6: 0.095, 7: 0.072, 8: 0.051, 9: 0.036,
    10: 0.024, 11: 0.015, 12: 0.010, 13: 0.006,
}


def nickname(full_name: str) -> str:
    """Return the team nickname (e.g. 'Brewers') or fall back to the full name."""
    return TEAM_NICKNAMES.get(full_name, full_name)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class TeamProgress:
    def __init__(self, team_name: str, participant: str = "-"):
        self.team_name = team_name
        self.participant = participant
        self.games_played = 0
        self.achieved = {}
        self.run_histogram = defaultdict(int)

    @property
    def remaining(self) -> list[int]:
        return [r for r in TARGET_RUNS if r not in self.achieved]

    @property
    def completed(self) -> bool:
        return len(self.achieved) == len(TARGET_RUNS)

    @property
    def scratched_count(self) -> int:
        return len(self.achieved)

    def record_game(self, runs: int, date: str):
        self.games_played += 1
        self.run_histogram[runs] += 1
        if runs in TARGET_RUNS and runs not in self.achieved:
            self.achieved[runs] = date

    def completion_date(self) -> Optional[str]:
        if not self.completed:
            return None
        return max(self.achieved.values())

    def tiebreaker_key(self):
        achieved_dates = []
        for r in range(13, -1, -1):
            achieved_dates.append(self.achieved.get(r, "9999-99-99"))
        return (-self.scratched_count, self.games_played, achieved_dates)


# ---------------------------------------------------------------------------
# API Fetching
# ---------------------------------------------------------------------------

def _request_with_retry(url, params=None):
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY)
    return None


def fetch_scores_range(start_date: str, end_date: str) -> list[dict]:
    """Fetch all completed regular-season MLB games in a date range."""
    params = {
        "sportId": 1,
        "startDate": start_date,
        "endDate": end_date,
        "hydrate": "team,linescore",
        "gameType": "R",
        "language": "en",
    }
    response = _request_with_retry(SCHEDULE_ENDPOINT, params=params)
    if response is None:
        return []

    data = response.json()
    games = []
    seen_game_ids = set()
    for date_entry in data.get("dates", []):
        game_date = date_entry.get("date", "")
        for game in date_entry.get("games", []):
            parsed = _parse_game(game, game_date)
            if parsed and parsed["game_id"] not in seen_game_ids:
                seen_game_ids.add(parsed["game_id"])
                games.append(parsed)
    return games


def _parse_game(game: dict, game_date: str) -> Optional[dict]:
    status_code = game.get("status", {}).get("statusCode", "")
    if status_code not in ("F", "FR", "FT"):
        return None

    teams = game.get("teams", {})
    away_score = teams.get("away", {}).get("score")
    home_score = teams.get("home", {}).get("score")
    if away_score is None or home_score is None:
        return None

    official_date = game.get("officialDate", game_date)
    away_name = " ".join(teams["away"]["team"]["name"].split())
    home_name = " ".join(teams["home"]["team"]["name"].split())

    return {
        "game_id": game.get("gamePk"),
        "game_type": game.get("gameType", "R"),
        "date": official_date,
        "away_team": away_name,
        "away_score": int(away_score),
        "home_team": home_name,
        "home_score": int(home_score),
    }


def fetch_season_scores(start_date: str, end_date: str, progress_bar=None):
    """Fetch all games in chunks of 30 days."""
    all_games = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - current).days + 1
    chunk_size = 30

    elapsed = 0
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_size - 1), end)
        chunk_start_str = current.strftime("%Y-%m-%d")
        chunk_end_str = chunk_end.strftime("%Y-%m-%d")

        if progress_bar:
            pct = min(elapsed / total_days, 1.0)
            progress_bar.progress(pct, text=f"Fetching {chunk_start_str} to {chunk_end_str}...")

        games = fetch_scores_range(chunk_start_str, chunk_end_str)
        all_games.extend(games)

        elapsed += (chunk_end - current).days + 1
        current = chunk_end + timedelta(days=1)
        time.sleep(0.3)

    if progress_bar:
        progress_bar.progress(1.0, text="Done!")

    return all_games


# ---------------------------------------------------------------------------
# Pool Logic
# ---------------------------------------------------------------------------

def discover_team_names(games: list[dict]) -> set[str]:
    """Extract all unique team names that appear in the fetched game data."""
    names = set()
    for game in games:
        names.add(game["away_team"])
        names.add(game["home_team"])
    return names


def build_team_progress(games, participants=None):
    participants = participants or {}
    teams = {}

    api_names = discover_team_names(games)

    for name in api_names:
        teams[name] = TeamProgress(name, participants.get(name, "-"))
        if name not in SHORT_NAMES:
            SHORT_NAMES[name] = "".join(
                word[0] for word in name.split() if word[0].isupper()
            )[:3].upper()
        if name not in TEAM_NICKNAMES:
            # Use last word of name as fallback nickname
            TEAM_NICKNAMES[name] = name.split()[-1]

    for name in ALL_TEAMS:
        if name not in teams:
            teams[name] = TeamProgress(name, participants.get(name, "-"))

    for game in sorted(games, key=lambda g: g["date"]):
        teams[game["away_team"]].record_game(game["away_score"], game["date"])
        teams[game["home_team"]].record_game(game["home_score"], game["date"])

    return teams


def get_leaderboard(teams):
    return sorted(teams.values(), key=lambda t: t.tiebreaker_key())


def compute_observed_frequencies(teams):
    total_games = 0
    run_counts = defaultdict(int)
    for team in teams.values():
        total_games += team.games_played
        for runs, count in team.run_histogram.items():
            run_counts[runs] += count
    if total_games == 0:
        return HISTORICAL_RUN_FREQ.copy()
    return {r: run_counts[r] / total_games for r in range(0, 20)}


def blended_frequencies(observed, total_team_games, blend_games=50):
    if total_team_games == 0:
        return HISTORICAL_RUN_FREQ.copy()
    obs_weight = total_team_games / (total_team_games + blend_games)
    hist_weight = 1.0 - obs_weight
    return {
        r: obs_weight * observed.get(r, 0.0) + hist_weight * HISTORICAL_RUN_FREQ.get(r, 0.001)
        for r in TARGET_RUNS
    }


def monte_carlo_expected_games(remaining, freq, games_played=0, n_simulations=MC_SIMULATIONS):
    if not remaining:
        return {
            "expected_games": 0, "median_games": 0,
            "p25": 0, "p75": 0, "p90": 0,
            "completion_prob_30": 1.0, "completion_prob_season": 1.0,
        }

    outcomes = list(range(0, 25))
    probs = [freq.get(r, 0.001 if r <= 13 else 0.002) for r in outcomes]
    total_p = sum(probs)
    probs = [p / total_p for p in probs]
    games_left_in_season = max(SEASON_GAMES - games_played, 0)

    results = []
    for _ in range(n_simulations):
        needed = set(remaining)
        games = 0
        while needed and games < 1000:
            games += 1
            r = random.choices(outcomes, weights=probs, k=1)[0]
            needed.discard(r)
        results.append(games)

    results.sort()
    n = len(results)
    return {
        "expected_games": sum(results) / n,
        "median_games": results[n // 2],
        "p25": results[n // 4],
        "p75": results[3 * n // 4],
        "p90": results[int(n * 0.9)],
        "completion_prob_30": sum(1 for g in results if g <= 30) / n,
        "completion_prob_season": sum(1 for g in results if g <= games_left_in_season) / n,
    }


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="MLB 13-Run Pool Tracker",
        page_icon="⚾",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Custom CSS ---
    st.markdown("""
    <style>
    .scratched {
        background-color: #2563eb;
        color: white;
        padding: 8px 10px;
        border-radius: 6px;
        font-weight: bold;
        text-align: center;
        font-size: 0.9rem;
    }
    .needed {
        background-color: #fef3c7;
        color: #92400e;
        padding: 8px 10px;
        border-radius: 6px;
        text-align: center;
        font-size: 0.9rem;
        border: 1px solid #fde68a;
    }
    .winner-banner {
        background: linear-gradient(135deg, #f59e0b, #ef4444);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 16px;
    }
    div[data-testid="stDataFrame"] table {
        font-size: 0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Title ---
    st.title("MLB 13-Run Pool Tracker")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")

        st.subheader("Date Range")
        start_date = st.date_input(
            "Season start",
            value=datetime.strptime(OPENING_DAY, "%Y-%m-%d").date(),
            help="Opening Day 2026"
        )
        end_date = st.date_input(
            "Through date",
            value=datetime.now().date(),
            help="Fetch scores through this date"
        )

        st.divider()

        st.subheader("Participant Assignments")
        st.caption("Map pool members to teams. Leave blank for unassigned.")

        if "participants" not in st.session_state:
            st.session_state.participants = {}

        uploaded = st.file_uploader(
            "Upload assignments (JSON)", type=["json"],
            help='Format: {"New York Yankees": "John", ...}'
        )
        if uploaded:
            try:
                st.session_state.participants = json.load(uploaded)
                st.success(f"Loaded {len(st.session_state.participants)} assignments")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

        all_known_teams = sorted(set(ALL_TEAMS) | set(SHORT_NAMES.keys()))
        with st.expander("Edit assignments manually"):
            for team in all_known_teams:
                val = st.text_input(
                    nickname(team),
                    value=st.session_state.participants.get(team, ""),
                    key=f"p_{team}",
                    label_visibility="visible",
                )
                if val.strip():
                    st.session_state.participants[team] = val.strip()
                elif team in st.session_state.participants:
                    del st.session_state.participants[team]

        st.divider()

        st.subheader("Simulations")
        sim_count = st.select_slider(
            "Monte Carlo runs",
            options=[1000, 5000, 10000, 20000, 50000],
            value=10000,
            help="Higher values produce more precise estimates but take longer"
        )

        fetch_btn = st.button("Fetch Scores", type="primary", use_container_width=True)

    # --- Main content ---
    if fetch_btn or "games" not in st.session_state:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        progress_bar = st.progress(0, text="Fetching scores from MLB API...")
        games = fetch_season_scores(start_str, end_str, progress_bar)
        progress_bar.empty()

        st.session_state.games = games
        st.session_state.end_date = end_str

    games = st.session_state.get("games", [])

    if not games:
        st.warning(
            "No completed games found. The season may not have started yet, "
            "or there may be a connectivity issue with the MLB API."
        )
        st.info("The 2026 MLB season begins March 26, 2026.")
        return

    # Build all derived data
    participants = st.session_state.get("participants", {})
    teams = build_team_progress(games, participants)
    board = get_leaderboard(teams)

    observed = compute_observed_frequencies(teams)
    avg_gp = sum(t.games_played for t in teams.values()) / len(teams)
    freq = blended_frequencies(observed, int(avg_gp))

    end_str = st.session_state.get("end_date", "-")

    # Diagnostic: surface any team names from the API not in the hardcoded list
    api_names = discover_team_names(games)
    unknown_names = api_names - set(ALL_TEAMS)
    if unknown_names:
        with st.sidebar:
            st.divider()
            st.subheader("Team Name Alerts")
            st.warning(
                f"The MLB API returned team name(s) not in the built-in list: "
                f"{', '.join(sorted(unknown_names))}. "
                f"The app tracks these teams automatically, but you may want to update "
                f"the ALL_TEAMS list and TEAM_NICKNAMES dict in the source code."
            )

    # --- Top metrics ---
    leader = board[0]
    total_games = len(games)
    most_scratched = leader.scratched_count
    completed_teams = sum(1 for t in board if t.completed)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Completed Games", f"{total_games:,}",
        help="Total regular-season MLB games with final scores recorded so far"
    )
    col2.metric(
        "Pool Leader", nickname(leader.team_name),
        help=(
            "The app ranks teams by: (1) most run totals scratched off, "
            "(2) fewest games played as a tiebreaker, then "
            "(3) which team scratched off the highest run total earliest"
        )
    )
    col3.metric(
        "Most Scratched Off", f"{most_scratched}/14",
        help="How many of the 14 run totals (0 through 13) the leading team has achieved"
    )
    col4.metric(
        "Teams Finished", f"{completed_teams}/30",
        help="Number of teams that have scratched off all 14 run totals"
    )

    # Winner banner
    winners = [t for t in board if t.completed]
    if winners:
        w = winners[0]
        st.markdown(
            f'<div class="winner-banner">WINNER: {w.team_name} '
            f'({w.participant}), completed on {w.completion_date()} '
            f'in {w.games_played} of {SEASON_GAMES} games</div>',
            unsafe_allow_html=True
        )

    st.caption(f"Data through: {end_str}")

    # --- Tabs ---
    tab_board, tab_grid, tab_detail, tab_odds, tab_race, tab_debug = st.tabs([
        "Leaderboard",
        "Scratch-Off Grid",
        "Team Detail",
        "Run Probabilities",
        "Race Projections",
        "Raw Data",
    ])

    # ========================== TAB 1: LEADERBOARD ==========================
    with tab_board:
        rows = []
        for rank, team in enumerate(board, 1):
            remaining = team.remaining
            games_left = SEASON_GAMES - team.games_played
            rows.append({
                "Rank": rank,
                "Team": nickname(team.team_name),
                "Owner": team.participant,
                "Scratched": team.scratched_count,
                "Games Played": f"{team.games_played} of {SEASON_GAMES}",
                "Run Totals Still Needed": ", ".join(str(r) for r in remaining) if remaining else "COMPLETE",
                "Still Need": len(remaining),
            })

        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Scratched": st.column_config.ProgressColumn(
                    "Progress", min_value=0, max_value=14, format="%d/14"
                ),
            },
        )

    # ======================== TAB 2: SCRATCH-OFF GRID ========================
    with tab_grid:
        st.subheader("Scratch-Off Grid")
        st.caption("Blue = scratched off, yellow = still needed. Hover for details.")

        grid_teams = [nickname(t.team_name) for t in board]
        grid_data = []
        hover_data = []
        for team in board:
            row = []
            hover_row = []
            for r in TARGET_RUNS:
                if r in team.achieved:
                    row.append(1)
                    hover_row.append(f"{team.team_name}<br>Runs: {r}<br>Scratched: {team.achieved[r]}")
                else:
                    row.append(0)
                    hover_row.append(f"{team.team_name}<br>Runs: {r}<br>Not yet")
            grid_data.append(row)
            hover_data.append(hover_row)

        fig_grid = go.Figure(data=go.Heatmap(
            z=grid_data,
            x=[str(r) for r in TARGET_RUNS],
            y=grid_teams,
            text=hover_data,
            hoverinfo="text",
            colorscale=[[0, "#fef3c7"], [1, "#2563eb"]],
            showscale=False,
            xgap=3,
            ygap=3,
        ))

        fig_grid.update_layout(
            xaxis_title="Run Total",
            yaxis=dict(autorange="reversed"),
            height=max(500, len(board) * 24),
            margin=dict(l=10, r=10, t=30, b=40),
            font=dict(size=12),
        )

        st.plotly_chart(fig_grid, use_container_width=True)

    # ======================== TAB 3: TEAM DETAIL ========================
    with tab_detail:
        selected_team_name = st.selectbox(
            "Select a team",
            [t.team_name for t in board],
            format_func=lambda x: f"{nickname(x)} ({participants.get(x, '-')})"
        )

        team = teams[selected_team_name]
        games_left = SEASON_GAMES - team.games_played

        st.subheader(team.team_name)
        if team.participant != "-":
            st.caption(f"Owner: {team.participant}")

        col_gp, col_done, col_left = st.columns(3)
        col_gp.metric(
            "Games Played", f"{team.games_played} of {SEASON_GAMES}",
            help=f"This team has {games_left} games remaining in the {SEASON_GAMES}-game season"
        )
        col_done.metric(
            "Run Totals Scratched Off", f"{team.scratched_count}/14",
            help="Out of 14 possible (0 through 13)"
        )
        col_left.metric(
            "Run Totals Still Needed", len(team.remaining),
            help="How many of the 14 run totals this team has not yet scored"
        )

        # Visual scratch card
        st.caption("Blue = scratched off, yellow = still needed")
        cols = st.columns(14)
        for i, r in enumerate(TARGET_RUNS):
            with cols[i]:
                if r in team.achieved:
                    st.markdown(f'<div class="scratched">{r}</div>', unsafe_allow_html=True)
                    st.caption(team.achieved[r][5:])  # show MM-DD
                else:
                    st.markdown(f'<div class="needed">{r}</div>', unsafe_allow_html=True)
                    st.caption("-")

        # Achievement timeline
        if team.achieved:
            st.subheader("Achievement Timeline")
            timeline_data = sorted(team.achieved.items(), key=lambda x: x[1])
            timeline_df = pd.DataFrame(
                [{"Run Total": r, "Date": d} for r, d in timeline_data]
            )
            st.dataframe(timeline_df, use_container_width=True, hide_index=True)

        # Run histogram
        if team.games_played > 0:
            st.subheader("Scoring Distribution")
            hist_data = []
            for r in range(0, max(team.run_histogram.keys()) + 1 if team.run_histogram else 14):
                count = team.run_histogram.get(r, 0)
                hist_data.append({"Runs": r, "Games": count})

            hist_df = pd.DataFrame(hist_data)
            fig_hist = px.bar(
                hist_df, x="Runs", y="Games",
                color_discrete_sequence=["#3b82f6"],
                title=f"How often the {nickname(team.team_name)} score N runs"
            )
            fig_hist.update_layout(
                xaxis=dict(dtick=1),
                margin=dict(t=40, b=30),
                height=300,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Monte Carlo projection
        if team.remaining and team.games_played > 0:
            st.subheader("Projection")
            with st.spinner("Running simulation..."):
                sim = monte_carlo_expected_games(
                    team.remaining, freq,
                    games_played=team.games_played,
                    n_simulations=sim_count
                )

            pcol1, pcol2, pcol3 = st.columns(3)
            pcol1.metric(
                "Est. Games to Complete",
                f"~{sim['expected_games']:.0f} of {SEASON_GAMES}",
                help=(
                    f"The simulation estimates this team needs about {sim['expected_games']:.0f} total games "
                    f"to scratch off all remaining run totals. Each team plays {SEASON_GAMES} games per season."
                )
            )
            pcol2.metric(
                "Median Games to Complete",
                f"{sim['median_games']:.0f} of {SEASON_GAMES}",
                help="Half of simulated seasons finished faster than this, half finished slower"
            )
            pcol3.metric(
                f"Chance of Finishing ({games_left} of {SEASON_GAMES} games left)",
                f"{sim['completion_prob_season']*100:.1f}%",
                help=(
                    f"In the simulation, this is the percentage of seasons where the team "
                    f"scratched off all remaining run totals within the {games_left} games left"
                )
            )

            hardest = max(team.remaining, key=lambda r: 1.0 / freq.get(r, 0.001))
            hardest_p = freq.get(hardest, 0.001)
            st.info(
                f"Hardest remaining: {hardest} runs. "
                f"Teams score exactly {hardest} runs in about {hardest_p*100:.1f}% of games, "
                f"or roughly once every {1/hardest_p:.0f} games."
            )

    # ====================== TAB 4: RUN PROBABILITIES ======================
    with tab_odds:
        st.subheader("Run-Scoring Probability Analysis")
        st.caption("Historical MLB averages blended with this season's observed data")

        prob_rows = []
        for r in TARGET_RUNS:
            hist = HISTORICAL_RUN_FREQ.get(r, 0.0)
            obs = observed.get(r, 0.0)
            blend = freq.get(r, 0.0)
            one_in = 1 / blend if blend > 0 else float("inf")
            prob_rows.append({
                "Runs Scored": r,
                "Historical Avg (2014-24)": f"{hist:.1%}",
                "This Season": f"{obs:.1%}" if avg_gp > 0 else "-",
                "Blended Estimate": f"{blend:.1%}",
                "Avg. Games Between Occurrences": f"{one_in:.1f}",
            })

        prob_df = pd.DataFrame(prob_rows)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

        # Bar chart comparing historical vs observed
        chart_df = pd.DataFrame({
            "Runs": TARGET_RUNS * 2,
            "Probability": (
                [HISTORICAL_RUN_FREQ.get(r, 0) for r in TARGET_RUNS] +
                [freq.get(r, 0) for r in TARGET_RUNS]
            ),
            "Source": (["Historical"] * 14) + (["Blended (this season)"] * 14),
        })

        fig_prob = px.bar(
            chart_df, x="Runs", y="Probability", color="Source",
            barmode="group",
            color_discrete_map={"Historical": "#64748b", "Blended (this season)": "#3b82f6"},
            title="Probability of Scoring Exactly N Runs"
        )
        fig_prob.update_layout(
            xaxis=dict(dtick=1),
            yaxis=dict(tickformat=".1%"),
            height=400,
            margin=dict(t=40),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        # Insight callouts
        p13 = freq.get(13, 0.006)
        p0 = freq.get(0, 0.07)
        p_sweet = (freq.get(3, 0) + freq.get(4, 0) + freq.get(5, 0)) * 100
        p_tail = sum(freq.get(r, 0) for r in range(10, 14)) * 100
        st.markdown(
            f"Teams score exactly 13 runs roughly once every {1/p13:.0f} games "
            f"({p13*100:.1f}% per game), which makes it the bottleneck for most pool entries. "
            f"Getting shut out (0 runs) is also relatively uncommon at about once every "
            f"{1/p0:.0f} games. The sweet spot of 3 to 5 runs accounts for about "
            f"{p_sweet:.0f}% of all games. Scoring 10 or more runs is rare; those totals "
            f"combine for only about {p_tail:.1f}% of games."
        )

    # ====================== TAB 5: RACE PROJECTIONS ======================
    with tab_race:
        st.subheader("Race to Finish: Projected Completion")

        with st.spinner("Running simulations for all 30 teams..."):
            race_rows = []
            for team in board:
                games_left = SEASON_GAMES - team.games_played
                if team.completed:
                    race_rows.append({
                        "Team": nickname(team.team_name),
                        "Owner": team.participant,
                        "Scratched": team.scratched_count,
                        "Games Played": f"{team.games_played} of {SEASON_GAMES}",
                        "Est. Games to Complete": 0,
                        "Chance of Finishing by Season End": 100.0,
                        "Status": f"DONE ({team.completion_date()})",
                    })
                elif team.games_played > 0:
                    sim = monte_carlo_expected_games(
                        team.remaining, freq,
                        games_played=team.games_played,
                        n_simulations=sim_count
                    )
                    race_rows.append({
                        "Team": nickname(team.team_name),
                        "Owner": team.participant,
                        "Scratched": team.scratched_count,
                        "Games Played": f"{team.games_played} of {SEASON_GAMES}",
                        "Est. Games to Complete": round(sim["expected_games"]),
                        "Chance of Finishing by Season End": round(sim["completion_prob_season"] * 100, 1),
                        "Status": f"Need {len(team.remaining)} more",
                    })

        race_df = pd.DataFrame(race_rows)
        if not race_df.empty:
            race_df = race_df.sort_values("Est. Games to Complete")

            st.caption(
                "Est. Games to Complete: the average number of games the simulation "
                "needed to scratch off all remaining run totals. "
                "Chance of Finishing by Season End: the percentage of simulated seasons "
                f"where the team completed the pool within the {SEASON_GAMES}-game season."
            )

            st.dataframe(
                race_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Chance of Finishing by Season End": st.column_config.ProgressColumn(
                        "Chance of Finishing by Season End", min_value=0, max_value=100, format="%.1f%%"
                    ),
                    "Scratched": st.column_config.ProgressColumn(
                        "Progress", min_value=0, max_value=14, format="%d/14"
                    ),
                },
            )

            # Chart: expected games remaining
            fig_race = px.bar(
                race_df.sort_values("Est. Games to Complete"),
                x="Team", y="Est. Games to Complete",
                color="Chance of Finishing by Season End",
                color_continuous_scale="RdYlGn",
                title=f"Estimated Games to Complete Pool (out of {SEASON_GAMES})",
                range_color=[0, 100],
            )
            fig_race.update_layout(
                xaxis_title="Team",
                yaxis_title=f"Est. Games to Complete (of {SEASON_GAMES})",
                height=400,
                margin=dict(t=40),
            )
            # Add a reference line at 162
            fig_race.add_hline(
                y=SEASON_GAMES, line_dash="dot", line_color="#ef4444",
                annotation_text=f"{SEASON_GAMES}-game season",
                annotation_position="top right",
            )
            st.plotly_chart(fig_race, use_container_width=True)

    # ======================== TAB 6: RAW DATA / DEBUG ========================
    with tab_debug:
        st.subheader("Raw Data and Diagnostics")
        st.caption(
            "Use this tab to verify the data the app uses. "
            "If a team or game appears to be missing, check here first."
        )

        api_names = discover_team_names(games)
        st.markdown("Team names from the API (the exact strings the MLB API returned):")
        api_names_sorted = sorted(api_names)
        name_rows = []
        for name in api_names_sorted:
            in_hardcoded = "Yes" if name in ALL_TEAMS else "NO, not in built-in list"
            gp = teams[name].games_played if name in teams else 0
            name_rows.append({
                "API Team Name": name,
                "Nickname": nickname(name),
                "In Built-in List?": in_hardcoded,
                "Games Played": gp,
            })
        st.dataframe(pd.DataFrame(name_rows), use_container_width=True, hide_index=True)

        zero_game_teams = [t.team_name for t in teams.values() if t.games_played == 0]
        if zero_game_teams:
            st.warning(
                f"Teams with 0 games: {', '.join(sorted(zero_game_teams))}. "
                f"If the season has started, the API may use a different name for these teams "
                f"than what the built-in list expects."
            )

        st.divider()

        st.markdown("All fetched games (most recent first):")
        game_log_rows = []
        for g in sorted(games, key=lambda x: x["date"], reverse=True):
            game_log_rows.append({
                "Date": g["date"],
                "Game ID": g["game_id"],
                "Type": g.get("game_type", "?"),
                "Away Team": g["away_team"],
                "Away Score": g["away_score"],
                "Home Team": g["home_team"],
                "Home Score": g["home_score"],
            })
        st.dataframe(
            pd.DataFrame(game_log_rows),
            use_container_width=True,
            hide_index=True,
            height=400,
        )

        st.caption(f"Total games loaded: {len(games)}")

        st.divider()
        search_team = st.text_input("Search for a team name in raw data", placeholder="e.g. Brewers")
        if search_team:
            matches = [
                g for g in games
                if search_team.lower() in g["away_team"].lower()
                or search_team.lower() in g["home_team"].lower()
            ]
            if matches:
                st.success(f"Found {len(matches)} game(s) matching '{search_team}':")
                match_rows = []
                for g in matches:
                    match_rows.append({
                        "Date": g["date"],
                        "Game ID": g["game_id"],
                        "Type": g.get("game_type", "?"),
                        "Away": f"{g['away_team']} ({g['away_score']})",
                        "Home": f"{g['home_team']} ({g['home_score']})",
                    })
                st.dataframe(pd.DataFrame(match_rows), use_container_width=True, hide_index=True)
            else:
                st.error(
                    f"No games found matching '{search_team}'. "
                    f"The API may use a different team name. "
                    f"Check the team name list above."
                )

    # --- Footer ---
    st.divider()
    st.caption(
        "The MLB Stats API (statsapi.mlb.com) provides all game data. "
        "The app blends historical run frequencies (2014 to 2024) with observed "
        "season data to estimate probabilities, then runs Monte Carlo simulations "
        "to project each team's chances of completing the pool."
    )


if __name__ == "__main__":
    main()
