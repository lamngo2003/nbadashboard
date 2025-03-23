import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time
import json
import os

# Set page configuration
st.set_page_config(
    page_title="NBA Performance Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1D428A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #C8102E;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1D428A;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# NBA API Headers - Required to mimic a browser request
nba_headers = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Referer': 'https://stats.nba.com/',
    'Origin': 'https://stats.nba.com'
}

# Current NBA season
current_season = "2023-24"

# Cache function to avoid repeated API calls
@st.cache_data(ttl=3600*24)  # Cache for 24 hours
def fetch_nba_data():
    """Fetch NBA data from the official NBA Stats API"""
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Check if we have cached data files and they're recent
    current_date = datetime.now().strftime('%Y-%m-%d')
    cache_file = f'data/nba_data_{current_date}.json'
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # If no cache or it's outdated, fetch new data
    data = {}
    
    # Fetch teams with error handling
    try:
        teams_url = "https://stats.nba.com/stats/leaguestandingsv3?LeagueID=00&Season={}&SeasonType=Regular%20Season".format(current_season)
        teams_response = requests.get(teams_url, headers=nba_headers, timeout=10)
        teams_response.raise_for_status()
        
        teams_data = teams_response.json()
        if 'resultSets' in teams_data:
            # Extract team data from the response
            standings = teams_data['resultSets'][0]
            headers = standings['headers']
            rows = standings['rowSet']
            
            teams = []
            for row in rows:
                team = {
                    "id": row[0],  # TeamID
                    "city": row[5],  # TeamCity
                    "name": row[6],  # TeamName
                    "full_name": f"{row[5]} {row[6]}",  # TeamCity + TeamName
                    "abbreviation": row[4],  # TeamAbbreviation
                    "conference": row[7],  # Conference
                    "division": row[8],  # Division
                    "wins": row[13],  # WINS
                    "losses": row[14],  # LOSSES
                    "win_pct": row[15],  # WIN_PCT
                    "points_pg": row[23],  # PTS
                    "opp_points_pg": row[24],  # OPP_PTS
                }
                teams.append(team)
            
            data['teams'] = teams
        else:
            st.error("Teams API response missing expected data structure")
            data['teams'] = []
            
        time.sleep(1)  # Respect rate limits
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching teams data: {e}")
        data['teams'] = []
    except ValueError as e:
        st.error(f"Error parsing JSON from API: {e}")
        data['teams'] = []
    
    # Fetch players with error handling
    try:
        players_url = "https://stats.nba.com/stats/leaguedashplayerstats?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season={}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight=".format(current_season)
        players_response = requests.get(players_url, headers=nba_headers, timeout=10)
        players_response.raise_for_status()
        
        players_data = players_response.json()
        if 'resultSets' in players_data:
            # Extract player data from the response
            player_stats = players_data['resultSets'][0]
            headers = player_stats['headers']
            rows = player_stats['rowSet']
            
            # Create a dictionary to map header names to indices
            header_indices = {header: idx for idx, header in enumerate(headers)}
            
            players = []
            for row in rows:
                player = {
                    "id": row[header_indices["PLAYER_ID"]],
                    "name": row[header_indices["PLAYER_NAME"]],
                    "team_id": row[header_indices["TEAM_ID"]],
                    "team_abbreviation": row[header_indices["TEAM_ABBREVIATION"]],
                    "age": row[header_indices["AGE"]],
                    "gp": row[header_indices["GP"]],
                    "pts": row[header_indices["PTS"]],
                    "reb": row[header_indices["REB"]],
                    "ast": row[header_indices["AST"]],
                    "stl": row[header_indices["STL"]],
                    "blk": row[header_indices["BLK"]],
                    "fg_pct": row[header_indices["FG_PCT"]],
                    "fg3_pct": row[header_indices["FG3_PCT"]],
                    "ft_pct": row[header_indices["FT_PCT"]],
                    "min": row[header_indices["MIN"]],
                    "tov": row[header_indices["TOV"]],
                }
                players.append(player)
            
            data['players'] = players
        else:
            st.error("Players API response missing expected data structure")
            data['players'] = []
            
        time.sleep(1)  # Respect rate limits
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching players data: {e}")
        data['players'] = []
    except ValueError as e:
        st.error(f"Error parsing JSON from API: {e}")
        data['players'] = []
    
    # Fetch recent games with error handling
    try:
        # Get the current date and date 30 days ago
        today = datetime.now()
        thirty_days_ago = today - timedelta(days=30)
        
        # Format dates for the API
        from_date = thirty_days_ago.strftime('%m/%d/%Y')
        to_date = today.strftime('%m/%d/%Y')
        
        games_url = f"https://stats.nba.com/stats/leaguegamelog?Counter=1000&DateFrom={from_date}&DateTo={to_date}&Direction=DESC&LeagueID=00&PlayerOrTeam=T&Season={current_season}&SeasonType=Regular+Season&Sorter=DATE"
        games_response = requests.get(games_url, headers=nba_headers, timeout=10)
        games_response.raise_for_status()
        
        games_data = games_response.json()
        if 'resultSets' in games_data:
            # Extract game data from the response
            game_logs = games_data['resultSets'][0]
            headers = game_logs['headers']
            rows = game_logs['rowSet']
            
            # Create a dictionary to map header names to indices
            header_indices = {header: idx for idx, header in enumerate(headers)}
            
            # Process game logs to create a list of games
            games_dict = {}  # Use a dictionary to avoid duplicates (each game appears twice, once for each team)
            
            for row in rows:
                game_id = row[header_indices["GAME_ID"]]
                team_id = row[header_indices["TEAM_ID"]]
                team_abbreviation = row[header_indices["TEAM_ABBREVIATION"]]
                team_name = row[header_indices["TEAM_NAME"]]
                game_date = row[header_indices["GAME_DATE"]]
                matchup = row[header_indices["MATCHUP"]]
                wl = row[header_indices["WL"]]
                pts = row[header_indices["PTS"]]
                
                # Parse matchup to determine home/away and opponent
                is_home = "@" not in matchup
                opponent_abbr = matchup.split()[-1]
                
                # If this game is already in our dictionary
                if game_id in games_dict:
                    existing_game = games_dict[game_id]
                    
                    # If current team is home, update home team info
                    if is_home:
                        existing_game["home_team_id"] = team_id
                        existing_game["home_team_abbreviation"] = team_abbreviation
                        existing_game["home_team_name"] = team_name
                        existing_game["home_team_score"] = pts
                    else:
                        existing_game["visitor_team_id"] = team_id
                        existing_game["visitor_team_abbreviation"] = team_abbreviation
                        existing_game["visitor_team_name"] = team_name
                        existing_game["visitor_team_score"] = pts
                else:
                    # Create a new game entry
                    game = {
                        "id": game_id,
                        "date": game_date,
                    }
                    
                    if is_home:
                        game["home_team_id"] = team_id
                        game["home_team_abbreviation"] = team_abbreviation
                        game["home_team_name"] = team_name
                        game["home_team_score"] = pts
                    else:
                        game["visitor_team_id"] = team_id
                        game["visitor_team_abbreviation"] = team_abbreviation
                        game["visitor_team_name"] = team_name
                        game["visitor_team_score"] = pts
                    
                    games_dict[game_id] = game
            
            # Convert dictionary to list
            games = list(games_dict.values())
            
            # Filter out incomplete games
            complete_games = []
            for game in games:
                if all(key in game for key in ["home_team_id", "visitor_team_id", "home_team_score", "visitor_team_score"]):
                    complete_games.append(game)
            
            data['games'] = complete_games
        else:
            st.error("Games API response missing expected data structure")
            data['games'] = []
            
        time.sleep(1)  # Respect rate limits
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching games data: {e}")
        data['games'] = []
    except ValueError as e:
        st.error(f"Error parsing JSON from API: {e}")
        data['games'] = []
    
    # Save to cache file
    with open(cache_file, 'w') as f:
        json.dump(data, f)
    
    return data

def get_sample_data():
    """Return sample data when API fails"""
    # Sample teams data
    teams = [
        {"id": 1610612737, "abbreviation": "ATL", "city": "Atlanta", "name": "Hawks", "full_name": "Atlanta Hawks", "conference": "East", "division": "Southeast", "wins": 32, "losses": 40, "win_pct": 0.444, "points_pg": 118.2, "opp_points_pg": 120.5},
        {"id": 1610612738, "abbreviation": "BOS", "city": "Boston", "name": "Celtics", "full_name": "Boston Celtics", "conference": "East", "division": "Atlantic", "wins": 62, "losses": 16, "win_pct": 0.795, "points_pg": 120.8, "opp_points_pg": 108.2},
        {"id": 1610612751, "abbreviation": "BKN", "city": "Brooklyn", "name": "Nets", "full_name": "Brooklyn Nets", "conference": "East", "division": "Atlantic", "wins": 30, "losses": 48, "win_pct": 0.385, "points_pg": 109.2, "opp_points_pg": 114.5},
        {"id": 1610612766, "abbreviation": "CHA", "city": "Charlotte", "name": "Hornets", "full_name": "Charlotte Hornets", "conference": "East", "division": "Southeast", "wins": 19, "losses": 59, "win_pct": 0.244, "points_pg": 106.6, "opp_points_pg": 116.8},
        {"id": 1610612741, "abbreviation": "CHI", "city": "Chicago", "name": "Bulls", "full_name": "Chicago Bulls", "conference": "East", "division": "Central", "wins": 37, "losses": 41, "win_pct": 0.474, "points_pg": 113.5, "opp_points_pg": 114.2}
    ]
    
    # Sample players data
    players = [
        {"id": 2544, "name": "LeBron James", "team_id": 1610612747, "team_abbreviation": "LAL", "age": 39, "gp": 71, "pts": 25.7, "reb": 7.3, "ast": 8.3, "stl": 1.3, "blk": 0.5, "fg_pct": 0.54, "fg3_pct": 0.41, "ft_pct": 0.75, "min": 35.3, "tov": 3.5},
        {"id": 201939, "name": "Stephen Curry", "team_id": 1610612744, "team_abbreviation": "GSW", "age": 36, "gp": 74, "pts": 26.4, "reb": 4.5, "ast": 5.1, "stl": 0.7, "blk": 0.4, "fg_pct": 0.45, "fg3_pct": 0.40, "ft_pct": 0.92, "min": 33.5, "tov": 2.8},
        {"id": 203507, "name": "Giannis Antetokounmpo", "team_id": 1610612749, "team_abbreviation": "MIL", "age": 29, "gp": 73, "pts": 30.4, "reb": 11.5, "ast": 6.5, "stl": 1.2, "blk": 1.1, "fg_pct": 0.61, "fg3_pct": 0.27, "ft_pct": 0.65, "min": 35.2, "tov": 3.4},
        {"id": 201142, "name": "Kevin Durant", "team_id": 1610612756, "team_abbreviation": "PHX", "age": 35, "gp": 75, "pts": 27.1, "reb": 6.6, "ast": 5.0, "stl": 0.9, "blk": 1.2, "fg_pct": 0.52, "fg3_pct": 0.41, "ft_pct": 0.85, "min": 36.0, "tov": 3.3},
        {"id": 203999, "name": "Nikola Jokic", "team_id": 1610612743, "team_abbreviation": "DEN", "age": 29, "gp": 79, "pts": 26.4, "reb": 12.4, "ast": 9.0, "stl": 1.4, "blk": 0.9, "fg_pct": 0.58, "fg3_pct": 0.35, "ft_pct": 0.81, "min": 34.6, "tov": 3.0}
    ]
    
    # Sample games data
    games = [
        {"id": "0022300001", "date": "2023-10-24", "home_team_id": 1610612738, "home_team_abbreviation": "BOS", "home_team_name": "Boston Celtics", "home_team_score": 126, "visitor_team_id": 1610612752, "visitor_team_abbreviation": "NYK", "visitor_team_name": "New York Knicks", "visitor_team_score": 113},
        {"id": "0022300002", "date": "2023-10-24", "home_team_id": 1610612747, "home_team_abbreviation": "LAL", "home_team_name": "Los Angeles Lakers", "home_team_score": 119, "visitor_team_id": 1610612744, "visitor_team_abbreviation": "GSW", "visitor_team_name": "Golden State Warriors", "visitor_team_score": 107},
        {"id": "0022300003", "date": "2023-10-25", "home_team_id": 1610612749, "home_team_abbreviation": "MIL", "home_team_name": "Milwaukee Bucks", "home_team_score": 120, "visitor_team_id": 1610612756, "visitor_team_abbreviation": "PHX", "visitor_team_name": "Phoenix Suns", "visitor_team_score": 115}
    ]
    
    return {
        "teams": teams,
        "players": players,
        "games": games
    }

def prepare_dataframes(data):
    """Convert API data to pandas DataFrames"""
    
    # Teams DataFrame
    teams_df = pd.DataFrame(data['teams'])
    
    # Players DataFrame
    players_df = pd.DataFrame(data['players'])
    
    # Games DataFrame
    games_df = pd.DataFrame(data['games'])
    
    # Convert date strings to datetime
    games_df['date'] = pd.to_datetime(games_df['date'])
    
    return teams_df, players_df, games_df

def calculate_team_stats(teams_df, games_df):
    """Calculate additional team performance metrics"""
    # Most of the team stats are already calculated in the API response
    # We'll just add a few more metrics
    
    team_stats = teams_df.copy()
    
    # Calculate point differential
    team_stats['point_differential'] = team_stats['points_pg'] - team_stats['opp_points_pg']
    
    # Rename columns for consistency with the rest of the app
    team_stats = team_stats.rename(columns={
        'points_pg': 'points_per_game',
        'opp_points_pg': 'points_allowed_per_game'
    })
    
    return team_stats

# Main application
def main():
    # Header
    st.markdown('<div class="main-header">🏀 NBA Performance Dashboard</div>', unsafe_allow_html=True)
    
    # Fetch data
    with st.spinner('Fetching NBA data...'):
        try:
            data = fetch_nba_data()
            
            # Check if we got empty data from API
            if not data.get('teams') or not data.get('players'):
                st.warning("Could not fetch complete data from the NBA API. Using sample data instead.")
                data = get_sample_data()
                st.info("⚠️ Using sample data for demonstration. The real API data is currently unavailable.")
            
            teams_df, players_df, games_df = prepare_dataframes(data)
            team_stats_df = calculate_team_stats(teams_df, games_df)
        except Exception as e:
            st.error(f"Error processing NBA data: {e}")
            st.info("⚠️ Using sample data for demonstration. The real API data is currently unavailable.")
            data = get_sample_data()
            teams_df, players_df, games_df = prepare_dataframes(data)
            team_stats_df = calculate_team_stats(teams_df, games_df)
    
    # Sidebar filters
    st.sidebar.title("Filters")
    
    # Team selection
    all_teams_option = "All Teams"
    team_options = [all_teams_option] + sorted(teams_df['full_name'].tolist())
    selected_team = st.sidebar.selectbox("Select Team", team_options)
    
    # Filter data based on team selection
    if selected_team != all_teams_option:
        filtered_team_stats = team_stats_df[team_stats_df['full_name'] == selected_team]
        filtered_players = players_df[players_df['team_abbreviation'] == filtered_team_stats.iloc[0]['abbreviation']]
        filtered_games = games_df[(games_df['home_team_name'] == selected_team) | 
                                 (games_df['visitor_team_name'] == selected_team)]
    else:
        filtered_team_stats = team_stats_df
        filtered_players = players_df
        filtered_games = games_df
    
    # Date range filter
    if not games_df.empty and 'date' in games_df.columns:
        min_date = games_df['date'].min().date()
        max_date = games_df['date'].max().date()
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_games = filtered_games[(filtered_games['date'].dt.date >= start_date) & 
                                           (filtered_games['date'].dt.date <= end_date)]
    
    # Stat type for player comparison
    stat_options = ['pts', 'reb', 'ast', 'stl', 'blk', 'fg_pct', 'fg3_pct', 'ft_pct']
    stat_labels = {
        'pts': 'Points Per Game',
        'reb': 'Rebounds Per Game',
        'ast': 'Assists Per Game',
        'stl': 'Steals Per Game',
        'blk': 'Blocks Per Game',
        'fg_pct': 'Field Goal %',
        'fg3_pct': '3-Point %',
        'ft_pct': 'Free Throw %'
    }
    selected_stat = st.sidebar.selectbox("Player Stat Comparison", 
                                        options=stat_options,
                                        format_func=lambda x: stat_labels[x])
    
    # Dashboard layout
    col1, col2 = st.columns(2)
    
    # Team Performance Section
    with col1:
        st.markdown('<div class="sub-header">Team Performance</div>', unsafe_allow_html=True)
        
        if not filtered_team_stats.empty:
            # Team standings
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Team Standings")
            
            # Sort by win percentage
            standings = filtered_team_stats.sort_values('win_pct', ascending=False).reset_index(drop=True)
            standings['rank'] = standings.index + 1
            standings_display = standings[['rank', 'full_name', 'wins', 'losses', 'win_pct']]
            standings_display['win_pct'] = standings_display['win_pct'].apply(lambda x: f"{x:.3f}")
            standings_display.columns = ['Rank', 'Team', 'Wins', 'Losses', 'Win %']
            
            st.dataframe(standings_display, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Team offensive and defensive ratings
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Team Offensive & Defensive Ratings")
            
            fig = make_subplots(rows=1, cols=2, 
                               subplot_titles=("Points Per Game", "Points Allowed Per Game"),
                               specs=[[{"type": "bar"}, {"type": "bar"}]])
            
            # Sort by points per game for offensive chart
            off_data = filtered_team_stats.sort_values('points_per_game', ascending=False).head(10)
            fig.add_trace(
                go.Bar(
                    x=off_data['full_name'],
                    y=off_data['points_per_game'],
                    marker_color='#17408B',
                    name="PPG"
                ),
                row=1, col=1
            )
            
            # Sort by points allowed for defensive chart (lower is better)
            def_data = filtered_team_stats.sort_values('points_allowed_per_game', ascending=True).head(10)
            fig.add_trace(
                go.Bar(
                    x=def_data['full_name'],
                    y=def_data['points_allowed_per_game'],
                    marker_color='#C9082A',
                    name="Opp PPG"
                ),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            fig.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Win-Loss Record Chart
            if selected_team != all_teams_option:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader(f"{selected_team} Game Results")
                
                team_games = filtered_games.copy()
                team_games['result'] = None
                team_games['points_scored'] = None
                team_games['points_allowed'] = None
                
                for idx, game in team_games.iterrows():
                    if game['home_team_name'] == selected_team:
                        team_games.at[idx, 'points_scored'] = game['home_team_score']
                        team_games.at[idx, 'points_allowed'] = game['visitor_team_score']
                        team_games.at[idx, 'result'] = 'Win' if game['home_team_score'] > game['visitor_team_score'] else 'Loss'
                    else:
                        team_games.at[idx, 'points_scored'] = game['visitor_team_score']
                        team_games.at[idx, 'points_allowed'] = game['home_team_score']
                        team_games.at[idx, 'result'] = 'Win' if game['visitor_team_score'] > game['home_team_score'] else 'Loss'
                
                # Create a line chart of game results
                team_games = team_games.sort_values('date')
                
                fig = go.Figure()
                
                # Add points scored line
                fig.add_trace(go.Scatter(
                    x=team_games['date'],
                    y=team_games['points_scored'],
                    mode='lines+markers',
                    name='Points Scored',
                    line=dict(color='#17408B', width=3),
                    marker=dict(
                        size=10,
                        color=team_games['result'].map({'Win': '#17408B', 'Loss': '#C9082A'}),
                        line=dict(width=2, color='DarkSlateGrey')
                    )
                ))
                
                # Add points allowed line
                fig.add_trace(go.Scatter(
                    x=team_games['date'],
                    y=team_games['points_allowed'],
                    mode='lines+markers',
                    name='Points Allowed',
                    line=dict(color='#C9082A', width=3, dash='dot'),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title=f"{selected_team} Game Performance",
                    xaxis_title="Game Date",
                    yaxis_title="Points",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Player Performance Section
    with col2:
        st.markdown('<div class="sub-header">Player Performance</div>', unsafe_allow_html=True)
        
        if not filtered_players.empty:
            # Top players by selected stat
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader(f"Top Players by {stat_labels[selected_stat]}")
            
            # Filter out players with few games
            qualified_players = filtered_players[filtered_players['gp'] >= 5]
            
            if not qualified_players.empty:
                # Sort by selected stat
                top_players = qualified_players.sort_values(selected_stat, ascending=False).head(10)
                
                fig = px.bar(
                    top_players,
                    x='name',
                    y=selected_stat,
                    color='team_abbreviation',
                    text=top_players[selected_stat].round(1),
                    title=f"Top 10 Players - {stat_labels[selected_stat]}",
                    labels={'name': 'Player', selected_stat: stat_labels[selected_stat]}
                )
                
                fig.update_layout(xaxis_tickangle=45, height=400)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough qualified players with the current filters.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Player comparison radar chart
            if selected_team != all_teams_option:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader(f"{selected_team} Player Comparison")
                
                team_players = filtered_players[filtered_players['team_abbreviation'] == filtered_team_stats.iloc[0]['abbreviation']]
                
                if not team_players.empty:
                    # Allow selection of players to compare
                    player_options = team_players['name'].tolist()
                    
                    if len(player_options) >= 2:
                        selected_players = st.multiselect(
                            "Select Players to Compare",
                            options=player_options,
                            default=player_options[:min(3, len(player_options))]
                        )
                        
                        if selected_players:
                            # Filter for selected players
                            players_to_compare = team_players[team_players['name'].isin(selected_players)]
                            
                            # Create radar chart
                            categories = ['pts', 'reb', 'ast', 'stl', 'blk']
                            category_labels = [stat_labels[cat] for cat in categories]
                            
                            fig = go.Figure()
                            
                            for _, player in players_to_compare.iterrows():
                                fig.add_trace(go.Scatterpolar(
                                    r=[player[cat] for cat in categories],
                                    theta=category_labels,
                                    fill='toself',
                                    name=player['name']
                                ))
                            
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                    )
                                ),
                                showlegend=True,
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Need at least 2 players for comparison.")
                else:
                    st.info("No player data available for the selected team.")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Recent Games Results
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Recent Game Results")
        
        if not filtered_games.empty:
            # Sort games by date (most recent first)
            recent_games = filtered_games.sort_values('date', ascending=False).head(5)
            
            # Format for display
            for _, game in recent_games.iterrows():
                date_str = game['date'].strftime('%b %d, %Y')
                
                # Format the score
                if pd.notna(game['home_team_score']) and pd.notna(game['visitor_team_score']):
                    score = f"{int(game['home_team_score'])} - {int(game['visitor_team_score'])}"
                    
                    # Determine winner
                    if game['home_team_score'] > game['visitor_team_score']:
                        home_style = "font-weight: bold; color: green;"
                        visitor_style = "color: red;"
                    else:
                        home_style = "color: red;"
                        visitor_style = "font-weight: bold; color: green;"
                else:
                    score = "Not Available"
                    home_style = ""
                    visitor_style = ""
                
                # Display the game
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <div style="width: 30%;">{date_str}</div>
                    <div style="width: 30%; {visitor_style}">{game['visitor_team_name']}</div>
                    <div style="width: 10%; text-align: center;">{score}</div>
                    <div style="width: 30%; {home_style}">{game['home_team_name']}</div>
                </div>
                <hr>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent games available with the current filters.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Full width sections
    st.markdown('<div class="sub-header">League-wide Analysis</div>', unsafe_allow_html=True)
    
    # Team comparison scatter plot
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Team Offensive vs Defensive Efficiency")
    
    if not filtered_team_stats.empty:
        fig = px.scatter(
            filtered_team_stats,
            x='points_allowed_per_game',
            y='points_per_game',
            size='win_pct',
            color='win_pct',
            hover_name='full_name',
            size_max=20,
            color_continuous_scale=px.colors.sequential.Blues,
            labels={
                'points_per_game': 'Points Per Game (Offense)',
                'points_allowed_per_game': 'Points Allowed Per Game (Defense)',
                'win_pct': 'Win Percentage'
            }
        )
        
        # Add quadrant lines
        if not filtered_team_stats.empty:
            avg_off = filtered_team_stats['points_per_game'].mean()
            avg_def = filtered_team_stats['points_allowed_per_game'].mean()
            
            fig.add_hline(y=avg_off, line_dash="dash", line_color="gray", opacity=0.7)
            fig.add_vline(x=avg_def, line_dash="dash", line_color="gray", opacity=0.7)
            
            # Add quadrant labels
            fig.add_annotation(x=avg_def-5, y=avg_off+5, text="Elite Teams<br>(Good offense, Good defense)", 
                              showarrow=False, font=dict(size=10, color="green"))
            fig.add_annotation(x=avg_def+5, y=avg_off+5, text="Offensive Teams<br>(Good offense, Poor defense)", 
                              showarrow=False, font=dict(size=10, color="orange"))
            fig.add_annotation(x=avg_def-5, y=avg_off-5, text="Defensive Teams<br>(Poor offense, Good defense)", 
                              showarrow=False, font=dict(size=10, color="blue"))
            fig.add_annotation(x=avg_def+5, y=avg_off-5, text="Rebuilding Teams<br>(Poor offense, Poor defense)", 
                              showarrow=False, font=dict(size=10, color="red"))
        
        fig.update_layout(height=600)
        fig.update_xaxes(title_text="Points Allowed Per Game", autorange="reversed")
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough team data available with the current filters.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Player stats table
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Player Statistics")
    
    if not filtered_players.empty:
        # Allow filtering by minimum games played
        min_games = st.slider("Minimum Games Played", 1, 20, 5)
        qualified_players = filtered_players[filtered_players['gp'] >= min_games]
        
        if not qualified_players.empty:
            # Select columns to display
            display_cols = ['name', 'team_abbreviation', 'gp', 'pts', 'reb', 'ast', 
                           'stl', 'blk', 'fg_pct', 'fg3_pct', 'ft_pct', 'min']
            
            # Format percentages
            formatted_players = qualified_players.copy()
            for col in ['fg_pct', 'fg3_pct', 'ft_pct']:
                formatted_players[col] = formatted_players[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
            
            # Round other stats to 1 decimal place
            for col in ['pts', 'reb', 'ast', 'stl', 'blk']:
                formatted_players[col] = formatted_players[col].round(1)
            
            # Rename columns for display
            display_df = formatted_players[display_cols].rename(columns={
                'name': 'Player',
                'team_abbreviation': 'Team',
                'gp': 'GP',
                'pts': 'PPG',
                'reb': 'RPG',
                'ast': 'APG',
                'stl': 'SPG',
                'blk': 'BPG',
                'fg_pct': 'FG%',
                'fg3_pct': '3P%',
                'ft_pct': 'FT%',
                'min': 'MPG'
            })
            
            # Add sorting capability
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No qualified players with the current filters.")
    else:
        st.info("No player data available.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with data update info
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; color: #666;">
        <p>Data last updated: {}</p>
        <p>Data source: NBA Stats API</p>
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d')), unsafe_allow_html=True)

if __name__ == "__main__":
    main()

