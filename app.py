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

# Cache function to avoid repeated API calls
@st.cache_data(ttl=3600*24)  # Cache for 24 hours
def fetch_nba_data():
    """Fetch NBA data from the balldontlie API"""
    
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
    
    # Fetch teams
    teams_response = requests.get('https://www.balldontlie.io/api/v1/teams')
    data['teams'] = teams_response.json()['data']
    
    # Fetch players (paginated)
    players = []
    page = 1
    per_page = 100
    total_pages = 1
    
    while page <= total_pages:
        players_response = requests.get(f'https://www.balldontlie.io/api/v1/players?page={page}&per_page={per_page}')
        response_data = players_response.json()
        players.extend(response_data['data'])
        total_pages = response_data['meta']['total_pages']
        page += 1
        time.sleep(1)  # Avoid rate limiting
    
    data['players'] = players
    
    # Fetch recent games
    today = datetime.now()
    start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    
    games_response = requests.get(
        f'https://www.balldontlie.io/api/v1/games?start_date={start_date}&end_date={end_date}&per_page=100'
    )
    data['games'] = games_response.json()['data']
    
    # Fetch stats for players
    stats = []
    for i in range(1, 4):  # Get stats from a few recent pages
        stats_response = requests.get(f'https://www.balldontlie.io/api/v1/stats?page={i}&per_page=100')
        stats.extend(stats_response.json()['data'])
        time.sleep(1)  # Avoid rate limiting
    
    data['stats'] = stats
    
    # Save to cache file
    with open(cache_file, 'w') as f:
        json.dump(data, f)
    
    return data

def prepare_dataframes(data):
    """Convert API data to pandas DataFrames"""
    
    # Teams DataFrame
    teams_df = pd.DataFrame(data['teams'])
    
    # Players DataFrame
    players_df = pd.DataFrame(data['players'])
    # Extract team info from nested structure
    players_df['team_id'] = players_df['team'].apply(lambda x: x['id'] if x else None)
    players_df['team_name'] = players_df['team'].apply(lambda x: x['full_name'] if x else None)
    players_df['team_abbreviation'] = players_df['team'].apply(lambda x: x['abbreviation'] if x else None)
    
    # Games DataFrame
    games_df = pd.DataFrame(data['games'])
    # Extract home and visitor team info
    games_df['home_team_id'] = games_df['home_team'].apply(lambda x: x['id'])
    games_df['home_team_name'] = games_df['home_team'].apply(lambda x: x['full_name'])
    games_df['visitor_team_id'] = games_df['visitor_team'].apply(lambda x: x['id'])
    games_df['visitor_team_name'] = games_df['visitor_team'].apply(lambda x: x['full_name'])
    
    # Stats DataFrame
    stats_df = pd.DataFrame(data['stats'])
    # Extract player and team info
    stats_df['player_id'] = stats_df['player'].apply(lambda x: x['id'] if x else None)
    stats_df['player_name'] = stats_df['player'].apply(lambda x: f"{x['first_name']} {x['last_name']}" if x else None)
    stats_df['team_id'] = stats_df['team'].apply(lambda x: x['id'] if x else None)
    stats_df['team_name'] = stats_df['team'].apply(lambda x: x['full_name'] if x else None)
    stats_df['game_id'] = stats_df['game'].apply(lambda x: x['id'] if x else None)
    
    # Convert date strings to datetime
    games_df['date'] = pd.to_datetime(games_df['date'])
    
    return teams_df, players_df, games_df, stats_df

def calculate_team_stats(games_df):
    """Calculate team performance metrics"""
    team_stats = {}
    
    for _, game in games_df.iterrows():
        # Skip games without scores
        if pd.isna(game['home_team_score']) or pd.isna(game['visitor_team_score']):
            continue
            
        home_id = game['home_team_id']
        visitor_id = game['visitor_team_id']
        
        # Initialize team stats if not exists
        if home_id not in team_stats:
            team_stats[home_id] = {
                'team_name': game['home_team_name'],
                'games_played': 0,
                'wins': 0,
                'losses': 0,
                'points_scored': 0,
                'points_allowed': 0
            }
            
        if visitor_id not in team_stats:
            team_stats[visitor_id] = {
                'team_name': game['visitor_team_name'],
                'games_played': 0,
                'wins': 0,
                'losses': 0,
                'points_scored': 0,
                'points_allowed': 0
            }
        
        # Update home team stats
        team_stats[home_id]['games_played'] += 1
        team_stats[home_id]['points_scored'] += game['home_team_score']
        team_stats[home_id]['points_allowed'] += game['visitor_team_score']
        
        # Update visitor team stats
        team_stats[visitor_id]['games_played'] += 1
        team_stats[visitor_id]['points_scored'] += game['visitor_team_score']
        team_stats[visitor_id]['points_allowed'] += game['home_team_score']
        
        # Update wins and losses
        if game['home_team_score'] > game['visitor_team_score']:
            team_stats[home_id]['wins'] += 1
            team_stats[visitor_id]['losses'] += 1
        else:
            team_stats[home_id]['losses'] += 1
            team_stats[visitor_id]['wins'] += 1
    
    # Convert to DataFrame and calculate additional metrics
    team_stats_df = pd.DataFrame.from_dict(team_stats, orient='index').reset_index()
    team_stats_df = team_stats_df.rename(columns={'index': 'team_id'})
    
    # Calculate win percentage and point differential
    team_stats_df['win_pct'] = team_stats_df['wins'] / team_stats_df['games_played']
    team_stats_df['point_differential'] = team_stats_df['points_scored'] - team_stats_df['points_allowed']
    team_stats_df['points_per_game'] = team_stats_df['points_scored'] / team_stats_df['games_played']
    team_stats_df['points_allowed_per_game'] = team_stats_df['points_allowed'] / team_stats_df['games_played']
    
    return team_stats_df

def calculate_player_stats(stats_df):
    """Calculate player performance metrics"""
    # Group by player and calculate averages
    player_stats = stats_df.groupby(['player_id', 'player_name', 'team_id', 'team_name']).agg({
        'pts': 'mean',
        'reb': 'mean',
        'ast': 'mean',
        'stl': 'mean',
        'blk': 'mean',
        'fg_pct': 'mean',
        'fg3_pct': 'mean',
        'ft_pct': 'mean',
        'turnover': 'mean',
        'min': lambda x: pd.to_numeric(x.str.replace(':', '.'), errors='coerce').mean()
    }).reset_index()
    
    # Rename columns to indicate they are averages
    player_stats = player_stats.rename(columns={
        'pts': 'ppg',
        'reb': 'rpg',
        'ast': 'apg',
        'stl': 'spg',
        'blk': 'bpg',
        'turnover': 'topg',
        'min': 'mpg'
    })
    
    # Calculate games played
    games_played = stats_df.groupby(['player_id']).size().reset_index(name='games_played')
    player_stats = player_stats.merge(games_played, on='player_id')
    
    # Calculate PER (simplified version)
    player_stats['per'] = (player_stats['ppg'] + player_stats['rpg'] + 
                          player_stats['apg'] + player_stats['spg'] + 
                          player_stats['bpg'] - player_stats['topg']) / 5
    
    return player_stats

# Main application
def main():
    # Header
    st.markdown('<div class="main-header">🏀 NBA Performance Dashboard</div>', unsafe_allow_html=True)
    
    # Fetch data
    with st.spinner('Fetching NBA data...'):
        data = fetch_nba_data()
        teams_df, players_df, games_df, stats_df = prepare_dataframes(data)
        team_stats_df = calculate_team_stats(games_df)
        player_stats_df = calculate_player_stats(stats_df)
    
    # Sidebar filters
    st.sidebar.title("Filters")
    
    # Team selection
    all_teams_option = "All Teams"
    team_options = [all_teams_option] + sorted(teams_df['full_name'].tolist())
    selected_team = st.sidebar.selectbox("Select Team", team_options)
    
    # Filter data based on team selection
    if selected_team != all_teams_option:
        filtered_team_stats = team_stats_df[team_stats_df['team_name'] == selected_team]
        filtered_player_stats = player_stats_df[player_stats_df['team_name'] == selected_team]
        filtered_games = games_df[(games_df['home_team_name'] == selected_team) | 
                                 (games_df['visitor_team_name'] == selected_team)]
    else:
        filtered_team_stats = team_stats_df
        filtered_player_stats = player_stats_df
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
    stat_options = ['ppg', 'rpg', 'apg', 'spg', 'bpg', 'fg_pct', 'fg3_pct', 'ft_pct', 'per']
    stat_labels = {
        'ppg': 'Points Per Game',
        'rpg': 'Rebounds Per Game',
        'apg': 'Assists Per Game',
        'spg': 'Steals Per Game',
        'bpg': 'Blocks Per Game',
        'fg_pct': 'Field Goal %',
        'fg3_pct': '3-Point %',
        'ft_pct': 'Free Throw %',
        'per': 'Player Efficiency Rating'
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
            standings_display = standings[['rank', 'team_name', 'wins', 'losses', 'win_pct']]
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
                    x=off_data['team_name'],
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
                    x=def_data['team_name'],
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
        
        if not filtered_player_stats.empty:
            # Top players by selected stat
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader(f"Top Players by {stat_labels[selected_stat]}")
            
            # Filter out players with few games
            qualified_players = filtered_player_stats[filtered_player_stats['games_played'] >= 5]
            
            if not qualified_players.empty:
                # Sort by selected stat
                top_players = qualified_players.sort_values(selected_stat, ascending=False).head(10)
                
                fig = px.bar(
                    top_players,
                    x='player_name',
                    y=selected_stat,
                    color='team_name',
                    text=top_players[selected_stat].round(1),
                    title=f"Top 10 Players - {stat_labels[selected_stat]}",
                    labels={'player_name': 'Player', selected_stat: stat_labels[selected_stat]}
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
                
                team_players = filtered_player_stats[filtered_player_stats['team_name'] == selected_team]
                
                if not team_players.empty:
                    # Allow selection of players to compare
                    player_options = team_players['player_name'].tolist()
                    
                    if len(player_options) >= 2:
                        selected_players = st.multiselect(
                            "Select Players to Compare",
                            options=player_options,
                            default=player_options[:min(3, len(player_options))]
                        )
                        
                        if selected_players:
                            # Filter for selected players
                            players_to_compare = team_players[team_players['player_name'].isin(selected_players)]
                            
                            # Create radar chart
                            categories = ['ppg', 'rpg', 'apg', 'spg', 'bpg']
                            category_labels = [stat_labels[cat] for cat in categories]
                            
                            fig = go.Figure()
                            
                            for _, player in players_to_compare.iterrows():
                                fig.add_trace(go.Scatterpolar(
                                    r=[player[cat] for cat in categories],
                                    theta=category_labels,
                                    fill='toself',
                                    name=player['player_name']
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
            hover_name='team_name',
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
    
    if not filtered_player_stats.empty:
        # Allow filtering by minimum games played
        min_games = st.slider("Minimum Games Played", 1, 20, 5)
        qualified_players = filtered_player_stats[filtered_player_stats['games_played'] >= min_games]
        
        if not qualified_players.empty:
            # Select columns to display
            display_cols = ['player_name', 'team_name', 'games_played', 'ppg', 'rpg', 'apg', 
                           'spg', 'bpg', 'fg_pct', 'fg3_pct', 'ft_pct', 'per']
            
            # Format percentages
            formatted_players = qualified_players.copy()
            for col in ['fg_pct', 'fg3_pct', 'ft_pct']:
                formatted_players[col] = formatted_players[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
            
            # Round other stats to 1 decimal place
            for col in ['ppg', 'rpg', 'apg', 'spg', 'bpg', 'per']:
                formatted_players[col] = formatted_players[col].round(1)
            
            # Rename columns for display
            display_df = formatted_players[display_cols].rename(columns={
                'player_name': 'Player',
                'team_name': 'Team',
                'games_played': 'GP',
                'ppg': 'PPG',
                'rpg': 'RPG',
                'apg': 'APG',
                'spg': 'SPG',
                'bpg': 'BPG',
                'fg_pct': 'FG%',
                'fg3_pct': '3P%',
                'ft_pct': 'FT%',
                'per': 'PER'
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
        <p>Data source: balldontlie API</p>
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d')), unsafe_allow_html=True)

if __name__ == "__main__":
    main()

