import pandas as pd
import requests
import os

"Functions used for data preparation"
def add_game_number_column(df):
    """
    Adds a 'game_number' column to the existing DataFrame to distinguish between multiple fixtures within the same gameweek.
    """
    # Initialize a new column for game number
    df['game_number'] = 1
    
    # Sort the DataFrame to ensure consistency in assigning game numbers
    df = df.sort_values(by=['player_ID', 'season', 'GW', 'kickoff_time'])
    
    # Group by 'player_ID', 'season' and assign a cumulative count to 'game_number' for each fixture within the same season
    df['game_number'] = df.groupby(['player_ID', 'season', 'GW']).cumcount() + 1
    
    return df

def total_GW_points(df):
    """Adds total points for a particular GW for each player to the dataframe."""

    df = df.sort_values(by=['player_ID', 'season', 'GW', 'kickoff_time'])
    df['total_GW_points'] = df.groupby(['player_ID', 'season', 'GW'])['points'].transform('sum')

    return df

def add_player_stats(df):
    """Adds point and minute statistics for each player to the dataframe."""
    
    # Sort dataframe by player ID, Season, and GW
    df = df.sort_values(by=['player_ID', 'season', 'GW', 'kickoff_time'])
    
    # Calculate cumulative statistics for points and minutes
    df['points_last_game'] = df.groupby(['player_ID', 'season'])['points'].shift(1)
    df['total_points'] = df.groupby(['player_ID', 'season'])['points_last_game'].cumsum()
    df['mins_last_game'] = df.groupby(['player_ID', 'season'])['minutes'].shift(1)
    df['total_mins'] = df.groupby(['player_ID', 'season'])['mins_last_game'].cumsum()

    # Rolling statistics for points and minutes
    for window in [3, 5, 10]:
        df[f'mean_points_last_{window}'] = df.groupby(['player_ID', 'season'])['points'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
        df[f'mean_mins_last_{window}'] = df.groupby(['player_ID', 'season'])['minutes'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
    
    return df

def add_team_stats(df):
    "Adds point and minute statistics for each player's team to the data frame"
    
    # Calculate team points per gameweek
    df = df.sort_values(by=['team', 'season', 'GW', 'kickoff_time'])
    df['team_points'] = df.groupby(['team','season', 'GW', 'kickoff_time'])['points'].transform('sum')

    # Sort dataframe back by player ID for calculating player-level team stats
    df = df.sort_values(by=['player_ID', 'season', 'GW', 'kickoff_time'])

    # Calculate cumulative and rolling team points for each player
    df['team_points_last_game'] = df.groupby(['player_ID', 'season'])['team_points'].shift(1)
    df['total_team_points'] = df.groupby(['player_ID', 'season'])['team_points_last_game'].cumsum()

    for window in [3, 5, 10]:
        df[f'mean_team_points_last_{window}'] = df.groupby(['player_ID', 'season'])['team_points'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
    
    return df

def add_team_conceded_stats(df):
    """Calculates conceded point statistics for the team based on opponent points."""
    
    # Rename columns for clarity and merge based on opponent team
    opponent_points = df[['team', 'season', 'GW', 'kickoff_time', 'team_points']].rename(columns={'team': 'opponent_team', 'team_points': 'opponent_points'}).drop_duplicates()

    # Merge opponent points back into the original dataframe
    df = df.merge(opponent_points, how='left', on=['opponent_team', 'season', 'GW', 'kickoff_time'])
    df.rename(columns={'opponent_points': 'team_conceded_points'}, inplace=True)

    df = df.sort_values(by=['player_ID', 'season', 'GW', 'kickoff_time'])
    # Calculate cumulative conceded points
    df['team_conceded_last_game'] = df.groupby(['player_ID', 'season'])['team_conceded_points'].shift(1)
    df['total_team_conceded'] = df.groupby(['player_ID', 'season'])['team_conceded_last_game'].cumsum()
    df['mean_team_conceded_last_3'] = df.groupby(['player_ID', 'season'])['team_conceded_points'].transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
    df['mean_team_conceded_last_5'] = df.groupby(['player_ID', 'season'])['team_conceded_points'].transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    df['mean_team_conceded_last_10'] = df.groupby(['player_ID', 'season'])['team_conceded_points'].transform(lambda x: x.rolling(window=10, min_periods=1).mean().shift(1))
    
    return df

def add_opponent_stats(df):
    "Calculates statistics for opponent team and adds to dataframe"

    team_stats = df[['season', 'team', 'GW', 'kickoff_time', 'total_team_points', 'team_points_last_game', 'mean_team_points_last_3', 'mean_team_points_last_5',
                     'mean_team_points_last_10', 'total_team_conceded', 'team_conceded_last_game', 'mean_team_conceded_last_3', 'mean_team_conceded_last_5',
                      'mean_team_conceded_last_10' ]].drop_duplicates()

    team_stats = team_stats.rename(columns={
        'team': 'opponent_team',
        'team_points_last_game': 'opponent_points_last_game',
        'total_team_points': 'total_opponent_points',
        'mean_team_points_last_3': 'mean_opponent_points_last_3',
        'mean_team_points_last_5': 'mean_opponent_points_last_5',
        'mean_team_points_last_10': 'mean_opponent_points_last_10',
        'team_conceded_last_game': 'opponent_conceded_last_game',
        'total_team_conceded': 'total_opponent_conceded',
        'mean_team_conceded_last_3': 'mean_opponent_conceded_last_3',
        'mean_team_conceded_last_5': 'mean_opponent_conceded_last_5',
        'mean_team_conceded_last_10': 'mean_opponent_conceded_last_10'})
    
    team_stats = team_stats.drop_duplicates(subset=['opponent_team', 'season', 'GW', 'kickoff_time'])

    df = df.merge(team_stats, how='left', on=['season', 'opponent_team', 'GW', 'kickoff_time']).drop_duplicates()

    return df

def add_previous_season_stats(df):
    """Adds player and team stats from the previous season using already calculated cumulative stats."""

    # Sort the dataframe by player ID, season, and GW to ensure data is in the right order
    df = df.sort_values(by=['player_ID', 'season', 'GW'])

    # Define a helper function for previous season column calculation
    def calculate_next_season(season):
        start, end = season.split('-')
        return f"{int(start) + 1}-{int(end) + 1}"

    # Get the total points and minutes from the current season for each player
    player_season_stats = df.groupby(['player_ID', 'season']).agg(total_points=('points', 'sum'), total_mins=('minutes', 'sum')).reset_index()

    # Calculate previous season column
    player_season_stats['season'] = player_season_stats['season'].apply(calculate_next_season)

    # Merge current season stats into the next season for players
    df = df.merge(
        player_season_stats[['player_ID', 'season', 'total_points', 'total_mins']],
        how='left',left_on=['player_ID', 'season'],right_on=['player_ID', 'season'],suffixes=(None, '_last_season'))

    # Perform similar operations for teams
    df_sorted = df.sort_values(by=['season', 'GW', 'team'])
    df_unique = df_sorted.drop_duplicates(subset=['season', 'GW', 'team'])

    team_season_stats = df_unique.groupby(['team', 'season']).agg(total_team_points=('team_points', 'sum'),total_team_conceded=('team_conceded_points', 'sum')).reset_index()

    team_season_stats['season'] = team_season_stats['season'].apply(calculate_next_season)

    df = df.merge(
        team_season_stats[['team', 'season', 'total_team_points', 'total_team_conceded']],
        how='left',left_on=['team', 'season'],right_on=['team', 'season'],suffixes=(None, '_last_season'))
    

    return df

def add_previous_opponent_stats(df):
    """Adds stats for opponent from previous season"""
    previous_team_stats = df[['season','team','total_team_points_last_season', 'total_team_conceded_last_season']].drop_duplicates()

    previous_team_stats = previous_team_stats.rename(columns={
        'team':'opponent_team',
        'total_team_points_last_season':'total_opponent_points_last_season',
        'total_team_conceded_last_season':'total_opponent_conceded_last_season'})
    
    previous_team_stats = previous_team_stats.drop_duplicates(subset=['opponent_team', 'season'])

    df = df.merge(previous_team_stats, how='left', on=['season', 'opponent_team']).drop_duplicates()

    return df
    
def fetch_api_data(team_market_values, team_stats_previous):
    """Fetches data for current season from fpl api"""

    bootstrap_url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    bootstrap_response = requests.get(bootstrap_url)
    data = bootstrap_response.json()

    # Get team IDs
    team_info = data['teams']
    team_ids = {team['id']: team['name'] for team in team_info}

    # map element types to position
    position_map = {1:'GK', 2:'DEF', 3:'MID', 4:'FWD'}

    rows_list = []
    for player in data['elements']:
        player_id = player['id']
        player_name = player['first_name'] + ' ' + player['second_name']
        player_position = position_map[player['element_type']]
        player_team = player['team']

        # Fetch player details
        player_detail_url = f'https://fantasy.premierleague.com/api/element-summary/{player_id}/'
        player_detail_response = requests.get(player_detail_url)
        player_detail_data = player_detail_response.json()

        values = []
        for gw in player_detail_data['history']:
            row = {'season': '2024-25', 'name': player_name, 'player_ID': player_id, 'position': player_position,'team': player_team, 'team_name': team_ids[player_team] , 
                    'team_market_value': team_market_values[player_team], 'element': gw['element'],
                    'kickoff_time': gw['kickoff_time'], 'minutes': gw['minutes'], 'opponent_team': gw['opponent_team'], 
                    'opp_team_name': team_ids[gw['opponent_team']], 'opponent_market_value': team_market_values[gw['opponent_team']], 
                    'selected': gw['selected'], 'team_h_score': gw['team_h_score'], 'team_a_score': gw['team_a_score'], 
                    'transfers_balance': gw['transfers_balance'],'transfers_in': gw['transfers_in'], 'transfers_out': gw['transfers_out'],
                    'value': gw['value'], 'was_home': int(gw['was_home']), 'GW': gw['round'], 'points': gw['total_points']}

            values.append(gw['value'])
                # Check if the player has history_past data (i.e., data from the previous season)
            if player_detail_data['history_past']:
                row['total_points_last_season'] = player_detail_data['history_past'][0]['total_points']
                row['total_mins_last_season'] = player_detail_data['history_past'][0]['minutes']
            else:
                row['total_points_last_season'] = None
                row['total_mins_last_season'] = None
        
                # Add additional team stats for last season
            row['total_team_points_last_season'] = team_stats_previous[player_team]['points']
            row['total_team_conceded_last_season'] = team_stats_previous[player_team]['conceded']
            row['total_opponent_points_last_season'] = team_stats_previous[gw['opponent_team']]['points']
            row['total_opponent_conceded_last_season'] = team_stats_previous[gw['opponent_team']]['conceded']
        
            rows_list.append(row)

        if not values:
            print(f"Warning: No history data for player {player_name} (ID: {player_id}).")
        if values:
            for gw in player_detail_data['fixtures']:
                last_value = values[-1]

                opponent_team = gw['team_a'] if gw['is_home'] else gw['team_h']
                row = {'season': '2024-25', 'name': player_name, 'player_ID': player_id, 'position': player_position, 'team': player_team, 
                    'team_name' : team_ids[player_team], 'team_market_value': team_market_values[player_team], 'element': None,
                    'kickoff_time': gw['kickoff_time'], 'minutes': None, 'opponent_team': opponent_team, 
                    'opp_team_name': team_ids[opponent_team], 'opponent_market_value': team_market_values[opponent_team], 
                    'selected': None, 'team_h_score': gw['team_h_score'], 'team_a_score': gw['team_a_score'], 
                    'transfers_balance': None,'transfers_in': None, 'transfers_out': None,
                    'value': last_value, 'was_home': int(gw['is_home']), 'GW': gw['event'], 'points': None}
            
                if player_detail_data['history_past']:
                    row['total_points_last_season'] = player_detail_data['history_past'][0]['total_points']
                    row['total_mins_last_season'] = player_detail_data['history_past'][0]['minutes']
                else:
                    row['total_points_last_season'] = None
                    row['total_mins_last_season'] = None
            
                row['total_team_points_last_season'] = team_stats_previous[player_team]['points']
                row['total_team_conceded_last_season'] = team_stats_previous[player_team]['conceded']
                row['total_opponent_points_last_season'] = team_stats_previous[opponent_team]['points']
                row['total_opponent_conceded_last_season'] = team_stats_previous[opponent_team]['conceded']

        
                rows_list.append(row)

    df_new = pd.DataFrame(rows_list)

    return df_new

def closest_players_value(row, df, stat_column):
    """Finds players with closest value and returns the mean of their statistics."""
    
    same_position_players = df[(df['position'] == row['position']) & (~df[stat_column].isna())]
    same_position_players['value_diff'] = (same_position_players['value'] - row['value']).abs()

    min_diff = same_position_players['value_diff'].min()
    closest_players = same_position_players[same_position_players['value_diff'] == min_diff]
    
    return closest_players[stat_column].mean()

