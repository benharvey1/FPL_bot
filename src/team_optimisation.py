import pandas as pd
import src.data_statistics as data_statistics
import src.web_service as web_service
import pulp
import os
from dotenv import load_dotenv

load_dotenv()
team_id = os.getenv('FPL_TEAM_ID')

"Functions used to optimise team selection"

# First 3 functions used to predict points players will score in upcoming GWs

def predict_player(row, classifiers, regressors):
        """make player predictions for player"""

        feature_columns = ['team_market_value', 'opponent_market_value', 'value', 'was_home','points_last_game', 'total_points', 'mins_last_game',
                        'total_mins', 'mean_points_last_3', 'mean_mins_last_3', 'mean_points_last_5','mean_mins_last_5', 'mean_points_last_10', 
                        'mean_mins_last_10', 'team_points_last_game', 'total_team_points', 'mean_team_points_last_3', 'mean_team_points_last_5',
                        'mean_team_points_last_10', 'team_conceded_last_game', 'total_team_conceded', 
                        'mean_team_conceded_last_3', 'mean_team_conceded_last_5', 'mean_team_conceded_last_10', 'total_opponent_points',
                        'opponent_points_last_game', 'mean_opponent_points_last_3', 'mean_opponent_points_last_5', 'mean_opponent_points_last_10',
                        'total_opponent_conceded', 'opponent_conceded_last_game', 'mean_opponent_conceded_last_3', 'mean_opponent_conceded_last_5',
                        'mean_opponent_conceded_last_10', 'total_points_last_season', 'total_mins_last_season', 'total_team_points_last_season',
                        'total_team_conceded_last_season', 'total_opponent_points_last_season', 'total_opponent_conceded_last_season']
        
        features = pd.DataFrame([row[feature_columns].values], columns=feature_columns)

        # Predict appearance based on the player's position
        position = row['position']
        player_id = row['player_ID']
        status = web_service.get_injury_status(player_id)
        appearance = 0
        predicted_points = 0

        if status == 'i' or status == 'u' or status == 's' or status == 'l' or status == 'n':
            appearance = 0
        else:
            appearance = classifiers[position].predict(features)[0]
        
        if status == 'a' and appearance == 1:
            # Predict points only if the player is predicted to appear
            predicted_points = regressors[position].predict(features)[0]

        elif status == 'd' and appearance == 1:
            predicted_points = regressors[position].predict(features)[0]*0.5

        return appearance, predicted_points

def predict_next_gw(df, classifiers, regressors):
    """Make predictions for the next gw.
    - Returns a list of dictionaries. Takes into account double/blank GWs."""
    played_games = df[df['points'].notnull()]
    last_gw = played_games['GW'].max()
    next_gw = last_gw + 1

    next_gw_df = df[df['GW'] == next_gw][['name', 'player_ID', 'position', 'kickoff_time', 'team_name', 'team', 'opp_team_name', 'team_market_value', 'opponent_market_value', 'value', 'was_home','points_last_game', 'total_points', 'mins_last_game',
                        'total_mins', 'mean_points_last_3', 'mean_mins_last_3', 'mean_points_last_5','mean_mins_last_5', 'mean_points_last_10', 
                        'mean_mins_last_10', 'team_points_last_game', 'total_team_points', 'mean_team_points_last_3', 'mean_team_points_last_5',
                        'mean_team_points_last_10', 'team_conceded_last_game', 'total_team_conceded', 
                        'mean_team_conceded_last_3', 'mean_team_conceded_last_5', 'mean_team_conceded_last_10', 'total_opponent_points',
                        'opponent_points_last_game', 'mean_opponent_points_last_3', 'mean_opponent_points_last_5', 'mean_opponent_points_last_10',
                        'total_opponent_conceded', 'opponent_conceded_last_game', 'mean_opponent_conceded_last_3', 'mean_opponent_conceded_last_5',
                        'mean_opponent_conceded_last_10', 'total_points_last_season', 'total_mins_last_season', 'total_team_points_last_season',
                        'total_team_conceded_last_season', 'total_opponent_points_last_season', 'total_opponent_conceded_last_season']]
    next_gw_df.dropna(inplace=True) 

    predictions = []
    for _, row in next_gw_df.iterrows():
        appearance, predicted_points = predict_player(row, classifiers, regressors)
        prediction = {'GW': next_gw, 'kickoff_time': row['kickoff_time'], 'name': row['name'], 'player_ID': row['player_ID'], 'position': row['position'], 'value': row['value'], 'team': row['team'], 
                      'team_name': row['team_name'], 'opp_team_name': row['opp_team_name'], 'appearance': appearance, 
                      'predicted_points': predicted_points}
        
        predictions.append(prediction)

    predictions_df = pd.DataFrame(predictions)  # for a player with a DGW, will be two rows for that player
    
    # Group by 'player_ID' and sum the 'predicted_points'
    predictions_df['total_GW_points'] = predictions_df.groupby('player_ID')['predicted_points'].transform('sum')
    predictions_list = predictions_df.to_dict(orient='records')
    
    return predictions_list

def predict_next_x_gws(df, classifiers, regressors, x):
    """Make predictions for next x gws.
    - Returns a list of dictionaries with one dictionary for each player """

    predicted_data = df.copy()
    player_ids = predicted_data['player_ID'].unique()
    predictions = []

    for id in player_ids:
        name = predicted_data.loc[predicted_data['player_ID'] == id, 'name'].iloc[0]
        positon = predicted_data.loc[predicted_data['player_ID'] == id, 'position'].iloc[0]
        team = predicted_data.loc[predicted_data['player_ID'] == id, 'team'].iloc[0]
        value = (predicted_data[predicted_data['player_ID'] == id].dropna(subset=['value']).sort_values(by='GW', ascending=False).iloc[0]['value'])
        predictions.append({'name': name, 'player_ID': id, 'position': positon, 'value': value, 'team': team, 'predicted_points': 0})
    
    for i in range(x):

        predicted_data = predicted_data[['season','name', 'player_ID', 'position', 'team', 'team_name', 'team_market_value', 'minutes', 
                                     'opponent_team', 'opp_team_name', 'opponent_market_value', 'value', 'was_home', 'GW', 'kickoff_time', 'points', 'total_GW_points',
                                     'total_points_last_season', 'total_mins_last_season', 'total_team_points_last_season', 'total_team_conceded_last_season',
                                     'total_opponent_points_last_season', 'total_opponent_conceded_last_season']]
    
        next_gw_predictions = predict_next_gw(df, classifiers, regressors)
        next_gw_df = pd.DataFrame(next_gw_predictions)
        next_gw_aggregated = next_gw_df.groupby('player_ID').agg({'predicted_points': 'sum'}).reset_index()

        for prediction in predictions:
            player_id = prediction['player_ID']
            next_gw_pred = next_gw_aggregated[next_gw_aggregated['player_ID'] == player_id]

            if not next_gw_pred.empty:
                prediction['predicted_points'] += next_gw_pred['predicted_points'].values[0]
        

        next_gw_df['minutes'] = 0
        next_gw_df.loc[(next_gw_df['position'] == 'FWD') & (next_gw_df['appearance'] >= 1), 'minutes'] = 75
        next_gw_df.loc[(next_gw_df['position'] == 'MID') & (next_gw_df['appearance'] >= 1), 'minutes'] = 80
        next_gw_df.loc[(next_gw_df['position'] == 'DEF') & (next_gw_df['appearance'] >= 1), 'minutes'] = 85
        next_gw_df.loc[(next_gw_df['position'] == 'GK') & (next_gw_df['appearance'] >= 1), 'minutes'] = 90

        next_gw_df.rename(columns={'predicted_points': 'points'}, inplace=True)
    
        predicted_data = predicted_data.merge(next_gw_df[['player_ID', 'GW', 'kickoff_time', 'minutes', 'points', 'total_GW_points']], how='left', on=['player_ID', 'GW', 'kickoff_time'], suffixes=('', '_y'))
    
        predicted_data['points'] = predicted_data['points_y'].combine_first(predicted_data['points'])
        predicted_data['minutes'] = predicted_data['minutes_y'].combine_first(predicted_data['minutes'])
        predicted_data['total_GW_points'] = predicted_data['total_GW_points_y'].combine_first(predicted_data['total_GW_points'])
        predicted_data.drop(columns=['points_y', 'minutes_y', 'total_GW_points_y'], inplace=True)

        predicted_data = data_statistics.add_player_stats(predicted_data)
        predicted_data = data_statistics.add_team_stats(predicted_data)
        predicted_data = data_statistics.add_team_conceded_stats(predicted_data)
        predicted_data = data_statistics.add_opponent_stats(predicted_data)

        df = predicted_data

    return predictions

def current_team_tuples(df, classifiers, regressors, x, current_team_ids):
    """Returns a list of named tuples for the current team from the team ids"""

    predictions_df = pd.DataFrame(predict_next_x_gws(df, classifiers, regressors, x))
    players = list(predictions_df.itertuples())



    current_team = []
    for player in players:
        if player.player_ID in current_team_ids:
            current_team.append(player)
    
    return current_team

def choose_starting_11(team):
    """Chooses the best starting 11 from the current team.
    - Returns: list of 11 named tuples (one for each player)"""

    # Initialize the linear optimization problem for the starting 11
    starting_11_prob = pulp.LpProblem("Starting11", pulp.LpMaximize)
    
    # Create decision variables for selecting starting 11 players
    starting_11_vars = {player.player_ID: pulp.LpVariable(f'starting11_{player.player_ID}', cat='Binary') for player in team}

    gk_count = pulp.lpSum([starting_11_vars[player.player_ID] for player in team if player.position == 'GK'])
    def_count = pulp.lpSum([starting_11_vars[player.player_ID] for player in team if player.position == 'DEF'])
    mid_count = pulp.lpSum([starting_11_vars[player.player_ID] for player in team if player.position == 'MID'])
    fwd_count = pulp.lpSum([starting_11_vars[player.player_ID] for player in team if player.position == 'FWD'])

    # Objective function: Maximize total predicted points for the starting 11
    starting_11_prob += pulp.lpSum([starting_11_vars[player.player_ID] * player.predicted_points for player in team])
    
    # Constraints for the starting 11 selection
    starting_11_prob += pulp.lpSum(starting_11_vars.values()) == 11
    starting_11_prob += gk_count == 1   # Must have 1 GK
    starting_11_prob += def_count >= 3  # Must have at least 3 DEFs
    starting_11_prob += mid_count >= 2  # Must have at least 2 MIDs
    starting_11_prob += fwd_count >= 1  # Must have at least 1 FWD

    # Solve the optimization problem for the starting 11
    starting_11_prob.solve()
        
    # Extract the selected starting 11 players
    starting_11 = [player for player in team if pulp.value(starting_11_vars[player.player_ID]) == 1]
    
    # The remaining players are on the bench
    bench_players = [player for player in team if player not in starting_11]

    position_order = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
    starting_11_sorted = sorted(starting_11, key=lambda player: position_order[player.position])
    bench_sorted = sort_subs(bench_players)

    return starting_11_sorted, bench_sorted

def choose_captains(starting_11):
    """Chooses the captain and vice captain from current starting 11.
    - Returns: id of captain,id of vice-captain"""

    starting_list = sorted(starting_11, key=lambda player: player.predicted_points, reverse=True)
    captain = starting_list[0]
    vice = starting_list[1]

    return captain.player_ID, vice.player_ID

def sort_subs(bench):
    """Sorts the subs into correct order.
    - Returns: list of 4 named tuples, with GK first and then 3 outfield players arranged in descending order of predicted points"""

    gk_sub = None
    outfield_subs = []
    for player in bench:
        if player.position == 'GK':
            gk_sub = player
        else:
            outfield_subs.append(player)

    sorted_subs = [gk_sub] + outfield_subs

    return sorted_subs


# Both following functions determine best free hit team for following GW
# Fist one maximises total team points but requires 4 cheap players
# Second one maximises points of starting 11 but requires bench predicted points to be larger than certain (user inputted) value 

def determine_best_free_hit(df, classifiers, regressors):
    """Determines best free hit team for following GW (i.e. does not take into account current team).
    Team has maximum predicted points subject to following constraints:
        - 2 GK, 5 DEF, 5 MID, 3 FWD
        - Max 3 players from same team
        - Total value <= 1000
        - Possible starting formations: 3-5-2, 4-4-2, 4-3-3, 3-4-3, 5-3-2, 5-2-3, 5-4-1
    Returns two lists of named tuples - starting 11 and bench """
    
    predictions = predict_next_x_gws(df, classifiers, regressors, 1)
    predictions_df = pd.DataFrame(predictions)

    gk_predictions = predictions_df[predictions_df['position'] == 'GK'].sort_values(by='predicted_points', ascending=False)
    def_predictions = predictions_df[predictions_df['position'] == 'DEF'].sort_values(by='predicted_points', ascending=False)
    mid_predictions = predictions_df[predictions_df['position'] == 'MID'].sort_values(by='predicted_points', ascending=False)
    fwd_predictions = predictions_df[predictions_df['position'] == 'FWD'].sort_values(by='predicted_points', ascending=False)

    squad_prob = pulp.LpProblem("BestFreeHitTeam", pulp.LpMaximize)

    # Create decision variables for selecting players
    players = list(predictions_df.itertuples())
    player_vars = {player.player_ID: pulp.LpVariable(f'player_{player.player_ID}', cat='Binary') for player in players}

    # Objective function: Maximize total predicted points
    squad_prob += pulp.lpSum([player_vars[player.player_ID] * player.predicted_points for player in players])
    
    # Constraints
    # 1. 15 players in the squad
    squad_prob += pulp.lpSum(player_vars.values()) == 15
    
    # 2. Squad composition: 2 GK, 5 DEF, 5 MID, 3 FWD
    squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in gk_predictions.itertuples()]) == 2
    squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in def_predictions.itertuples()]) == 5
    squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in mid_predictions.itertuples()]) == 5
    squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in fwd_predictions.itertuples()]) == 3
    
    # 3. Maximum 3 players from the same team
    for team in predictions_df['team'].unique():
        squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in predictions_df.itertuples() if player.team == team]) <= 3
    
    # 4. Total value constraint (1000 = 100.0 in FPL)
    squad_prob += pulp.lpSum([player_vars[player.player_ID] * player.value for player in players]) <= 1000
    
    # 5. At least 4 cheap players with value <= 50 - this is used to take the bench into account
    squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in players if player.value <= 50]) >= 4
    
    # Solve the optimization problem
    squad_prob.solve()

    # Extract the selected players
    selected_players = [player for player in players if pulp.value(player_vars[player.player_ID]) == 1]

    if not selected_players:
        print("No players selected. Check constraints and input data.")
    
    # choose starting 11 and bench
    starting_11, bench = choose_starting_11(selected_players)
    
    return starting_11, bench

def determine_best_free_hit_modified(df, classifiers, regressors, min_bench_points=0):
    """Determines best free hit team for following GW (i.e. does not take into account current team).
    starting 11 has maximum predicted points subject to following constraints:
        - squad is made from 2 GK, 5 DEF, 5 MID, 3 FWD
        - starting 11 has 1 GK, at least 3 DEF, at least 2 MID, at least 1 FWD
        - Max 3 players from same team
        - Total value <= 1000
        - predicted points of bench >= min_bench_points
    Returns two lists of named tuples - starting 11 and bench """
    
    predictions = predict_next_x_gws(df, classifiers, regressors, 1)
    predictions_df = pd.DataFrame(predictions)

    gk_predictions = predictions_df[predictions_df['position'] == 'GK'].sort_values(by='predicted_points', ascending=False)
    def_predictions = predictions_df[predictions_df['position'] == 'DEF'].sort_values(by='predicted_points', ascending=False)
    mid_predictions = predictions_df[predictions_df['position'] == 'MID'].sort_values(by='predicted_points', ascending=False)
    fwd_predictions = predictions_df[predictions_df['position'] == 'FWD'].sort_values(by='predicted_points', ascending=False)

    squad_prob = pulp.LpProblem("BestFreeHitTeam", pulp.LpMaximize)

    # Create decision variables for selecting players
    players = list(predictions_df.itertuples())
    player_vars = {player.player_ID: pulp.LpVariable(f'player_{player.player_ID}', cat='Binary') for player in players}
    starting_11_vars = {player.player_ID: pulp.LpVariable(f'starting11_player_{player.player_ID}', cat='Binary') for player in players}
    bench_vars = {player.player_ID: pulp.LpVariable(f'bench_player_{player.player_ID}', cat='Binary') for player in players}

    # Objective function: Maximize total predicted points for starting 11
    squad_prob += pulp.lpSum([starting_11_vars[player.player_ID] * player.predicted_points for player in players])
    
    # Constraints

    # 15 players in the squad and 11 in starting 11
    squad_prob += pulp.lpSum(player_vars.values()) == 15
    squad_prob += pulp.lpSum(starting_11_vars.values()) == 11

    # If player in starting 11, must be in squad
    # If player in squad, must be in starting 11 or bench
    for player in players:
        squad_prob += starting_11_vars[player.player_ID] <= player_vars[player.player_ID]
        squad_prob += player_vars[player.player_ID] == starting_11_vars[player.player_ID] + bench_vars[player.player_ID]
    
    # Squad composition: 2 GK, 5 DEF, 5 MID, 3 FWD
    squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in gk_predictions.itertuples()]) == 2
    squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in def_predictions.itertuples()]) == 5
    squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in mid_predictions.itertuples()]) == 5
    squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in fwd_predictions.itertuples()]) == 3
    
    # Maximum 3 players from the same team
    for team in predictions_df['team'].unique():
        squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in predictions_df.itertuples() if player.team == team]) <= 3
    
    # Total value constraint (1000 = 100.0 in FPL)
    squad_prob += pulp.lpSum([player_vars[player.player_ID] * player.value for player in players]) <= 1000

    # Starting 11 composition
    squad_prob += pulp.lpSum([starting_11_vars[player.player_ID] for player in gk_predictions.itertuples()]) == 1
    squad_prob += pulp.lpSum([starting_11_vars[player.player_ID] for player in def_predictions.itertuples()]) >= 3
    squad_prob += pulp.lpSum([starting_11_vars[player.player_ID] for player in mid_predictions.itertuples()]) >= 2
    squad_prob += pulp.lpSum([starting_11_vars[player.player_ID] for player in fwd_predictions.itertuples()]) >= 1

    # Bench composition
    squad_prob += pulp.lpSum([bench_vars[player.player_ID] for player in gk_predictions.itertuples()]) == 1
    squad_prob += pulp.lpSum([bench_vars[player.player_ID] for player in def_predictions.itertuples()]) >= 0
    squad_prob += pulp.lpSum([bench_vars[player.player_ID] for player in mid_predictions.itertuples()]) >= 0
    squad_prob += pulp.lpSum([bench_vars[player.player_ID] for player in fwd_predictions.itertuples()]) >= 0
    
    # want bench players to still be quite good - implement variable constraint as to how many points bench must be predicted
    squad_prob += pulp.lpSum([bench_vars[player.player_ID] * player.predicted_points for player in players]) >= min_bench_points
    
    # Solve the optimization problem
    squad_prob.solve()

    # Extract the selected players
    starting_11 = [player for player in players if pulp.value(starting_11_vars[player.player_ID]) == 1]
    bench = [player for player in players if pulp.value(bench_vars[player.player_ID]) == 1]

    position_order = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
    starting_11_sorted = sorted(starting_11, key=lambda player: position_order[player.position])
    bench_sorted = sort_subs(bench)

    return starting_11_sorted, bench_sorted

# Both following functions determine best team for next x GWs based on current team
# Fist one maximises total team points but requires 4 cheap players
# Second one maximises points of starting 11 but requires bench predicted points to be larger than certain (user inputted) value

def best_team_next_x_gws(df, classifiers, regressors, x, current_team, current_team_value, bank, free_transfers=0):
    """Determines best team for the next x gws based on the current team
    """
    predictions_df = pd.DataFrame(predict_next_x_gws(df, classifiers, regressors, x))

    gk_predictions = predictions_df[predictions_df['position'] == 'GK'].sort_values(by='predicted_points', ascending=False)
    def_predictions = predictions_df[predictions_df['position'] == 'DEF'].sort_values(by='predicted_points', ascending=False)
    mid_predictions = predictions_df[predictions_df['position'] == 'MID'].sort_values(by='predicted_points', ascending=False)
    fwd_predictions = predictions_df[predictions_df['position'] == 'FWD'].sort_values(by='predicted_points', ascending=False)

    squad_prob = pulp.LpProblem("BestTeam", pulp.LpMaximize)

    # Create decision variables for selecting players
    players = list(predictions_df.itertuples())
    player_vars = {player.player_ID: pulp.LpVariable(f'player_{player.player_ID}', cat='Binary') for player in players}

    # Objective function: Maximize total predicted points
    team_predicted_points = pulp.lpSum([player_vars[player.player_ID] * player.predicted_points for player in players])
    squad_prob += team_predicted_points

    # Constraints
    # 1. 15 players in the squad
    squad_prob += pulp.lpSum(player_vars.values()) == 15
    
    # 2. Squad composition: 2 GK, 5 DEF, 5 MID, 3 FWD
    squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in gk_predictions.itertuples()]) == 2
    squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in def_predictions.itertuples()]) == 5
    squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in mid_predictions.itertuples()]) == 5
    squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in fwd_predictions.itertuples()]) == 3
    
    # 3. Maximum 3 players from the same team
    for team in predictions_df['team'].unique():
        squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in predictions_df.itertuples() if player.team == team]) <= 3

    current_team_vars = {player.player_ID : player_vars[player.player_ID] for player in current_team}
    current_team_predicted_points = pulp.lpSum([player.predicted_points for player in current_team])

    # 4. cost of new team <= selling price of current team + bank
    new_team_value = pulp.lpSum([player_vars[player.player_ID]*player.value for player in players])
    squad_prob += new_team_value <= current_team_value + bank

    # 5. Account for free transfers and cost transfers
    number_transfers = pulp.lpSum([player_vars[player.player_ID] for player in players]) - pulp.lpSum([current_team_vars[player.player_ID] for player in current_team])
    number_cost_transfers = max(number_transfers - free_transfers, 0)
    # Fpl transfers (after free transfers used) cost 4 points
    # Therefore, for a (non-free) transfer to be worth doing it must improve the expected points be more than 4
    # However, we have been maximising squad points not starting 11 points (tricky to set up problem to maximise starting 11 as still want bench players to have high predicted points)
    # to take this into account, we increase point threshold for a transfer to 6 points
    squad_prob += team_predicted_points - current_team_predicted_points >= 6*number_cost_transfers
    
    # 6. At least 4 cheap players with value <= 50 - this is used to take the bench into account
    squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in players if player.value <= 50]) >= 4

    # Solve the optimization problem
    squad_prob.solve()

    # Extract the selected players
    selected_players = [player for player in players if pulp.value(player_vars[player.player_ID]) == 1]

    if not selected_players:
        print("No players selected. Check constraints and input data.")
    
    # choose starting 11 and bench
    starting_11, bench = choose_starting_11(selected_players)

    print(f'number of transfers: {pulp.value(number_transfers)}')
    print(f'number of cost transfers: {pulp.value(number_cost_transfers)}')
    print(f'new_team_predicted_points: {pulp.value(team_predicted_points)}')
    print(f'current_team_predicted_points: {pulp.value(current_team_predicted_points)}')
    print(f'new_team_value: {pulp.value(new_team_value)}')
    
    return starting_11, bench

def best_team_next_x_gws_modified(df, classifiers, regressors, x, current_team, current_team_value, bank, free_transfers=0, min_bench_points=0):
    """Determines best team for the next x gws based on the current team
    """
    predictions_df = pd.DataFrame(predict_next_x_gws(df, classifiers, regressors, x))

    gk_predictions = predictions_df[predictions_df['position'] == 'GK'].sort_values(by='predicted_points', ascending=False)
    def_predictions = predictions_df[predictions_df['position'] == 'DEF'].sort_values(by='predicted_points', ascending=False)
    mid_predictions = predictions_df[predictions_df['position'] == 'MID'].sort_values(by='predicted_points', ascending=False)
    fwd_predictions = predictions_df[predictions_df['position'] == 'FWD'].sort_values(by='predicted_points', ascending=False)

    squad_prob = pulp.LpProblem("BestTeam", pulp.LpMaximize)

    # Create decision variables for selecting players
    players = list(predictions_df.itertuples())
    player_vars = {player.player_ID: pulp.LpVariable(f'player_{player.player_ID}', cat='Binary') for player in players}
    starting_11_vars = {player.player_ID: pulp.LpVariable(f'starting11_player_{player.player_ID}', cat='Binary') for player in players}
    bench_vars = {player.player_ID: pulp.LpVariable(f'bench_player_{player.player_ID}', cat='Binary') for player in players}
    current_team_vars = {player.player_ID : player_vars[player.player_ID] for player in current_team}
    current_starting_11, current_bench = choose_starting_11(current_team)
    current_starting_11_predicted_points = pulp.lpSum([player.predicted_points for player in current_starting_11])

    starting_11_predicted_points = pulp.lpSum([starting_11_vars[player.player_ID] * player.predicted_points for player in players])

    # Constraints
    # 15 players in the squad, 11 in starting 11
    squad_prob += pulp.lpSum(player_vars.values()) == 15
    squad_prob += pulp.lpSum(starting_11_vars.values()) == 11
    
    # Squad composition: 2 GK, 5 DEF, 5 MID, 3 FWD
    squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in gk_predictions.itertuples()]) == 2
    squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in def_predictions.itertuples()]) == 5
    squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in mid_predictions.itertuples()]) == 5
    squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in fwd_predictions.itertuples()]) == 3

    # Starting 11 composition
    squad_prob += pulp.lpSum([starting_11_vars[player.player_ID] for player in gk_predictions.itertuples()]) == 1
    squad_prob += pulp.lpSum([starting_11_vars[player.player_ID] for player in def_predictions.itertuples()]) >= 3
    squad_prob += pulp.lpSum([starting_11_vars[player.player_ID] for player in mid_predictions.itertuples()]) >= 2
    squad_prob += pulp.lpSum([starting_11_vars[player.player_ID] for player in fwd_predictions.itertuples()]) >= 1

    # Bench composition
    squad_prob += pulp.lpSum([bench_vars[player.player_ID] for player in gk_predictions.itertuples()]) == 1
    squad_prob += pulp.lpSum([bench_vars[player.player_ID] for player in def_predictions.itertuples()]) >= 0
    squad_prob += pulp.lpSum([bench_vars[player.player_ID] for player in mid_predictions.itertuples()]) >= 0
    squad_prob += pulp.lpSum([bench_vars[player.player_ID] for player in fwd_predictions.itertuples()]) >= 0

    # If player in starting 11, must be in squad
    # If player in squad, must be in starting 11 or bench
    for player in players:
        squad_prob += starting_11_vars[player.player_ID] <= player_vars[player.player_ID]
        squad_prob += player_vars[player.player_ID] == starting_11_vars[player.player_ID] + bench_vars[player.player_ID]

    # Maximum 3 players from the same team
    for team in predictions_df['team'].unique():
        squad_prob += pulp.lpSum([player_vars[player.player_ID] for player in predictions_df.itertuples() if player.team == team]) <= 3

    # Cost of new team <= selling price of current team + bank
    new_team_value = pulp.lpSum([player_vars[player.player_ID]*player.value for player in players])
    squad_prob += new_team_value <= current_team_value + bank

    # Account for free transfers and cost transfers
    number_transfers = pulp.lpSum([player_vars[player.player_ID] for player in players]) - pulp.lpSum([current_team_vars[player.player_ID] for player in current_team])
    number_cost_transfers = pulp.LpVariable('number_cost_transfers', lowBound=0, cat='Continuous')
    
    squad_prob += number_cost_transfers >= number_transfers - free_transfers
    squad_prob += number_cost_transfers <= number_transfers - free_transfers

    squad_prob += starting_11_predicted_points - current_starting_11_predicted_points >= 4*number_cost_transfers
    
    # Bench predicted points >= min_bench_points
    bench_predicted_points = pulp.lpSum([bench_vars[player.player_ID]*player.predicted_points for player in players])
    squad_prob += bench_predicted_points >= min_bench_points

    predicted_points_score = starting_11_predicted_points - 4*number_cost_transfers

    # Objective function: Maximize total predicted points of starting 11
    squad_prob += starting_11_predicted_points - 4*number_cost_transfers

    # Solve the optimization problem
    squad_prob.solve()

    # Extract the selected players
    starting_11 = [player for player in players if pulp.value(starting_11_vars[player.player_ID]) == 1]
    bench = [player for player in players if pulp.value(bench_vars[player.player_ID]) == 1]

    position_order = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
    starting_11_sorted = sorted(starting_11, key=lambda player: position_order[player.position])
    bench_sorted = sort_subs(bench)

    print(f'new_starting_11_predicted_points: {pulp.value(starting_11_predicted_points)}')
    print(f'current_starting_11_predicted_points: {pulp.value(current_starting_11_predicted_points)}')
    print(f'number of transfers: {pulp.value(number_transfers)}')
    print(f'number of cost transfers: {pulp.value(number_cost_transfers)}')
    print(f'GW predicted score: {pulp.value(predicted_points_score)}')

    return starting_11_sorted, bench_sorted, pulp.value(number_cost_transfers), pulp.value(predicted_points_score), pulp.value(bench_predicted_points)

# Create transfers object using current squad and new squad

def create_transfers_object(team_info, current_team, current_team_ids, new_team, new_team_ids, next_GW, use_wildcard=False):
    """Creates a transfers object using the current and new squads.
    - current_team and new_team are both lists of named tuples"""

    players_in = [player for player in new_team if player.player_ID not in current_team_ids]
    players_out = [player for player in current_team if player.player_ID not in new_team_ids]

    transfer_object = {'entry': team_id, 'event': next_GW, 'transfers': [], 'chip': 'wildcard' if use_wildcard else None}

    # sort by position
    position_order = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
    players_out = sorted(players_out, key=lambda player: position_order[player.position])
    players_in = sorted(players_in, key=lambda player: position_order[player.position])

    for i in range(len(players_in)):
        selling_price = web_service.get_selling_price(team_info, players_out[i].player_ID)
        purchase_price = web_service.get_player_purchase_cost(players_in[i].player_ID)

        transfer_object['transfers'].append({'element_in': players_in[i].player_ID, 'purchase_price': purchase_price,
                                            'element_out': players_out[i].player_ID, 'selling_price': selling_price})
        
    return transfer_object

# Functions to determine when to use chips

def determine_chip_use(team_info, starting_11, predicted_score, bench_predicted_points, captain_id, next_GW):
    """Determines which chips to use for current GW"""

    wildcard_availability = web_service.get_wildcard_availability(team_info)
    wildcard_number = web_service.get_wildcard_number(team_info)
    freehit_availability = web_service.get_freehit_availability(team_info)
    benchboost_availability = web_service.get_benchboost_availability(team_info)
    triplecaptain_availability = web_service.get_triplecaptain_availability(team_info)
    any_chip_in_use = web_service.chip_in_use(team_info)

    captain_predicted_points = 0
    for player in starting_11:
        if player.player_ID == captain_id:
            captain_predicted_points = player.predicted_points

    chip_object = None
    
    if not any_chip_in_use and freehit_availability:
        if predicted_score < 30 or next_GW == 38:
            any_chip_in_use = True
            chip_object = 'freehit'

    if not any_chip_in_use and wildcard_availability:
        if wildcard_number == 1:
            if predicted_score < 35 or next_GW == 19:
                any_chip_in_use = True
                chip_object = 'wildcard'

        elif wildcard_number == 2:
            if predicted_score < 35 or next_GW == 38:
                any_chip_in_use = True
                chip_object = 'wildcard'


    if not any_chip_in_use and benchboost_availability:
        if bench_predicted_points > 16 or next_GW == 38:
            any_chip_in_use = True
            chip_object = 'bboost'


    if not any_chip_in_use and triplecaptain_availability:
        if captain_predicted_points > 12 or next_GW == 38:
            any_chip_in_use = False
            chip_object = '3xc'

    return chip_object

    


    



















