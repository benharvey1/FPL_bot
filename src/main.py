"""
The main programme

"""

import src.web_service as web_service
import src.team_optimisation as team_optimisation
import src.data_statistics as data_statistics
import pandas as pd
import joblib
import os
import requests
from datetime import datetime
import pytz
from dotenv import load_dotenv

load_dotenv()
team_id = os.getenv('FPL_TEAM_ID')

"""
1. Retrieve data for the current season from the FPL API

"""

# team market values for 2024-25 season
team_market_values = {1: 0.9069767441860465, 2: 0.4930232558139535, 3: 0.2875968992248062, 4: 0.33565891472868215, 5: 0.4186046511627907, 
                      6: 0.8604651162790699, 7: 0.3372093023255814, 8: 0.25116279069767444, 9: 0.2170542635658915, 10: 0.09534883720930232, 
                      11: 0.1558139534883721, 12: 0.7178294573643411, 13: 1.0, 14: 0.6922480620155038, 15: 0.5116279069767442, 
                      16: 0.29224806201550385, 17: 0.20852713178294574, 18: 0.6333333333333333, 19: 0.4356589147286822, 20: 0.2806201550387597}

# team stats from 2023-24 season
team_stats_previous = {1 : {'points': 2369, 'conceded': 1191}, 2: {'points': 1682, 'conceded': 1549}, 3: {'points': 1672, 'conceded': 1687},
                        4: {'points': 1464, 'conceded': 1847}, 5: {'points': 1492, 'conceded': 1694}, 6: {'points': 1888, 'conceded': 1636},
                        7: {'points': 1712, 'conceded': 1586}, 8: {'points': 1620, 'conceded': 1776}, 9: {'points': 1526, 'conceded': 1775},
                        10: {'points': 1030, 'conceded': 2242}, 11: {'points': 1270, 'conceded': 1954}, 12: {'points': 2104, 'conceded': 1304}, 
                        13: {'points': 2324, 'conceded': 1158}, 14 : {'points': 1518, 'conceded': 1593},
                        15: {'points': 1854, 'conceded': 1577}, 16: {'points': 1353, 'conceded': 1805}, 17: {'points': 1270, 'conceded': 1954}, 
                        18: {'points': 1781, 'conceded': 1486}, 19: {'points': 1436, 'conceded': 1715}, 20: {'points': 1399, 'conceded': 1795}}

current_data = 'data/current_season_dataset.csv'

if os.path.exists(current_data):
    os.remove(current_data)

# Create data set for the 2024-25 season using fpl api

actual_data = data_statistics.fetch_api_data(team_market_values, team_stats_previous)
actual_data = data_statistics.add_game_number_column(actual_data)
actual_data = data_statistics.total_GW_points(actual_data)

# Identify players with missing stats for previous season
# Find players with same value and use average of their stats

missing_points = actual_data['total_points_last_season'].isna()
missing_mins = actual_data['total_mins_last_season'].isna()

actual_data.loc[missing_points, 'total_points_last_season'] = actual_data[missing_points].apply(lambda row: data_statistics.closest_players_value(row, actual_data, 'total_points_last_season'), axis=1)
actual_data.loc[missing_mins, 'total_mins_last_season'] = actual_data[missing_mins].apply(lambda row: data_statistics.closest_players_value(row, actual_data, 'total_mins_last_season'), axis=1)

# Add features required to run models

actual_data = data_statistics.add_player_stats(actual_data)
actual_data = data_statistics.add_team_stats(actual_data)
actual_data = data_statistics.add_team_conceded_stats(actual_data)
actual_data = data_statistics.add_opponent_stats(actual_data)

actual_data.to_csv(current_data, index=False)

"""
2. Load the models

"""

# Load the classifiers  for appearance prediction
GK_appearance_classifier = joblib.load('models/GK_appearance_classifier.pkl')
DEF_appearance_classifier = joblib.load('models/DEF_appearance_classifier.pkl')
MID_appearance_classifier = joblib.load('models/MID_appearance_classifier.pkl')
FWD_appearance_classifier = joblib.load('models/FWD_appearance_classifier.pkl')

# Load the regressors for points prediction
GK_lgbm = joblib.load('models/GK_lgbm.pkl')
DEF_lgbm = joblib.load('models/DEF_lgbm.pkl')
MID_lgbm = joblib.load('models/MID_lgbm.pkl')
FWD_lgbm = joblib.load('models/FWD_lgbm.pkl')

appearance_classifiers = {'GK': GK_appearance_classifier, 'DEF': DEF_appearance_classifier, 'MID': MID_appearance_classifier, 'FWD': FWD_appearance_classifier}
point_predictors = {'GK': GK_lgbm, 'DEF': DEF_lgbm, 'MID': MID_lgbm, 'FWD': FWD_lgbm}

"""
3. Log in to fpl and get current squad info

"""

team_info = web_service.login_fpl()
current_team_ids = web_service.get_current_team_ids(team_info)
current_team = team_optimisation.current_team_tuples(actual_data, appearance_classifiers, point_predictors, 1, current_team_ids)
bank = web_service.get_bank_value(team_info)
current_squad_value = web_service.get_squad_value(team_info)
free_transfers = web_service.get_number_free_transfers(team_info)
next_GW = web_service.get_next_GW()
deadline = web_service.get_deadline(next_GW)

"""
4. Determine best squad for upcoming x GWs and what chips to use

"""
min_bench_points = 8   # Set this variable yourself. Gives the minimum required predicted points for the bench
                        # For high risk strategy choose low predicted points, for low risk choose high predicted points

x = 1

best_starting, best_bench, cost_transfers, predicted_points, bench_pred_points = team_optimisation.best_team_next_x_gws_modified(actual_data, appearance_classifiers, point_predictors, x, current_team, current_squad_value, bank, free_transfers, min_bench_points)
captain, vice = team_optimisation.choose_captains(best_starting)

new_team = best_starting + best_bench
new_team_ids = [player.player_ID for player in new_team]
new_lineup = []
for i, player in enumerate(best_starting):
    new_lineup.append({'id':player.player_ID, 'position': i+1})

for i, player in enumerate(best_bench):
    new_lineup.append({'id':player.player_ID, 'position': i+12})

print(f'current squad: {[player.name for player in current_team]}')
print(f'new squad: {[player.name for player in new_team]}')

chip_object = team_optimisation.determine_chip_use(team_info, best_starting, predicted_points, bench_pred_points, captain, next_GW)

print(f'Best chip to use is: {chip_object}')

use_wildcard = False

if chip_object == 'freehit':
    best_starting, best_bench = team_optimisation.determine_best_free_hit_modified(actual_data, appearance_classifiers, point_predictors, min_bench_points)
    captain, vice = team_optimisation.choose_captains(best_starting)
    new_team = best_starting + best_bench
    new_team_ids = [player.player_ID for player in new_team]
    new_lineup = []
    for i, player in enumerate(best_starting):
        new_lineup.append({'id':player.player_ID, 'position': i+1})

    for i, player in enumerate(best_bench):
        new_lineup.append({'id':player.player_ID, 'position': i+12})

if chip_object == 'wildcard':
    use_wildcard = True
    best_starting, best_bench, cost_transfers, predicted_points, bench_pred_points = team_optimisation.best_team_next_x_gws_modified(actual_data, appearance_classifiers, point_predictors, x, current_team, current_squad_value, bank, free_transfers=100, min_bench_points=min_bench_points)
    captain, vice = team_optimisation.choose_captains(best_starting)
    new_team = best_starting + best_bench
    new_team_ids = [player.player_ID for player in new_team]
    new_lineup = []
    for i, player in enumerate(best_starting):
        new_lineup.append({'id':player.player_ID, 'position': i+1})

    for i, player in enumerate(best_bench):
        new_lineup.append({'id':player.player_ID, 'position': i+12})
    
    print(f'current squad: {[player.name for player in current_team]}')
    print(f'new squad: {[player.name for player in new_team]}')


transfers_object = team_optimisation.create_transfers_object(team_info, current_team, current_team_ids, new_team, new_team_ids, next_GW, use_wildcard)

print(f'transfers: {transfers_object['transfers']}')

"""
5. Update squad on FPL website

"""
if input("Do you want to make these changes? (y/n): ").strip().lower() == 'y':

    deadline_converted = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    uk_tz = pytz.timezone('Europe/London')
    now_utc = datetime.now(pytz.utc)
    now_uk = now_utc.astimezone(uk_tz)

    if now_uk < deadline_converted:
        
        session = requests.Session()
        web_service.extract_cookies(session)
        web_service.make_transfers(session, transfers_object)
        web_service.activate_chips_set_lineup(session, team_info, chip_object, team_id, new_lineup, captain, vice)





