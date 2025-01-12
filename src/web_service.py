import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
team_id = os.getenv('FPL_TEAM_ID')

"Functions used to get data from fpl website"

def login_fpl():
    """Login to official FPL website and recover team information url"""

    target_url = f'https://fantasy.premierleague.com/api/my-team/{team_id}/'

    print(f'Log into the FPL website and go to the following link: {target_url} ')
    
    team_info = input('copy the response of the url: ')

    return json.loads(team_info)

def get_current_team_ids(team_info):
    """Returns the player_IDs of the current team"""
    team = []
    for dict in team_info['picks']:
        team.append(dict['element'])

    return team

def get_number_free_transfers(team_info):
    """Returns current number of free transfers"""
    limit = team_info['transfers']['limit']
    made = team_info['transfers']['made']

    free_transfers = limit-made

    return max(0, free_transfers)

def get_bank_value(team_info):
    """Returns value available in bank"""

    return team_info['transfers']['bank']

def get_squad_value(team_info):
    """Returns total value of current squad"""

    return team_info['transfers']['value']

def get_wildcard_availability(team_info):
    "Checks if wildcard is available"

    for dict in team_info['chips']:
        if dict['name'] == 'wildcard':
            if dict['status_for_entry'] == 'available':
                return True
            else:
                return False
    return None

def get_wildcard_number(team_info):
    "Gets wildcard number (1 or 2)"

    for dict in team_info['chips']:
        if dict['name'] == 'wildcard':
            return dict['number']
    return None

def get_freehit_availability(team_info):
    "Checks if free hit is available"

    for dict in team_info['chips']:
        if dict['name'] == 'freehit':
            if dict['status_for_entry'] == 'available':
                return True
            else:
                return False      
    return None

def get_benchboost_availability(team_info):
    "Checks if bench boost is available"

    for dict in team_info['chips']:
        if dict['name'] == 'bboost':
            if dict['status_for_entry'] == 'available':
                return True
            else:
                return False
    return None

def get_triplecaptain_availability(team_info):
    "Checks if bench boost is available"

    for dict in team_info['chips']:
        if dict['name'] == '3xc':
            if dict['status_for_entry'] == 'available':
                return True
            else:
                return False
    return None

def get_selling_price(team_info, player_ID):
    "Gets the selling price of a player in the current squad"
    for dict in team_info['picks']:
        if dict['element'] == player_ID:
            return dict['selling_price']
    return None

def get_injury_status(player_ID):
    """Returns injury/suspension status of a player
    - "a": available
    - "i": injured
    - "d": doubtful
    - "u": unavailable
    - "s": suspended 
    - "l": loan
    - "n": not in squad
    """


    bootstrap_url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    bootstrap_response = requests.get(bootstrap_url)
    data = bootstrap_response.json()

    for player in data['elements']:
        player_id = player['id']
        
        if player_id == player_ID:
            return player['status']
    
    return None

def extract_cookies(session):
    """Manually extracts cookies from fpl website using EditThisCookie extension for Chrome"""

    url = 'https://fantasy.premierleague.com/'
    print(f"Make sure you're logged in to the FPL website and go to following link: {url}")
    cookies = input('Click on extensions and then EditThisCookie. Copy the cookies to the clipboard and paste here (make sure to paste as one line.): ')

    cookies_list = json.loads(cookies)  # convert pasted cookie string into list of dictionaries
    cookies_dict = {cookie['name']:cookie['value'] for cookie in cookies_list}
    session.cookies.update(cookies_dict)    # update session with cookies

def set_lineup(session, team_id, new_lineup, captain_id, vice_captain_id):
    """
    Updates the starting lineup, captain, and vice-captain for the FPL team.

    new_linup: list of dictionaries {'id': player_ID, 'position': position in lineup (number from 1-15)}
    """
    
    url = f'https://fantasy.premierleague.com/api/my-team/{team_id}/'
    
    # Build the payload with the new lineup
    payload = {"picks": [{"element": player["id"], "position": player["position"],"is_captain": player["id"] == captain_id,
                          "is_vice_captain": player["id"] == vice_captain_id} for player in new_lineup]}

    # Headers to include the CSRF token (stored in the session's cookies)
    headers = {'Content-Type': 'application/json','X-CSRFToken': session.cookies.get('csrftoken'),'Referer': url }

    # Make the POST request to update the lineup
    response = session.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        print("Lineup updated successfully!")
    else:
        print("Failed to update lineup:", response.text)

def get_current_GW():
    "Get current GW from FPL API"

    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = requests.get(url)
    static = response.json()

    current_GW = None
    for event in static['events']:
        if event['is_current'] == True:
            current_GW = event['id']
            break

    return current_GW

def get_next_GW():
    "Get next GW from FPL API"

    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = requests.get(url)
    static = response.json()

    next_GW = None
    for event in static['events']:
        if event['is_next'] == True:
            next_GW = event['id']
            break

    return next_GW

def get_deadline(next_GW):
    "Get deadline for submitting transfers"

    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = requests.get(url)
    static = response.json()

    deadline = None
    for event in static['events']:
        if event['id'] == next_GW:
            deadline = event['deadline_time']
            break
    
    return deadline

def get_player_purchase_cost(id):
    "Get the current purchase cost of a player"

    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = requests.get(url)
    static = response.json()

    purchase_cost = None
    for player in static['elements']:
        if player['id'] == id:
            purchase_cost = player['now_cost']
            break
    return purchase_cost

def make_transfers(session, transfer_object):
    "Make transfers on fpl website"

    url = 'https://fantasy.premierleague.com/api/transfers/'
    headers = {'Content-Type': 'application/json','X-CSRFToken': session.cookies.get('csrftoken'),'Referer': url }

    if len(transfer_object['transfers']) > 0:
        response = session.post(url, headers=headers, json=transfer_object)

        if response.status_code == 200:
            print("Made transfers successfully!")
        else:
            print("Failed to make transfers:", response.text)

    else:
        print("No transfers needed!")

def chip_in_use(team_info):
    "Checks if any chip is being used"

    for chip in team_info['chips']:
        if chip['is_pending'] == True:
            return True
        
    return False

def activate_chips_set_lineup(session, team_info, chip_object, team_id, new_lineup, captain_id, vice_captain_id):
    "Activates chips and sets lineup for current GW on FPL website"

    url = f'https://fantasy.premierleague.com/api/my-team/{team_id}/'

    headers = {'Content-Type': 'application/json','X-CSRFToken': session.cookies.get('csrftoken'),'Referer': url }

    payload = {"chip": chip_object, "picks": [{"element": player["id"], "position": player["position"],"is_captain": player["id"] == captain_id,
                          "is_vice_captain": player["id"] == vice_captain_id} for player in new_lineup]}

    response = session.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        print("chips activated and lineup updated successfully!")

    elif response.status_code != 200 and json.loads(response.text) == team_info:
        print("No changes needed")
        
    else:
        print("Error:", response.text)



