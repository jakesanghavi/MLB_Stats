import requests
import pandas as pd
import numpy as np
import time
import ast
from itertools import chain
from datetime import datetime

# Set the year you want to scrape plays from
year = "2025"
today = datetime.today().date()

# Set the minimum date you want to start scraping from
# If you've run this and want to run it again, don't want to re-get old games
min_date = datetime(2025, 3, 18).date()

header_data = {
    'Connection': 'keep-alive',
    # 'Accept': 'application/json, text/plain, */*',
    'Accept': 'application/json',
    'x-nba-stats-token': 'true',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/79.0.3945.130 Safari/537.36',
    'x-nba-stats-origin': 'stats',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Referer': 'https://statsapi.mlb.com/',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Teams dict
TEAMS = {"Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
         "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS", "Cincinnati Reds": "CIN",
         "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
         "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
         "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY",
         "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
         "San Diego Padres": "SD", "San Francisco Giants": "SF", "Seattle Mariners": "SEA",
         "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
         "Washington Nationals": "WSH", "Athletics": "ATH"}


# Scrape a game from just a gameID
def game_scraper(gameID):
    game_url = f"https://statsapi.mlb.com/api/v1.1/game/{gameID}/feed/live"
    # https: // statsapi.mlb.com / api / v1 / game / 747053 / content - this for mp4s
    r = None
    try:
        r = requests.get(game_url, headers=header_data, timeout=20)
    except requests.exceptions.JSONDecodeError as rerr:
        print(rerr)
    return r.json()


# Quickly flatten a list (for JSON data)
def fast_flatten(input_list):
    return list(chain.from_iterable(input_list))


# Wrapper function for fast_flatten above
def flattenizer(dfList, dict_df, col_names):
    for col in col_names:
        extracted = (frame[col] for frame in dfList)

        # Flatten and save to df_dict
        dict_df[col] = fast_flatten(extracted)

    return dict_df


# Read in your schedule list
schedule = pd.read_csv(f"{year}_schedule.csv")
schedule['teams'] = schedule['teams'].apply(ast.literal_eval)
schedule['status'] = schedule['status'].apply(ast.literal_eval)

rlist = []
x = 1

# Iterate over all games
for game in schedule['gamePk']:

    # Only get games that have completed
    if schedule.loc[schedule['gamePk'] == game]['status'].iloc[0]['detailedState'] != 'Final':
        continue
    out_list = []

    # Get team ABBRs for file naming later
    a_abbr = TEAMS[schedule.loc[schedule['gamePk'] == game]['teams'].iloc[0]['away']['team']['name']]
    h_abbr = TEAMS[schedule.loc[schedule['gamePk'] == game]['teams'].iloc[0]['home']['team']['name']]
    date = schedule.loc[schedule['gamePk'] == game]['officialDate'].iloc[0]
    full_date = schedule.loc[schedule['gamePk'] == game]['gameDate'].iloc[0]

    # If the game hasn't happened yet or you already have the data, pass by it
    if datetime.strptime(date, "%Y-%m-%d").date() >= today or datetime.strptime(date, "%Y-%m-%d").date() < min_date:
        continue

    print(x)

    # Sleep for a second to prevent getting bounced by the API
    time.sleep(1)

    # Get just the plays data from the game
    game_data = game_scraper(game)['liveData']['plays']['allPlays']
    gid = pd.DataFrame({'game_id': [game], 'away_team_abbr': [a_abbr], 'home_team_abbr': [h_abbr],
                        'date': [date], 'full_date': [full_date]})

    # Iterate over each plate appearance
    for pa in game_data:
        # Get various necessary data
        events = pa['playEvents']
        inning = pa['about']['inning']
        top_half = int(pa['about']['isTopInning'])
        ab_id = pa['atBatIndex']
        bid = pa['matchup']['batter']['id']
        pid = pa['matchup']['pitcher']['id']
        bh = pa.get('matchup', {}).get('batSide', {}).get('code', np.nan)
        ph = pa.get('matchup', {}).get('pitchHand', {}).get('code', np.nan)
        matchup_bio = pd.DataFrame(
            {'inning': [inning], 'top_of_inning': [top_half], 'ab_id': [ab_id], 'batter_id': [bid],
             'bat_side': [bh], 'pitcher_id': [pid], 'pitcher_hand': [ph]})

        event = pa['result']['event']
        event_type = pa['result']['eventType']
        desc = pa['result']['description']
        a_score = pa['result']['awayScore']
        h_score = pa['result']['homeScore']
        rbi = pa['result']['rbi']

        res_df = pd.DataFrame({'event_name': [event], 'event_type': [event_type], 'description': [desc],
                               'away_score': [a_score], 'home_score': [h_score], 'rbi': [rbi]})

        # Initalize runners for each base
        runners = [0, np.nan, 0, np.nan, 0, np.nan]

        for runner in pa['runners']:
            if runner['details']['movementReason'] is None:
                rlist.append('0')
            else:
                rlist.append(runner['details']['movementReason'])

            if runner['movement']['start'] is not None and len(runner['credits']) == 0:
                # If the runner changed bases during the AB, only care about their end base
                if runner['details']['movementReason'].startswith('r_stolen_base') or \
                        runner['details']['movementReason'].startswith('r_pickoff') or \
                        runner['details']['movementReason'].startswith('r_caught_stealing') or \
                        runner['details']['movementReason'].startswith('r_stolen_base') or \
                        runner['details']['movementReason'].startswith('r_defensive_indiff'):
                    runner['movement']['originBase'] = runner['movement']['end']
                if runner['movement']['originBase'] == '1B':
                    runners[0] = 1
                    runners[1] = runner['details']['runner']['id']
                elif runner['movement']['originBase'] == '2B':
                    runners[2] = 1
                    runners[3] = runner['details']['runner']['id']
                elif runner['movement']['originBase'] == '3B':
                    runners[4] = 1
                    runners[5] = runner['details']['runner']['id']

            # Debug any other movement reasons
            if runner['details']['movementReason'] == '0':
                print(f"0: {pa['result']['description']}")
            elif runner['details']['movementReason'] == 'r_hbr':
                print(f"r_hbr: {pa['result']['description']}")
            elif runner['details']['movementReason'] == 'r_interference':
                print(f"r_interference: {pa['result']['description']}")
            elif runner['details']['movementReason'] == 'r_out_appeal':
                print(f"r_out_appeal: {pa['result']['description']}")
            elif runner['details']['movementReason'] == 'r_rundown':
                print(f"r_rundown: {pa['result']['description']}")

        runner_df = pd.DataFrame(
            {'runner_on_1B_bool': [runners[0]], 'runner_on_1B_id': [runners[1]], 'runner_on_2B_bool':
                [runners[2]], 'runner_on_2B_id': [runners[3]], 'runner_on_3B_bool': [runners[4]],
             'runner_on_3B_id': [runners[5]]})

        # Iterate through all events of the at bat
        for event in events:
            d = event['details']
            try:
                h = int(d['isStrike']) if 'isStrike' in d else np.nan
                pitch_result = d['code']
                pitch_code = d['type']['code']
                pitch_results = pd.DataFrame(
                    {'strike': [h], 'detailed_pitch_outcome': [pitch_result], 'pitch_type': [pitch_code]})
                pdata = event.get('pitchData', {})
                play_id = event['playId']
                count = event['count']
                coordinates = pdata.get('coordinates', {})

                # Various safety checks for nulls to avoid crashing or omission
                pitch_stats = pd.DataFrame({
                    'start_speed': [pdata.get('startSpeed', np.nan)],
                    'end_speed': [pdata.get('endSpeed', np.nan)],
                    'pitch_coordinate_X': [coordinates.get('pX', np.nan)],
                    'pitch_coordinate_Z': [coordinates.get('pZ', np.nan)],
                    'balls': [event.get('count', {}).get('balls', np.nan)],
                    'strikes': [event.get('count', {}).get('strikes', np.nan)],
                    'outs': [event.get('count', {}).get('outs', np.nan)],
                    'playID': [event.get('playId', np.nan)]
                })

                breaks = pd.DataFrame({
                    'break_angle': [pdata.get('breaks', {}).get('breakAngle', np.nan)],
                    'break_length': [pdata.get('breaks', {}).get('breakLength', np.nan)],
                    'spin_rate': [pdata.get('breaks', {}).get('spinRate', np.nan)]
                })

                # Set default hit data and set it below
                hit_data = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                try:
                    hit_data[0] = event['hitData']['launchSpeed']
                    hit_data[1] = event['hitData']['launchAngle']
                    hit_data[2] = event['hitData']['totalDistance']
                    hit_data[3] = event['hitData']['trajectory']
                    hit_data[4] = event['hitData']['hardness']
                    if event['hitData']['coordinates'] is not None:
                        hit_data[5] = event['hitData']['coordinates']['coordX']
                        hit_data[6] = event['hitData']['coordinates']['coordY']
                except KeyError:
                    hit_data = hit_data

                hit_data_df = pd.DataFrame({'exit_velocity': [hit_data[0]], 'launch_angle': [hit_data[1]],
                                            'hit_distance': [hit_data[2]], 'hit_type': [hit_data[3]],
                                            'contact_hardness': [hit_data[4]], 'hit_location_X': [hit_data[5]],
                                            'hit_location_Y': [hit_data[6]]})

                # Concat all of our DFs sideways to get all data for each pitch!
                play_df = pd.concat([gid, matchup_bio, res_df, pitch_results, pitch_stats, breaks, runner_df,
                                     hit_data_df], axis=1)
                # Add the play to the total list
                out_list.append(play_df)
                del play_df
            # Have had issues debugging this final reason so for now just pass it
            except KeyError as k:
                # print(k)
                continue
    # Attempt to create a DF for each game now. If it crashes, try to see why.
    try:
        df_dict = dict.fromkeys(out_list[0].columns, [])
        df_dict = flattenizer(out_list, df_dict, out_list[0].columns)
        out_df = pd.DataFrame.from_dict(df_dict)[out_list[0].columns]
        # Write the CSV to an appropriately named file
        out_df.to_csv(f"Games/{year}/{date}_{a_abbr}_{h_abbr}_{game}.csv", index=False)
    except IndexError as i:
        print(i)
        print(date)
        print(gid)
    x += 1
