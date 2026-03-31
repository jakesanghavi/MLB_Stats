import pandas as pd
import requests
import json
import time
from pathlib import Path
import re
import datetime
from itertools import chain
import ast
import numpy as np


def safe_get(d, *keys):
    cur = d
    for k in keys:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return None
    return cur


def key_exists(d, key):
    if isinstance(d, dict):
        if key in d:
            return True
        return any(key_exists(v, key) for v in d.values())
    elif isinstance(d, list):
        return any(key_exists(i, key) for i in d)
    return False


def get_header_data():
    return {
        'Connection': 'keep-alive',
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


def get_team_map():
    return {"Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
            "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS", "Cincinnati Reds": "CIN",
            "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
            "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
            "Miami Marlins": "MIA",
            "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY",
            "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
            "San Diego Padres": "SD", "San Francisco Giants": "SF", "Seattle Mariners": "SEA",
            "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
            "Washington Nationals": "WSH", "Athletics": "ATH"}


def game_scraper(game_id):
    game_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    # https: // statsapi.mlb.com / api / v1 / game / 747053 / content - this for mp4s
    header_data = get_header_data()
    r = None
    try:
        r = requests.get(game_url, headers=header_data, timeout=20)
    except requests.exceptions.JSONDecodeError as rerr:
        print(rerr)
    return r.json()


def fast_flatten(input_list):
    return list(chain.from_iterable(input_list))


def flattenizer(df_list, dict_df, col_names):
    for col in col_names:
        extracted = (frame[col] for frame in df_list)

        # Flatten and save to df_dict
        dict_df[col] = fast_flatten(extracted)

    return dict_df


def save_file(data, directory, filename):
    cwd = Path.cwd()
    data_pack_dir = cwd / directory

    data_pack_dir.mkdir(parents=True, exist_ok=True)
    full_path = data_pack_dir / filename

    if isinstance(data, pd.DataFrame):
        if not str(full_path).lower().endswith('.csv'):
            full_path = Path(full_path).stem + '.csv'
        data.to_csv(full_path, index=False)
    elif isinstance(data, (dict, list)):
        if not str(full_path).lower().endswith('.json'):
            full_path = Path(full_path).stem + '.json'
        with open(full_path, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        raise TypeError("Unsupported data type. Must be .json or .csv (Pandas DataFrame).")


def get_schedule(year):
    start_date = f'{year}-03-18'
    end_date = f'{year}-10-02'
    sched_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_date}&endDate={end_date}"
    header_data = get_header_data()

    schedule = requests.get(sched_url, headers=header_data, timeout=100)
    resp = schedule.json()

    schedule = []

    for d in resp['dates']:
        for game in d['games']:
            if game['gameType'] != 'R':
                continue
            elif game['status']['detailedState'] == 'Postponed' or game['status']['detailedState'] == 'Suspended':
                continue
            else:
                schedule.append(game)

    df = pd.DataFrame(data=schedule)
    df = df.drop_duplicates(['gamePk'])

    out_dir = Path.cwd() / "DataPack" / str(year)
    out_file = f"{year}_schedule.csv"

    save_file(df, out_dir, out_file)


def get_pbp(year):
    schedule_path = Path.cwd() / "DataPack" / str(year) / f"{year}_schedule.csv"
    schedule = pd.read_csv(schedule_path)
    schedule['teams'] = schedule['teams'].apply(ast.literal_eval)
    schedule['status'] = schedule['status'].apply(ast.literal_eval)

    today = datetime.datetime.today().date()
    directory = Path.cwd() / "DataPack" / "Games" / str(year)
    directory.mkdir(parents=True, exist_ok=True)
    date_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2})')

    try:
        newest = max(
            datetime.datetime.strptime(m.group(1), "%Y-%m-%d").date()
            for f in directory.iterdir()
            if (m := date_pattern.match(f.name))
        )
        min_date = newest + datetime.timedelta(days=1)
    except ValueError:
        min_date = datetime.date(2023, 3, 25)

    TEAMS = get_team_map()

    rlist = []
    x = 1

    for game in schedule['gamePk']:
        if schedule.loc[schedule['gamePk'] == game]['status'].iloc[0]['detailedState'] != 'Final':
            continue
        out_list = []
        a_abbr = TEAMS[schedule.loc[schedule['gamePk'] == game]['teams'].iloc[0]['away']['team']['name']]
        h_abbr = TEAMS[schedule.loc[schedule['gamePk'] == game]['teams'].iloc[0]['home']['team']['name']]
        d = schedule.loc[schedule['gamePk'] == game]['officialDate'].iloc[0]
        full_date = schedule.loc[schedule['gamePk'] == game]['gameDate'].iloc[0]
        if datetime.datetime.strptime(d, "%Y-%m-%d").date() >= today or datetime.datetime.strptime(d, "%Y-%m-%d").date() < min_date:
            continue
        #
        # if x > 200:
        #     break
        time.sleep(1)
        game_data = game_scraper(game)['liveData']['plays']['allPlays']
        gid = pd.DataFrame({'game_id': [game], 'away_team_abbr': [a_abbr], 'home_team_abbr': [h_abbr],
                            'date': [d], 'full_date': [full_date]})
        print(x)
        print(game)
        for pa in game_data:
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

            runners = [0, np.nan, 0, np.nan, 0, np.nan]

            for runner in pa['runners']:
                if runner['details']['movementReason'] is None:
                    rlist.append('0')
                else:
                    rlist.append(runner['details']['movementReason'])

                if runner['movement']['start'] is not None and len(runner['credits']) == 0:
                    if runner['details']['movementReason'].startswith('r_stolen_base') or \
                            runner['details']['movementReason'].startswith('r_pickoff') or \
                            runner['details']['movementReason'].startswith('r_caught_stealing') or \
                            runner['details']['movementReason'].startswith('r_stolen_base') == 'r_defensive_indiff':
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

            for event in events:
                det = event['details']
                try:
                    # h = int(d['isStrike']) if 'isStrike' in d else np.nan
                    pitch_result = det['code']
                    h = 1 if pitch_result in ['C', 'D', 'E', 'F', 'L', 'M', 'O', 'S', 'T', 'W', 'X'] else 0
                    pitch_code = det['type']['code']
                    pitch_results = pd.DataFrame(
                        {'strike': [h], 'detailed_pitch_outcome': [pitch_result], 'pitch_type': [pitch_code]})
                    pdata = event.get('pitchData', {})
                    play_id = event['playId']
                    count = event['count']
                    coordinates = pdata.get('coordinates', {})

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
                        'break_vertical_induced': [pdata.get('breaks', {}).get('breakVerticalInduced', np.nan)],
                        'break_horizontal': [pdata.get('breaks', {}).get('breakHorizontal', np.nan)],
                        'spin_rate': [pdata.get('breaks', {}).get('spinRate', np.nan)]
                    })

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
                                                'hit_location_X': [hit_data[5]],
                                                'hit_location_Y': [hit_data[6]]})

                    play_df = pd.concat([gid, matchup_bio, res_df, pitch_results, pitch_stats, breaks, runner_df,
                                         hit_data_df], axis=1)
                    out_list.append(play_df)
                    del play_df
                except KeyError as k:
                    # print(k)
                    continue
        try:
            df_dict = dict.fromkeys(out_list[0].columns, [])
            df_dict = flattenizer(out_list, df_dict, out_list[0].columns)
            out_df = pd.DataFrame.from_dict(df_dict)[out_list[0].columns]
            out_dir = Path.cwd() / "DataPack" / "Games" / str(year)
            out_file = f"{d}_{a_abbr}_{h_abbr}_{game}.csv"
            save_file(out_df, out_dir, out_file)
        except IndexError as i:
            print(i)
            print(d)
            print(gid)
        x += 1


def concat_all(year):
    folder_path = Path.cwd() / "DataPack" / "Games" / str(year)
    all_files = list(folder_path.glob("*.csv"))

    dfs = []
    for filename in all_files:
        dfs.append(pd.read_csv(filename))

    combined_df = pd.concat(dfs, ignore_index=True)

    out_dir = Path.cwd() / "DataPack" / "PBP"
    out_file = f"{year}_full_pbp.csv"

    save_file(combined_df, out_dir, out_file)


def get_pitcher_bios(year):
    header_data = get_header_data()
    pd.options.mode.chained_assignment = None

    pbp_path = Path.cwd() / "DataPack" / "PBP" / f'{year}_full_pbp.csv'

    df = pd.read_csv(pbp_path)
    players = np.unique(df['pitcher_id'])

    def player_scraper(p_id, x):
        game_url = f"https://statsapi.mlb.com/api/v1/people/{p_id}"
        try:
            r = requests.get(game_url, headers=header_data, timeout=20)
            return r.json()
        except requests.exceptions.JSONDecodeError as rerr:
            print(rerr)
            print(p)
        except requests.exceptions.ReadTimeout as rerr:
            print(rerr)
            print(p)
            time.sleep(1.5)
            player_scraper(p_id, x + 1)

    final = pd.DataFrame(columns=['id', 'fullName', 'primaryNumber', 'birthDate', 'birthStateProvince', 'birthCountry',
                                  'height', 'weight', 'draftYear', 'strikeZoneTop', 'strikeZoneBottom', 'position',
                                  'batSide', 'pitchHand', 'img'])

    for p in players:
        d = player_scraper(p, 1)
        if d is None:
            continue
        data = d['people'][0]
        attrs = ['id', 'fullName', 'primaryNumber', 'birthDate', 'birthStateProvince', 'birthCountry',
                 'height', 'weight', 'draftYear', 'strikeZoneTop', 'strikeZoneBottom']

        # Extract the desired attributes using a list comprehension
        l1 = [data.get(attr, None) for attr in attrs]
        # l1 = data[['id', 'fullName', 'primaryNumber', 'birthDate', 'birthStateProvince',
        #            'height', 'weight', 'draftYear', 'strikeZoneTop', 'strikeZoneBottom']]
        ppos = data['primaryPosition']['code']
        pb = data['batSide']['code']
        ph = data['pitchHand']['code']
        pimg = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_426,q_auto:best/v1/people/{p}/headshot/67/current"
        l1.extend([ppos, pb, ph, pimg])
        mini = pd.DataFrame(data=[l1], columns=final.columns)
        final = pd.concat([final, mini], axis=0)
        # time.sleep(1.5)

    out_dir = Path.cwd() / "DataPack" / "Misc_Data"
    out_file = f"pitcher_bios_{year}.csv"

    save_file(final, out_dir, out_file)


def get_rosters(year):
    team_ids_path = Path.cwd() / "DataPack" / "Misc_Data" / "team_id_map.csv"
    team_id_df = pd.read_csv(team_ids_path)
    team_ids = list(team_id_df['id'])

    url_template = "https://statsapi.mlb.com/api/v1/teams/{}/roster/40Man?fields=roster,person,fullName,jerseyNumber"

    all_players = []

    for team_id in team_ids:
        url = url_template.format(team_id)
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            for player in data.get("roster", []):
                full_name = player["person"].get("fullName", "")
                jersey_number = player.get("jerseyNumber", "")
                all_players.append({
                    "team_id": team_id,
                    "full_name": full_name,
                    "jersey_number": jersey_number
                })
        except Exception as e:
            print(f"Error with team {team_id}: {e}")

    df = pd.DataFrame(all_players)
    out_dir = Path.cwd() / "DataPack" / "Misc_Data"
    out_file = f"mlb_40man_roster_{year}.csv"

    save_file(df, out_dir, out_file)


def get_all_data(year):
    get_schedule(year)
    get_pbp(year)
    concat_all(year)
    get_pitcher_bios(year)
    get_rosters(year)
