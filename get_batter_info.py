import pandas as pd
import numpy as np
import requests
import time

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

pd.options.mode.chained_assignment = None

# Set your year and get all unique batters from your data
year = 2025

df = pd.read_csv(f'{year}_full_pbp.csv')
players = np.unique(df['batter_id'])


# Give a player ID, get all of their data
# Some print statements at the bottom for debugging
def player_scraper(pID, x):
    game_url = f"https://statsapi.mlb.com/api/v1/people/{pID}"
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
        player_scraper(pID, x + 1)


# Define the structure of your output dataframe
final = pd.DataFrame(columns=['id', 'fullName', 'primaryNumber', 'birthDate', 'birthStateProvince', 'birthCountry',
                              'height', 'weight', 'draftYear', 'strikeZoneTop', 'strikeZoneBottom', 'position',
                              'batSide', 'img'])

# Iterate over every player
for p in players:
    print(p)

    # Get the player data. If none, pass
    d = player_scraper(p, 1)
    if d is None:
        continue
    data = d['people'][0]
    attrs = ['id', 'fullName', 'primaryNumber', 'birthDate', 'birthStateProvince', 'birthCountry',
               'height', 'weight', 'draftYear', 'strikeZoneTop', 'strikeZoneBottom']

    # Extract the desired attributes using a list comprehension
    l1 = [data.get(attr, None) for attr in attrs]

    # Get further 2nd level detail of note
    ppos = data['primaryPosition']['code']
    pb = data['batSide']['code']

    # Add in their photo from this URL
    pimg = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_426,q_auto:best/v1/people/{p}/headshot/67/current"
    l1.extend([ppos, pb, pimg])
    mini = pd.DataFrame(data=[l1], columns=final.columns)
    final = pd.concat([final, mini], axis=0)
    ## Sleep if needed
    # time.sleep(1)

# Write your data to a csv
final.to_csv(f'{year}_batter_bios.csv', index=False)
