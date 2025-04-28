import requests
import pandas as pd
import json

# Set the year you want to scrape from
year = 2025

# Constrain dates to regular season
start_date = f'{year}-03-18'
end_date = f'{year}-10-02'

sched_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_date}&endDate={end_date}"

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

# Grab the JSON from the MLB API endpoint
schedule = requests.get(sched_url, headers=header_data, timeout=100)

# Convert the data to JSON format
resp = schedule.json()

schedule = []
x = 0

# Iterate over each date
for date in resp['dates']:
    for game in date['games']:
        # Again make sure to only get regular season
        if game['gameType'] != 'R':
            continue
        # Ignore games that were not played
        elif game['status']['detailedState'] == 'Postponed' or game['status']['detailedState'] == 'Suspended':
            continue
        else:
            schedule.append(game)

# Make the schedule into a dataframe and write it to the appropriate location
df = pd.DataFrame(data=schedule)
df = df.drop_duplicates(['gamePk'])

df.to_csv(f"{year}_schedule.csv", index=False)
