from plotting_helper_master import *
import pandas as pd

year = 2026
player_info_path = Path.cwd() / "DataPack" / "Misc_Data" / f"pitcher_bios_{year}.csv"
player_info = pd.read_csv(player_info_path)
pbp_path = Path.cwd() / "DataPack" / "PBP" / f"{year}_full_pbp.csv"
pbp_df = pd.read_csv(pbp_path)
pitch_map_path = Path.cwd() / "DataPack" / "Misc_Data" / "pitch_types_map.csv"
pitch_map = pd.read_csv(pitch_map_path)

plot_pitcher_dashboard(player='Ranger Suárez', game_date='2026-03-30', player_info=player_info, pbp_df=pbp_df,
                       pitch_map=pitch_map)
