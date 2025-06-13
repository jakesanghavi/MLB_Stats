import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from datetime import date
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import plotting_helper_master
from matplotlib import rcParams, font_manager
from pyfonts import load_google_font
import file_utils


def plot_pitcher_dashboard(player, todays_date, player_info, pbp_df, pitch_map, year=date.today().year):
    fn = "SUSE"
    font_regular = load_google_font(fn, weight="regular")
    font_bold = load_google_font(fn, weight="bold")

    # Register both fonts
    font_path_regular = font_regular.get_file()
    font_path_bold = font_bold.get_file()
    font_manager.fontManager.addfont(font_path_regular)
    font_manager.fontManager.addfont(font_path_bold)

    # Extract the regular font family name
    font_name = font_manager.FontProperties(fname=font_path_regular).get_name()

    pitcher = player_info[player_info['fullName'] == player]

    df_game = pbp_df[(pbp_df['pitcher_id'] == pitcher['id'].iloc[0]) & (pbp_df['date'] == todays_date)].copy()
    score_game_df = pbp_df[pbp_df['game_id'] == df_game['game_id'].iloc[0]]
    last_row = score_game_df.iloc[-1]
    away_score = last_row['away_score']
    home_score = last_row['home_score']

    home = df_game['top_of_inning'].iloc[0] == 1
    opponent = df_game['home_team_abbr'].iloc[0] if not home else df_game['away_team_abbr'].iloc[0]
    against = '@' if not home else 'vs.'
    stadium_path = '../Stadiums/' + df_game['home_team_abbr'].iloc[0] + '_stadium.svg'

    rosters = pd.read_csv('../Misc_Data/mlb_40man_roster.csv')
    team_id = rosters[rosters['full_name'] == player]['team_id'].iloc[0]
    team_map = pd.read_csv('../Misc_Data/team_id_map.csv')
    team_name = team_map[team_map['id'] == team_id]['teamCode'].iloc[0]
    cdf = pd.read_csv('../Misc_Data/team_colors.csv')
    team_colors = list(cdf[cdf['Team'] == team_name].iloc[0].T)
    c1, c2 = team_colors[1], team_colors[2]
    ax_color, fig_color = file_utils.lighten_color(c1), file_utils.lighten_color(c2)

    rcParams.update({
        'font.family': font_name,
        'font.size': 10,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'axes.facecolor': ax_color,
        'figure.facecolor': fig_color,
        'axes.edgecolor': '#cccccc',
        'axes.titleweight': 'bold'  # Only titles use the bold variant
    })

    # Load player image
    response = requests.get(pitcher['img'].iloc[0])
    player_img = Image.open(BytesIO(response.content))

    logo1 = Image.open(f"../Logos/{df_game['away_team_abbr'].iloc[0]}.png")
    logo2 = Image.open(f"../Logos/{df_game['home_team_abbr'].iloc[0]}.png")

    fig = plt.figure(figsize=(14, 10))

    plt.subplots_adjust(hspace=0.0, top=0.95, bottom=0)

    # Main outer layout (1 column, 3 vertical chunks)
    outer_gs = GridSpec(5, 1, height_ratios=[1, 0.5, 1.4, 1.6, 1], hspace=0.2)

    # --- Top Block: Title, Image ---
    top_gs = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer_gs[0], hspace=0)
    ax_header_holder = fig.add_subplot(top_gs[0, :])

    for spine in ax_header_holder.spines.values():
        spine.set_visible(False)

    ax_header_holder.set_xticklabels([])
    ax_header_holder.set_yticklabels([])

    ax_header_holder.tick_params(bottom=False, left=False)

    ax_header_holder.set_facecolor(c1)

    top2_gs = GridSpecFromSubplotSpec(2, 5, subplot_spec=outer_gs[1], hspace=0)
    ax_table = fig.add_subplot(top2_gs[0, :])
    ax_breaks_legend = fig.add_subplot(top2_gs[1, :])

    mid_gs = GridSpecFromSubplotSpec(1, 9, subplot_spec=outer_gs[2], hspace=0)
    ax_lhb = fig.add_subplot(mid_gs[0, :3])
    ax_mirror_bar = fig.add_subplot(mid_gs[0, 3:6])
    ax_rhb = fig.add_subplot(mid_gs[0, 6:])

    # --- Bottom Block: Breaks, Velo, Stats ---
    bot_gs = GridSpecFromSubplotSpec(1, 6, subplot_spec=outer_gs[3], hspace=0)
    ax_breaks = fig.add_subplot(bot_gs[0, :2])
    ax_velocities = fig.add_subplot(bot_gs[0, 2:4])
    ax_spray = fig.add_subplot(bot_gs[0, 4:])

    bot_gs2 = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer_gs[4], hspace=0)
    ax_stats = fig.add_subplot(bot_gs2[0, :])

    ax_header = fig.add_subplot(outer_gs[0])

    plotting_helper_master.plot_header(fig, ax_header, ax_header_holder, c1, player, todays_date, logo1, logo2,
                                       away_score, home_score, player_img)

    plotting_helper_master.plot_game_overview(ax_table, df_game)

    summary_df, unique_pitch_types, pitch_colors_dict, desc_to_code_dict = plotting_helper_master.plot_pitch_breaks(
        ax_breaks, df_game,
        pitch_map)

    plotting_helper_master.plot_velocity_distributions(ax_velocities, df_game, unique_pitch_types, pitch_colors_dict,
                                                       desc_to_code_dict)

    plotting_helper_master.plot_spray(ax_spray, stadium_path, df_game)

    plotting_helper_master.plot_legend(ax_breaks_legend, df_game, unique_pitch_types, pitch_colors_dict,
                                       desc_to_code_dict)

    plotting_helper_master.plot_strike_zone(ax_lhb, "Pitch Locations vs LHB", 'L', df_game, unique_pitch_types,
                                            desc_to_code_dict,
                                            pitch_colors_dict, '../Misc_Images/rhb.svg')

    plotting_helper_master.plot_strike_zone(ax_rhb, "Pitch Locations vs RHB", 'R', df_game, unique_pitch_types,
                                            desc_to_code_dict,
                                            pitch_colors_dict, '../Misc_Images/lhb.svg')

    plotting_helper_master.plot_pitch_usage_bar(ax_mirror_bar, df_game, pitch_map, pitch_colors_dict)

    plotting_helper_master.plot_pitch_by_pitch_info(ax_stats, summary_df)

    plt.show()
