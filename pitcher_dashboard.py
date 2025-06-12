import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from datetime import date
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import plotting_helper_master


def plot_pitcher_dashboard(player, todays_date, player_info, pbp_df, pitch_map, year=date.today().year):
    pitcher = player_info[player_info['fullName'] == player]

    df_game = pbp_df[(pbp_df['pitcher_id'] == pitcher['id'].iloc[0]) & (pbp_df['date'] == todays_date)].copy()
    # df_game[['ab_id', 'detailed_pitch_outcome', 'event_type', 'description']].to_csv('df_game.csv', index=False)

    home = df_game['top_of_inning'].iloc[0] == 1
    opponent = df_game['home_team_abbr'].iloc[0] if not home else df_game['away_team_abbr'].iloc[0]
    against = '@' if not home else 'vs.'
    stadium_path = '../Stadiums/' + df_game['home_team_abbr'].iloc[0] + '_stadium.svg'

    # Load player image
    response = requests.get(pitcher['img'].iloc[0])
    player_img = Image.open(BytesIO(response.content))

    fig = plt.figure(figsize=(14, 10))

    plt.subplots_adjust(hspace=0.0, top=0.95, bottom=0)

    # Main outer layout (1 column, 3 vertical chunks)
    outer_gs = GridSpec(5, 1, height_ratios=[1, 0.5, 1.4, 1.6, 1], hspace=0.2)

    # --- Top Block: Title, Image ---
    top_gs = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer_gs[0], hspace=0)
    ax_title = fig.add_subplot(top_gs[0, :3])
    ax_img = fig.add_subplot(top_gs[0, 3:4])

    top2_gs = GridSpecFromSubplotSpec(2, 5, subplot_spec=outer_gs[1], hspace=0)
    ax_table = fig.add_subplot(top2_gs[0, :])
    ax_breaks_legend = fig.add_subplot(top2_gs[1, :])

    mid_gs = GridSpecFromSubplotSpec(1, 7, subplot_spec=outer_gs[2], hspace=0)
    ax_lhb = fig.add_subplot(mid_gs[0, :2])
    ax_mirror_bar = fig.add_subplot(mid_gs[0, 2:5])
    ax_rhb = fig.add_subplot(mid_gs[0, 5:])

    # --- Bottom Block: Breaks, Velo, Stats ---
    bot_gs = GridSpecFromSubplotSpec(1, 6, subplot_spec=outer_gs[3], hspace=0)
    ax_breaks = fig.add_subplot(bot_gs[0, :2])
    ax_velocities = fig.add_subplot(bot_gs[0, 2:4])
    ax_spray = fig.add_subplot(bot_gs[0, 4:])

    bot_gs2 = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer_gs[4], hspace=0)
    ax_stats = fig.add_subplot(bot_gs2[0, :])

    for ax in [ax_img, ax_title, ax_table, ax_lhb, ax_rhb, ax_stats]:
        ax.axis("off")

    # Player Image
    ax_img.imshow(player_img)
    ax_img.set_title(player, fontsize=14)

    # Title
    ax_title.text(0, 0.8, f"Daily Pitching Summary\n{year} MLB Season", fontsize=18, fontweight='bold')
    ax_title.text(0, 0.4, f"{todays_date} {against} {opponent}", fontsize=14)

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
