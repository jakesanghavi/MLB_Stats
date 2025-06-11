import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from scipy.stats import gaussian_kde
import file_utils
from matplotlib.lines import Line2D
from datetime import date
from matplotlib.font_manager import FontProperties


def plot_pitcher_dashboard(player, date, player_info, pbp_df, pitch_map, year=date.today().year):
    pitcher = player_info[player_info['fullName'] == player]

    df_game = pbp_df[(pbp_df['pitcher_id'] == pitcher['id'].iloc[0]) & (pbp_df['date'] == date)]

    home = df_game['top_of_inning'].iloc[0] == 1
    opponent = df_game['home_team_abbr'].iloc[0] if home is True else df_game['away_team_abbr'].iloc[0]

    # Load player image
    response = requests.get(pitcher['img'].iloc[0])
    player_img = Image.open(BytesIO(response.content))

    # Set up the layout
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(
        5, 4,
        height_ratios=[1, 0.75, 0.5, 2, 2]
    )
    ax_img = fig.add_subplot(gs[0, 0])
    ax_title = fig.add_subplot(gs[0, 1:4])
    ax_table = fig.add_subplot(gs[1, :])
    ax_breaks_legend = fig.add_subplot(gs[2, :])
    ax_lhb = fig.add_subplot(gs[3, 0])
    ax_rhb = fig.add_subplot(gs[3, 1])
    ax_breaks = fig.add_subplot(gs[3, 2:4])
    ax_stats = fig.add_subplot(gs[4, :])

    # Remove axes where not needed
    for ax in [ax_img, ax_title, ax_table, ax_breaks, ax_lhb, ax_rhb, ax_stats]:
        ax.axis("off")

    # Player Image
    ax_img.imshow(player_img)
    ax_img.set_title(player, fontsize=14)

    # Title
    ax_title.text(0, 0.8, f"Daily Pitching Summary\n{year} MLB Season", fontsize=18, fontweight='bold')
    ax_title.text(0, 0.4, f"{date} vs {opponent}", fontsize=14)

    # Plate Appearances = number of unique ab_id
    pa = df_game['ab_id'].nunique()
    sps = df_game[(df_game['detailed_pitch_outcome'] == 'X') | (df_game['event_type'].str.contains('sac_'))]['ab_id'].\
          nunique()
    dps = df_game[df_game['event_type'].str.contains('double_play')]['ab_id'].nunique()
    tps = df_game[df_game['event_type'].str.contains('triple_play')]['ab_id'].nunique()
    outs = sps + dps + tps*2
    ip = float(str(int(outs/3)) + '.' + str(outs % 3))
    er = df_game.groupby('ab_id')['rbi'].max().sum()
    hit_events = ['single', 'double', 'triple', 'home_run']
    h = df_game[df_game['event_name'].str.lower().isin(hit_events)]['ab_id'].nunique()
    k = df_game[df_game['event_name'].str.lower().str.contains("strikeout")]['ab_id'].nunique()
    bb = df_game[df_game['event_name'].str.lower().str.contains("walk")]['ab_id'].nunique()
    # hbp = df_game[df_game['detailed_pitch_outcome'] == 'HBP']['ab_id'].nunique()
    hr = df_game[df_game['event_name'].str.lower() == 'home_run']['ab_id'].nunique()
    strike_pct = df_game['strike'].sum() / len(df_game) * 100
    # whiff_pct = df_game[df_game['detailed_pitch_outcome'] == 'S'].shape[0] / len(df_game) * 100
    pitch_count = len(df_game)

    # Combine into dict
    summary_data = {
        "IP": round(ip, 1),
        "Pitches": pitch_count,
        "BF": int(pa),
        "ER": er,
        "H": h,
        "K": k,
        "BB": bb,
        # "HBP": hbp,
        "HR": hr,
        "Strike%": round(strike_pct, 1),
        # "Whiff%": round(whiff_pct, 1)
    }

    table_text = pd.DataFrame([summary_data])

    int_cols = ["Pitches", "BF", "ER", "H", "K", "BB", "HR"]
    for col in int_cols:
        if col in table_text.columns:
            table_text[col] = table_text[col].astype('Int64')

    overall_table = ax_table.table(cellText=table_text.values, colLabels=table_text.columns, loc='center',
                                   cellLoc='center')
    for (row, col), cell in overall_table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
    ax_table.set_title("Pitching Line")

    df_game['whiff'] = (df_game['detailed_pitch_outcome'] == 'S').astype(int)
    df_game['x_break'] = df_game['break_length'] * np.sin(np.deg2rad(df_game['break_angle']))
    df_game['y_break'] = df_game['break_length'] * np.cos(np.deg2rad(df_game['break_angle']))

    summary_df = df_game.groupby(['pitch_type'], as_index=False).agg(
        Count=('pitch_type', 'size'),
        Velocity=('start_speed', 'mean'),
        Whiffs=('whiff', 'sum'),
        Break=('break_length', 'mean')
    )

    unique_codes = summary_df['pitch_type'].unique()

    summary_df = summary_df.merge(pitch_map, left_on='pitch_type', right_on='Code', how='inner')
    summary_df['Whiff%'] = summary_df['Whiffs']/summary_df['Count'] * 100

    summary_df['Count'] = summary_df['Count'].astype(int)
    summary_df['Velocity'] = summary_df['Velocity'].round(1)
    summary_df['Whiff%'] = summary_df['Whiff%'].round(1)
    summary_df['Break'] = summary_df['Break'].round(1)
    summary_df['Pitch Type'] = summary_df['Description']

    summary_df = summary_df[['Pitch Type', 'Count', 'Velocity', 'Whiff%', 'Break']].sort_values('Count', ascending=False)

    int_cols2 = ["Count"]
    for col in int_cols2:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].astype('Int64')

    unique_pitch_types = summary_df['Pitch Type'].unique()

    pitch_colors_dict = pitch_map.set_index('Description')['Color'].to_dict()
    desc_to_code_dict = pitch_map.set_index('Description')['Code'].to_dict()

    # Fake break data
    for pitch_desc in unique_pitch_types:
        pitch_data = df_game[df_game['pitch_type'] == desc_to_code_dict.get(pitch_desc)]

        current_color = pitch_colors_dict.get(pitch_desc)

        ax_breaks.scatter(
            pitch_data['x_break'],
            pitch_data['y_break'],
            label=pitch_desc,
            color=current_color,
            alpha=0.6,
            s=50
        )

    ax_breaks.axvline(0, color='gray', linestyle='--')
    ax_breaks.axhline(0, color='gray', linestyle='--')

    ax_breaks.set_xlabel("Horizontal Break (in)")
    ax_breaks.set_ylabel("Induced Vertical Break (in)")
    ax_breaks.set_title("Pitch Breaks")

    legend_handles = []
    for pitch_desc in unique_pitch_types:
        color = pitch_colors_dict.get(pitch_desc)
        if color is None:
            continue
        # You can also check if this pitch was in df_game, to filter unused ones
        if df_game[df_game['pitch_type'] == desc_to_code_dict.get(pitch_desc)].empty:
            continue
        handle = Line2D([0], [0], color=color, lw=5, label=pitch_desc)
        legend_handles.append(handle)

    ax_breaks_legend.legend(handles=legend_handles, loc='center', ncol=3, frameon=False)
    ax_breaks_legend.axis('off')

    def plot_strike_zone(ax, title, bat_side):
        # --- Draw the strike zone box ---
        strike_zone = file_utils.draw_strike_zone_rect()
        ax.add_patch(strike_zone)

        padding_x = 0.25
        padding_y = 0.25

        # Based on actual box size
        ax.set_xlim(-1 - padding_x, 1 + padding_x)
        ax.set_ylim(0.75 - padding_y, 4 + padding_y)
        ax.set_aspect('equal')
        ax.set_title(title)

        # --- Filter data by batter side ---
        filtered_df_game = df_game[df_game['bat_side'] == bat_side].copy()

        # --- Grid for KDE evaluation ---
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])

        for pitch_desc in unique_pitch_types:
            pitch_code = desc_to_code_dict.get(pitch_desc)
            if not pitch_code:
                continue

            pitch_data = filtered_df_game[filtered_df_game['pitch_type'] == pitch_code]
            if len(pitch_data) < 2:
                continue

            x_locs = pitch_data['pitch_coordinate_X'].values
            z_locs = pitch_data['pitch_coordinate_Z'].values
            values = np.vstack([x_locs, z_locs])

            # --- KDE and mass contour logic ---
            try:
                kernel = gaussian_kde(values)
                Z = np.reshape(kernel(positions).T, xx.shape)

                # Compute 75% KDE level
                Z_flat = Z.ravel()
                Z_sorted = np.sort(Z_flat)[::-1]
                cumsum = np.cumsum(Z_sorted)
                cumsum /= cumsum[-1]
                level_75 = Z_sorted[np.searchsorted(cumsum, 0.5)]

                # Extract contour at 75% level
                contours = ax.contour(xx, yy, Z, levels=[level_75], linewidths=1.5, colors='none')

                # Get the largest path
                max_area = 0
                main_path = None
                for collection in contours.collections:
                    for path in collection.get_paths():
                        verts = path.vertices
                        area = 0.5 * np.abs(np.dot(verts[:, 0], np.roll(verts[:, 1], 1)) -
                                            np.dot(verts[:, 1], np.roll(verts[:, 0], 1)))
                        if area > max_area:
                            max_area = area
                            main_path = path

                if main_path is not None:
                    color = pitch_colors_dict.get(pitch_desc, 'gray')
                    patch = PathPatch(main_path, facecolor=color, edgecolor=color, alpha=0.4, lw=2)
                    ax.add_patch(patch)

                    # Outline
                    ax.plot(main_path.vertices[:, 0], main_path.vertices[:, 1], color=color, lw=2)

                    # Dummy legend
                    ax.plot([], [], color=color, label=pitch_desc, linewidth=5, alpha=0.6)

            except Exception as e:
                print(f"Skipping {pitch_desc} due to KDE or contour error: {e}")

        ax.axis('off')

    plot_strike_zone(ax_lhb, "Pitch Locations vs LHB", 'L')
    plot_strike_zone(ax_rhb, "Pitch Locations vs RHB", 'R')

    ax_stats.axis('tight')
    ax_stats.axis('off')
    pitches_table = ax_stats.table(cellText=summary_df.values, colLabels=summary_df.columns, loc='center',
                                   cellLoc='center')
    for (row, col), cell in pitches_table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
    ax_stats.set_title("Pitch Type Summary")

    plt.tight_layout()
    plt.show()
