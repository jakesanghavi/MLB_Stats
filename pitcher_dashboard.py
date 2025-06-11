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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def plot_game_overview(ax_table, df_game):
    # Plate Appearances = number of unique ab_id
    pa = df_game['ab_id'].nunique()
    sps = df_game[(df_game['detailed_pitch_outcome'] == 'X') | (df_game['event_type'].str.contains('sac_'))]['ab_id']. \
        nunique()
    dps = df_game[df_game['event_type'].str.contains('double_play')]['ab_id'].nunique()
    tps = df_game[df_game['event_type'].str.contains('triple_play')]['ab_id'].nunique()
    outs = sps + dps + tps * 2
    ip = float(str(int(outs / 3)) + '.' + str(outs % 3))
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
    # ax_table.set_title("Pitching Line")


def plot_legend(ax_breaks_legend, df_game, unique_pitch_types, pitch_colors_dict, desc_to_code_dict):
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

    ax_breaks_legend.legend(
        handles=legend_handles,
        loc='center',
        bbox_to_anchor=(0.5, 0.5, 0.1, 0.1),  # shrink box size
        bbox_transform=ax_breaks_legend.transAxes,
        ncol=3,
        frameon=False,
        handlelength=2.5,
        fontsize=9,
        borderpad=0.1,
        labelspacing=0.2,
        handletextpad=0.5,
        columnspacing=0.8
    )
    ax_breaks_legend.set_xlim(0, 1)  # force tight x space
    ax_breaks_legend.set_ylim(0, 1)  # force tight y space
    ax_breaks_legend.axis('off')


def plot_pitch_breaks(ax_breaks, df_game, pitch_map):
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
    summary_df['Whiff%'] = summary_df['Whiffs'] / summary_df['Count'] * 100

    summary_df['Count'] = summary_df['Count'].astype(int)
    summary_df['Velocity'] = summary_df['Velocity'].round(1)
    summary_df['Whiff%'] = summary_df['Whiff%'].round(1)
    summary_df['Break'] = summary_df['Break'].round(1)
    summary_df['Pitch Type'] = summary_df['Description']

    summary_df = summary_df[['Pitch Type', 'Count', 'Velocity', 'Whiff%', 'Break']].sort_values('Count',
                                                                                                ascending=False)

    int_cols2 = ["Count"]
    for col in int_cols2:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].astype('Int64')

    unique_pitch_types = summary_df['Pitch Type'].unique()

    pitch_colors_dict = pitch_map.set_index('Description')['Color'].to_dict()
    desc_to_code_dict = pitch_map.set_index('Description')['Code'].to_dict()

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

    return summary_df, unique_pitch_types, pitch_colors_dict, desc_to_code_dict


def plot_strike_zone(ax, title, bat_side, df_game, unique_pitch_types, desc_to_code_dict, pitch_colors_dict):
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

        pitch_data = filtered_df_game[filtered_df_game['pitch_type'] == pitch_code]
        if len(pitch_data) < 2:
            continue

        x_locs = pitch_data['pitch_coordinate_X'].values
        z_locs = pitch_data['pitch_coordinate_Z'].values
        values = np.vstack([x_locs, z_locs])

        # --- KDE and mass contour logic ---
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

    ax.axis('off')


def plot_pitch_usage_bar(ax, df_game, pitch_map, pitch_colors_dict):
    pitch_counts = df_game.groupby(['bat_side', 'pitch_type']).size().unstack(fill_value=0)

    # Get counts for each side
    lhb_counts = pitch_counts.loc['L'] if 'L' in pitch_counts.index else pd.Series(dtype=int)
    lhb_tot = lhb_counts.sum()
    rhb_counts = pitch_counts.loc['R'] if 'R' in pitch_counts.index else pd.Series(dtype=int)
    rhb_tot = rhb_counts.sum()

    # Get all pitch types involved
    all_pitch_types = sorted(set(lhb_counts.index).union(rhb_counts.index))
    lhb_counts = lhb_counts.reindex(all_pitch_types, fill_value=0)
    rhb_counts = rhb_counts.reindex(all_pitch_types, fill_value=0)

    sorted_idx = rhb_counts.sort_values(ascending=False).index
    lhb_counts = lhb_counts.loc[sorted_idx]
    rhb_counts = rhb_counts.loc[sorted_idx]

    lhb_pct = lhb_counts / lhb_tot * 100
    rhb_pct = rhb_counts / rhb_tot * 100

    y_labels = []
    for pt in sorted_idx:
        desc_row = pitch_map[pitch_map['Code'] == pt]
        desc = desc_row['Description'].values[0] if not desc_row.empty else pt
        y_labels.append(desc)

    # Plot bars with consistent colors
    for i, pt in enumerate(sorted_idx):
        desc_row = pitch_map[pitch_map['Code'] == pt]
        desc = desc_row['Description'].values[0] if not desc_row.empty else pt
        color = pitch_colors_dict.get(desc, 'gray')

        # Plot bars
        ax.barh(i, lhb_pct[pt], color=color, alpha=0.6, edgecolor='black')  # LHB: right side
        ax.barh(i, -rhb_pct[pt], color=color, alpha=0.6, edgecolor='black')  # RHB: left side

        # Centered count text inside bars
        count_l = int(lhb_counts[pt])
        count_r = int(rhb_counts[pt])
        if count_l > 0:
            ax.text(lhb_pct[pt] / 2, i, f"{count_l}", va='center', ha='center', fontsize=8)
        if count_r > 0:
            ax.text(-rhb_pct[pt] / 2, i, f"{count_r}", va='center', ha='center', fontsize=8)

    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlim(-100, 100)
    # ax.set_xlabel("% of Total Pitches")
    ax.set_xticklabels([100, 50, 0, 50, 100])
    ax.set_title("Pitch Usage by Batter Side")
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)


def plot_velocity_distributions(ax, title, df_game, unique_pitch_types, pitch_colors_dict, desc_to_code_dict):
    ax.set_title(title)
    ax.set_xlabel("Release Speed (MPH)")
    ax.set_ylabel("Density")

    df_game_copy = df_game.copy()

    min_speed = df_game_copy['start_speed'].min() - 5
    max_speed = df_game_copy['start_speed'].max() + 5
    velocity_range = np.linspace(min_speed, max_speed, 500)  # 500 points for smooth curve
    KDE_CLIP_THRESHOLD = 0.005

    for pitch_desc in unique_pitch_types:
        pitch_data = df_game_copy[df_game_copy['pitch_type'] == desc_to_code_dict[pitch_desc]]
        speeds = pitch_data['start_speed'].values

        if len(speeds) > 1:
            # Perform 1D KDE for velocity
            kernel = gaussian_kde(speeds, bw_method='scott')
            density = kernel(velocity_range)

            density[density < KDE_CLIP_THRESHOLD] = 0

            current_color = pitch_colors_dict.get(pitch_desc)

            ax.fill_between(velocity_range, 0, density, color=current_color, alpha=0.4, label=pitch_desc)
            ax.plot(velocity_range, density, color=current_color, linewidth=0.8, alpha=0.7)

    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_visible(False)


def plot_pitch_by_pitch_info(ax_stats, summary_df):
    ax_stats.axis('tight')
    ax_stats.axis('off')
    pitches_table = ax_stats.table(cellText=summary_df.values, colLabels=summary_df.columns, loc='center',
                                   cellLoc='center')
    for (row, col), cell in pitches_table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
    # ax_stats.set_title("Pitch Type Summary")


def plot_pitcher_dashboard(player, date, player_info, pbp_df, pitch_map, year=date.today().year):
    pitcher = player_info[player_info['fullName'] == player]

    df_game = pbp_df[(pbp_df['pitcher_id'] == pitcher['id'].iloc[0]) & (pbp_df['date'] == date)]

    home = df_game['top_of_inning'].iloc[0] == 1
    opponent = df_game['home_team_abbr'].iloc[0] if home is True else df_game['away_team_abbr'].iloc[0]

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

    mid_gs = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer_gs[2], hspace=0)
    ax_rhb = fig.add_subplot(mid_gs[0, :2])
    ax_mirror_bar = fig.add_subplot(mid_gs[0, 2:3])
    ax_lhb = fig.add_subplot(mid_gs[0, 3:])

    # --- Bottom Block: Breaks, Velo, Stats ---
    bot_gs = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer_gs[3], hspace=0)
    ax_breaks = fig.add_subplot(bot_gs[0, 1:3])
    ax_velocities = fig.add_subplot(bot_gs[0, 3:])

    bot_gs2 = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer_gs[4], hspace=0)
    ax_stats = fig.add_subplot(bot_gs2[0, :])

    for ax in [ax_img, ax_title, ax_table, ax_breaks, ax_lhb, ax_rhb, ax_stats]:
        ax.axis("off")

    # Player Image
    ax_img.imshow(player_img)
    ax_img.set_title(player, fontsize=14)

    # Title
    ax_title.text(0, 0.8, f"Daily Pitching Summary\n{year} MLB Season", fontsize=18, fontweight='bold')
    ax_title.text(0, 0.4, f"{date} vs {opponent}", fontsize=14)

    plot_game_overview(ax_table, df_game)

    summary_df, unique_pitch_types, pitch_colors_dict, desc_to_code_dict = plot_pitch_breaks(ax_breaks, df_game, pitch_map)

    plot_velocity_distributions(ax_velocities, "Velocities", df_game, unique_pitch_types, pitch_colors_dict, desc_to_code_dict)

    plot_legend(ax_breaks_legend, df_game, unique_pitch_types, pitch_colors_dict, desc_to_code_dict)


    plot_strike_zone(ax_rhb, "Pitch Locations vs RHB", 'R', df_game, unique_pitch_types, desc_to_code_dict,
                     pitch_colors_dict)

    plot_strike_zone(ax_lhb, "Pitch Locations vs LHB", 'L', df_game, unique_pitch_types, desc_to_code_dict,
                     pitch_colors_dict)

    plot_pitch_usage_bar(ax_mirror_bar, df_game, pitch_map, pitch_colors_dict)

    plot_pitch_by_pitch_info(ax_stats, summary_df)

    mid_gs_axes = [ax_rhb, ax_mirror_bar, ax_lhb]

    fig.tight_layout()
    plt.tight_layout(h_pad=0)
    plt.show()
