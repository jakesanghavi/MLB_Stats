import numpy.linalg
import pandas as pd
from matplotlib.patches import PathPatch
from PIL import Image
import numpy as np
from scipy.stats import gaussian_kde
import file_utils
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import io
import os
import re
os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = '/opt/homebrew/lib'
import cairosvg


def plot_header(fig, ax_header, ax_header_holder, c1, pitcher, todays_date, logo1, logo2, away_score, home_score, player_img):
    # player_img = file_utils.make_color_transparent(player_img, '#c9c7cc', tol=35)
    player_img = file_utils.remove_background(player_img)
    ax_header.axis("off")  # no ticks or spines
    ax_header.set_facecolor(c1)  # set background color

    text_color = 'white' if file_utils.is_dark(c1) else 'black'

    # Now place text and image inside ax_header:
    # Text on left side
    ax_header.text(
        0.35, 0.75,
        f"{pitcher}\nDaily Pitching Summary",
        fontsize=18, fontweight='bold', va='top', ha='left',
        transform=ax_header.transAxes,
        color = text_color
    )
    ax_header.text(
        0.35, 0.3,
        f"{todays_date}",
        fontsize=14, va='top', ha='left',
        transform=ax_header.transAxes,
        color=text_color
    )

    bbox = ax_header_holder.get_position()
    x0, y0 = bbox.x0, bbox.y0
    _, image_height = player_img.size

    fig_width, fig_height = fig.get_size_inches() * fig.dpi
    bbox = ax_header_holder.get_position()

    # Compute the height in pixels of the target box
    box_height_px = bbox.height * fig_height
    target_height_px = box_height_px * 0.35  # 95% of height

    target_height_px_logo = box_height_px * 0.15

    zoom = target_height_px / image_height

    imagebox = OffsetImage(player_img, zoom=zoom)  # Adjust zoom for size

    ab_hs = AnnotationBbox(
        imagebox,
        (x0 + 0.1, y0),
        xycoords='figure fraction',  # <- note this
        box_alignment=(0, 0),  # <- anchor image bottom-left
        frameon=False,
        zorder=1
    )

    logo1 = logo1.convert("RGBA")
    logo2 = logo2.convert("RGBA")

    # Determine logo scaling
    _, logo1_height = logo1.size
    _, logo2_height = logo2.size
    max_logo_height = target_height_px_logo
    max_logo_width = box_height_px * 0.15  # adjust as needed

    # logo1 scaling
    logo1_w, logo1_h = logo1.size
    logo1_zoom = min(max_logo_height / logo1_h, max_logo_width / logo1_w)

    # logo2 scaling
    logo2_w, logo2_h = logo2.size
    logo2_zoom = min(max_logo_height / logo2_h, max_logo_width / logo2_w)

    # X positions for layout (tweak spacing as needed)
    start_x = bbox.x1 - 0.075  # start from far right
    spacing = 0.04

    positions = {
        'score2': start_x,
        'logo2': start_x - spacing,
        '@': start_x - 2 * spacing,
        'logo1': start_x - 3 * spacing,
        'score1': start_x - 4 * spacing,
        'author': start_x + 0.01
    }

    y_pos_text = y0 + 0.042  # vertical alignment
    y_pos_img = y0 + 0.07

    # Add logo1
    imgbox1 = OffsetImage(logo1, zoom=logo1_zoom)
    ab1 = AnnotationBbox(imgbox1, (positions['logo1'], y_pos_img), xycoords='figure fraction',
                         box_alignment=(0.5, 0.5), frameon=False, zorder=3)
    ax_header.add_artist(ab1)

    # Add logo2
    imgbox2 = OffsetImage(logo2, zoom=logo2_zoom)
    ab2 = AnnotationBbox(imgbox2, (positions['logo2'], y_pos_img), xycoords='figure fraction',
                         box_alignment=(0.5, 0.5), frameon=False, zorder=3)
    ax_header.add_artist(ab2)

    # Add scores and "@"
    fig.text(positions['score1'], y_pos_text + 0.02, str(away_score), fontsize=16, ha='center',
             va='bottom', color=text_color, transform=fig.transFigure, fontweight='bold')
    fig.text(positions['score2'], y_pos_text + 0.02, str(home_score), fontsize=16, ha='center',
             va='bottom', color=text_color, transform=fig.transFigure, fontweight='bold')
    fig.text(positions['@'], y_pos_text + 0.02, "@", fontsize=16, ha='center',
             va='bottom', color=text_color, transform=fig.transFigure, fontweight='bold')
    fig.text(positions['author'], y_pos_text - 0.04, "Author: Jake Sanghavi", fontsize=9, ha='center',
             va='bottom', color=text_color, transform=fig.transFigure)

    ax_header.add_artist(ab_hs)
    ax_header.axis('off')
    for spine in ax_header.spines.values():
        spine.set_visible(False)


def plot_game_overview(ax_table, df_game, c1):
    # Plate Appearances = number of unique ab_id
    bf = df_game['ab_id'].nunique()
    fulls = (df_game['inning'].max() - df_game['inning'].min()) * 3
    partials = df_game['outs'].iloc[-1] - df_game['outs'].iloc[0]
    add = 1 if df_game['detailed_pitch_outcome'].iloc[-1] in ['C', 'S', 'W', 'X'] else 0

    outs = fulls + partials + add

    ip = float(str(int(outs / 3)) + '.' + str(outs % 3))
    er = df_game.groupby('ab_id')['rbi'].max().sum()
    hit_events = ['single', 'double', 'triple', 'home_run']
    pattern = '|'.join(hit_events)
    mask = df_game['event_type'].str.contains(pattern, case=False, na=False)
    h = df_game[mask]['ab_id'].nunique()
    k = df_game[df_game['event_name'].str.lower().str.contains("strikeout")]['ab_id'].nunique()
    bb = df_game[df_game['event_name'].str.lower().str.contains("walk")]['ab_id'].nunique()
    # hbp = df_game[df_game['detailed_pitch_outcome'] == 'HBP']['ab_id'].nunique()
    hr = df_game[df_game['event_type'].str.contains('home_run')]['ab_id'].nunique()
    strike_pct = df_game['strike'].sum() / len(df_game) * 100
    # whiff_pct = df_game[df_game['detailed_pitch_outcome'] == 'S'].shape[0] / len(df_game) * 100
    pitch_count = len(df_game)

    # Combine into dict
    summary_data = {
        "IP": round(ip, 1),
        "Pitches": pitch_count,
        "BF": int(bf),
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
        # Align all cells
        cell.get_text().set_horizontalalignment('center')
        cell.get_text().set_verticalalignment('center')

        cell.PAD = 0.8
        text_color = 'white' if file_utils.is_dark(c1) else 'black'

        # Bold header
        if row == 0:
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
            cell.set_facecolor(c1)
            cell.get_text().set_color(text_color)
        else:
            cell.set_facecolor(file_utils.lighten_color(c1, 0.7) if row % 2 == 1 else file_utils.lighten_color(c1, 0.9))
    ax_table.axis('off')


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
    df_game['whiff'] = ((df_game['detailed_pitch_outcome'] == 'S') |
                        (df_game['detailed_pitch_outcome'] == 'W') |
                        (df_game['detailed_pitch_outcome'] == 'T')).astype(int)

    summary_df = df_game.groupby(['pitch_type'], as_index=False).agg(
        Count=('pitch_type', 'size'),
        Velocity=('start_speed', 'mean'),
        Whiffs=('whiff', 'sum')
    )

    summary_df = summary_df.merge(pitch_map, left_on='pitch_type', right_on='Code', how='inner')
    summary_df['Whiff%'] = summary_df['Whiffs'] / summary_df['Count'] * 100

    summary_df['Count'] = summary_df['Count'].astype(int)
    summary_df['Velocity'] = summary_df['Velocity'].round(1)
    summary_df['Whiff%'] = summary_df['Whiff%'].round(1)
    summary_df['Pitch Type'] = summary_df['Description']

    summary_df = summary_df[['Pitch Type', 'Count', 'Velocity', 'Whiff%']].sort_values('Count',
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
            pitch_data['break_horizontal'],
            pitch_data['break_vertical_induced'],
            label=pitch_desc,
            color=current_color,
            alpha=0.6,
            s=50,
            zorder=3
        )

    ax_breaks.set_aspect('equal')
    ax_breaks.set_xlim([-30, 30])
    ax_breaks.set_ylim([-30, 30])
    ax_breaks.set_xticks(range(-30, 35, 5))
    ax_breaks.set_yticks(range(-30, 35, 5))

    ticks = [x for x in range(-30, 35, 5) if x != 0]

    for x in ticks:
        line = ax_breaks.axvline(x, color='lightgray', linewidth=0.8, zorder=1)
        line.set_dashes([2, 4])  # 2 points dash, 4 points space

    for y in ticks:
        line = ax_breaks.axhline(y, color='lightgray', linewidth=0.8, zorder=1)
        line.set_dashes([2, 4])  # same dash pattern

    # Solid lines at origin (above grid, below scatter)
    ax_breaks.axhline(0, color='black', linewidth=1.5, zorder=2)
    ax_breaks.axvline(0, color='black', linewidth=1.5, zorder=2)

    ax_breaks.set_xlabel("Horizontal Break (in)")
    ax_breaks.set_ylabel("Vertical Break (in)")

    for spine in ax_breaks.spines.values():
        spine.set_visible(False)

    ax_breaks.tick_params(axis='both', which='both', length=0, labelsize=6)

    return summary_df, unique_pitch_types, pitch_colors_dict, desc_to_code_dict


def plot_strike_zone(ax, title, bat_side, df_game, unique_pitch_types, desc_to_code_dict, pitch_colors_dict, svg_path):
    # --- Draw the strike zone box ---
    strike_zone = file_utils.draw_strike_zone_rect()
    strike_zone.set_zorder(-10)
    ax.add_patch(strike_zone)
    ax.set_title(title)

    # --- Filter data by batter side ---
    filtered_df_game = df_game[df_game['bat_side'] == bat_side].copy()

    with open(svg_path, 'r', encoding='utf-8') as f:
        svg_content = f.read()

    # Convert SVG to PNG bytes
    png_bytes = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'))
    image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    if bat_side == 'L':
        xmin, xmax = -1.7, 0.8
        ymin, ymax = 1.1, 4.6
    else:
        xmin, xmax = 2, 4.5
        ymin, ymax = 1.1, 4.6

    xmin_scaled, xmax_scaled, ymin_scaled, ymax_scaled = file_utils.scale_svg(xmin, xmax, ymin, ymax)

    # Display image with scaled extent
    ax.imshow(image, extent=[xmin_scaled, xmax_scaled, ymin_scaled, ymax_scaled], aspect='auto', zorder=-10)

    x_center = 0
    y_center = (0.75 + 4) / 2

    new_half_width, new_half_height = file_utils.scale_axes()

    ax.set_xlim(x_center - new_half_width, x_center + new_half_width)
    ax.set_ylim(y_center - new_half_height, y_center + new_half_height)
    ax.set_aspect('equal')

    # --- Grid for KDE evaluation ---
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    for pitch_desc in unique_pitch_types:
        try:
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
            total = cumsum[-1]

            # Skip if total is 0 or NaN to avoid divide-by-zero
            if total == 0 or np.isnan(total):
                continue

            cumsum /= total
            level_75 = Z_sorted[np.searchsorted(cumsum, 0.3)]

            # Extract contour at 75% level
            contours = ax.contour(xx, yy, Z, levels=[level_75], linewidths=1.5, colors='none')
            for coll in contours.collections:
                coll.set_zorder(10)

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
                patch = PathPatch(main_path, facecolor=color, edgecolor=color, alpha=0.4, lw=2, zorder=10)
                ax.add_patch(patch)

                # Outline
                ax.plot(main_path.vertices[:, 0], main_path.vertices[:, 1], color=color, lw=2,  zorder=11)

                # Dummy legend
                ax.plot([], [], color=color, label=pitch_desc, linewidth=5, alpha=0.6)
        except numpy.linalg.LinAlgError:
            pass

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
        ax.barh(i, -lhb_pct[pt], color=color, alpha=0.6, edgecolor='black')  # LHB: right side
        ax.barh(i, rhb_pct[pt], color=color, alpha=0.6, edgecolor='black')  # RHB: left side

        # Centered count text inside bars
        count_l = int(lhb_counts[pt])
        count_r = int(rhb_counts[pt])
        if count_l > 0:
            ax.text(-lhb_pct[pt] / 2, i, f"{count_l}", va='center', ha='center', fontsize=8)
        if count_r > 0:
            ax.text(rhb_pct[pt] / 2, i, f"{count_r}", va='center', ha='center', fontsize=8)

    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlim(-100, 100)
    # ax.set_xlabel("% of Total Pitches")
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xticklabels(['100%', '50%', '0%', '50%', '100%'])
    ax.set_yticks([])
    ax.set_title("Pitch Usage by Batter Side")
    ax.invert_yaxis()
    ax.spines[['left', 'top', 'right']].set_visible(False)


def plot_velocity_distributions(ax, df_game, unique_pitch_types, pitch_colors_dict, desc_to_code_dict):
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


def plot_spray(ax, svg_path, df_game):
    # Read and modify SVG (hide distances)
    with open(svg_path, 'r', encoding='utf-8') as f:
        svg_content = f.read()

    svg_content = re.sub(
        r'(#distances\s*\{\s*visibility\s*:\s*)visible(\s*;?\s*})',
        r'\1hidden\2',
        svg_content,
        flags=re.IGNORECASE
    )

    png_bytes = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'))
    image = Image.open(io.BytesIO(png_bytes))

    # Filter dataframe & last per ab_id
    filtered = df_game[df_game['detailed_pitch_outcome'].isin(['D', 'E', 'X'])].copy()
    filtered['event_label'] = filtered['event_type'].apply(file_utils.get_event_label)
    last_per_ab = filtered.groupby('ab_id').last().reset_index()

    # Define SVG bounds (example, replace with your actual bounds)
    xmin, xmax = -250, 250
    ymin, ymax = -10, 450

    # Plot SVG image with imshow and extent
    ax.imshow(image, extent=[xmin, xmax, ymin, ymax], aspect='auto')

    # Plot points
    label_order = ['X', '1B', '2B', '3B', 'HR']
    color_map = file_utils.SPRAY_CMAP

    for event_label, group in last_per_ab.groupby('event_label'):
        transformed_coords = group.apply(
            lambda row: file_utils.mlbam_xy_transformation(
                row['hit_location_X'],
                row['hit_location_Y'],
                row['hit_distance']
            ),
            axis=1
        )

        transformed_x = transformed_coords.apply(lambda coord: coord[0])
        transformed_y = transformed_coords.apply(lambda coord: coord[1])

        ax.scatter(
            transformed_x,
            transformed_y,
            label=event_label,
            color=color_map.get(event_label, 'black'),
            edgecolor='black',
            alpha=0.8,
            s=50
        )

    # Set axis limits combining SVG and points bounds
    x_min_points = last_per_ab['hit_location_X'].min()
    x_max_points = last_per_ab['hit_location_X'].max()
    y_min_points = last_per_ab['hit_location_Y'].min()
    y_max_points = last_per_ab['hit_location_Y'].max()

    xmin_plot = min(xmin, x_min_points)
    xmax_plot = max(xmax, x_max_points)
    ymin_plot = min(ymin, y_min_points)
    ymax_plot = max(ymax, y_max_points)

    padding_x = (xmax_plot - xmin_plot) * 0.05
    padding_y = (ymax_plot - ymin_plot) * 0.05

    ax.set_xlim(xmin_plot - padding_x, xmax_plot + padding_x)
    ax.set_ylim(ymin_plot - padding_y, ymax_plot + padding_y)
    ax.set_aspect('equal')
    ax.axis("off")

    present_labels = last_per_ab['event_label'].unique()

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=label,
               markerfacecolor=color_map.get(label, 'black'), markersize=10)
        for label in label_order if label in present_labels
    ]

    ax.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),  # Moved closer to plot
        ncol=len(legend_elements),
        frameon=False
    )


def plot_pitch_by_pitch_info(ax_stats, summary_df, c1):
    ax_stats.axis('tight')
    ax_stats.axis('off')
    pitches_table = ax_stats.table(cellText=summary_df.values, colLabels=summary_df.columns, loc='center',
                                   cellLoc='center')
    for (row, col), cell in pitches_table.get_celld().items():
        # Align all cells
        cell.get_text().set_horizontalalignment('center')
        cell.get_text().set_verticalalignment('center')

        cell.PAD = 0.8

        # Bold header
        text_color = 'white' if file_utils.is_dark(c1) else 'black'

        # Bold header
        if row == 0:
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
            cell.set_facecolor(c1)
            cell.get_text().set_color(text_color)
        else:
            cell.set_facecolor(file_utils.lighten_color(c1, 0.7) if row % 2 == 1 else file_utils.lighten_color(c1, 0.9))
