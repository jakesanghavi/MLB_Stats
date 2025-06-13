import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import label
import matplotlib.colors as mcolors
import colorsys
import cv2
from PIL import Image
from rembg import remove
import io
from collections import defaultdict

SPRAY_CMAP = {
    '1B': '#fe6000',
    '2B': '#775eef',
    '3B': '#ffb005',
    'HR': '#dc2680',
    'X': '#c2c2c2'
}


def plot_empty_strike_zone(gridlines=True, home_plate=True):
    HOME_WIDTH = 17
    HOME_HEIGHT = 17
    PLATE_LEFT = -(HOME_WIDTH/2)
    PLATE_RIGHT = (HOME_WIDTH/2)
    PLATE_SIDE = (HOME_WIDTH/2)
    KNEES = 18
    CHEST = 42

    BASEBALL_AREA = np.pi*(9.125/2)**2

    V_THIRD = (PLATE_RIGHT - PLATE_LEFT)/3
    H_THIRD = (CHEST - KNEES)/3

    # Strike zone vertical lines
    plt.vlines(PLATE_LEFT / 12, KNEES / 12, CHEST / 12)
    plt.vlines(PLATE_RIGHT / 12, KNEES / 12, CHEST / 12)

    # Strike zone horizontal lines
    plt.hlines(KNEES / 12, PLATE_LEFT / 12, PLATE_RIGHT / 12)
    plt.hlines(CHEST / 12, PLATE_LEFT / 12, PLATE_RIGHT / 12)

    if gridlines is True:
        plt.vlines((PLATE_LEFT + V_THIRD) / 12, KNEES / 12, CHEST / 12, alpha=0.2)
        plt.vlines((PLATE_RIGHT - V_THIRD) / 12, KNEES / 12, CHEST / 12, alpha=0.2)
        plt.hlines((KNEES + H_THIRD) / 12, PLATE_LEFT / 12, PLATE_RIGHT / 12, alpha=0.2)
        plt.hlines((CHEST - H_THIRD) / 12, PLATE_LEFT / 12, PLATE_RIGHT / 12, alpha=0.2)

    # home_plate = Polygon([[PLATE_LEFT/12, 0], [PLATE_RIGHT/12, 0], [PLATE_RIGHT/12, -PLATE_SIDE/12 * 0.8],
    #                       [0, -HOME_HEIGHT/12 * 0.8], [PLATE_LEFT/12, -PLATE_SIDE/12 * 0.8]],
    #                      closed=True, fill=None, edgecolor='black')
    # plt.gca().add_patch(home_plate)

    ls='k-'
    if home_plate is True:
        plt.plot([-0.708, 0.708], [0,0], ls)
        plt.plot([-0.708, -0.708], [0,-0.3], ls)
        plt.plot([0.708, 0.708], [0,-0.3], ls)
        plt.plot([-0.708, 0], [-0.3, -0.6], ls)
        plt.plot([0.708, 0], [-0.3, -0.6], ls)

    plt.axis('equal')
    plt.xlim([-2, 2])
    plt.ylim([-1.2, 5])
    plt.show()


def draw_strike_zone_rect():
    HOME_WIDTH = 17  # inches
    KNEES = 18       # inches
    CHEST = 42       # inches

    PLATE_LEFT = -(HOME_WIDTH / 2)
    width = HOME_WIDTH
    height = CHEST - KNEES

    # Convert to feet
    x = PLATE_LEFT / 12
    y = KNEES / 12
    width /= 12
    height /= 12

    return Rectangle((x, y), width, height, linewidth=1.5, edgecolor='black', facecolor='none')


def get_largest_region_contour(xx, yy, Z, level):
    # Mask points above KDE threshold
    mask = Z >= level

    # Label connected regions in mask
    labeled, ncomponents = label(mask)

    if ncomponents == 0:
        return None

    # Find largest connected component by pixel count
    largest_label = 1
    largest_size = 0
    for i in range(1, ncomponents + 1):
        size = np.sum(labeled == i)
        if size > largest_size:
            largest_size = size
            largest_label = i

    # Create binary mask of largest region
    largest_region = (labeled == largest_label)

    # Extract boundary contour of this region using matplotlib contour on mask
    contours = plt.contour(xx, yy, largest_region.astype(float), levels=[0.5])

    # Find the longest contour path (in case multiple)
    max_len = 0
    main_path = None
    for collection in contours.collections:
        for p in collection.get_paths():
            if len(p.vertices) > max_len:
                max_len = len(p.vertices)
                main_path = p

    plt.close()  # Close the temporary contour plot

    return main_path


# def mlbam_xy_transformation(x, y, distance, scale=2.495671):
#     new_x = scale * (x - 130)
#     new_y = scale * (213 - y)
#     return new_x, new_y

def mlbam_xy_transformation(x, y, distance, scale=2.495671):
    # Calculate spray angle in radians
    spray_angle = -np.arctan((x - 130) / (213 - y)) + np.pi / 2

    # Convert polar to Cartesian using distance
    new_x = distance * np.cos(spray_angle)
    # new_y = distance * np.sin(spray_angle)
    new_y = scale * (213 - y)

    return new_x, new_y


def get_event_label(event_type):
    event_type = event_type.lower()
    if 'home_run' in event_type:
        return 'HR'
    elif 'triple' in event_type and 'play' not in event_type:
        return '3B'
    elif 'double' in event_type and 'play' not in event_type:
        return '2B'
    elif 'single' in event_type:
        return '1B'
    elif 'field_out' in event_type or 'double_play' in event_type or 'triple_play' in event_type:
        return 'X'
    else:
        pass


def scale_svg(xmin, xmax, ymin, ymax):
    # Compute centers
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2

    # Half sizes
    half_width = (xmax - xmin) / 2
    half_height = (ymax - ymin) / 2

    # Scale factor (e.g., 1.5)
    scale_svg = 1.8

    # New half sizes after scaling
    new_half_width = half_width * scale_svg
    new_half_height = half_height * scale_svg

    # New scaled extents, keeping center fixed
    xmin_scaled = x_center - new_half_width
    xmax_scaled = x_center + new_half_width
    ymin_scaled = y_center - new_half_height
    ymax_scaled = y_center + new_half_height

    return xmin_scaled, xmax_scaled, ymin_scaled, ymax_scaled


def scale_axes():
    padding_x = 0.25
    padding_y = 0.25

    scale_factor = 2.25

    half_width = (1 + padding_x)
    half_height = (4 - 0.75) / 2 + padding_y

    # scaled half-width and half-height
    new_half_width = half_width * scale_factor
    new_half_height = half_height * scale_factor

    return new_half_width, new_half_height


def lighten_color(hex_color, amount=0.6):
    rgb = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = min(1, l + amount * (1 - l))  # increase lightness toward 1
    light_rgb = colorsys.hls_to_rgb(h, l, s)
    return mcolors.to_hex(light_rgb)


def resize_image(img, max_height=512):
    w, h = img.size
    if h > max_height:
        ratio = max_height / h
        return img.resize((int(w * ratio), max_height), Image.Resampling.LANCZOS)
    return img


def make_color_transparent(img, color='#c2c2c2', tol=20):
    img_rgb = tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

    img = img.convert("RGBA")
    datas = img.getdata()

    def close_enough(pixel, target, tol):
        return all(abs(p - t) <= tol for p, t in zip(pixel[:3], target))

    newData = []
    for item in datas:
        if close_enough(item, img_rgb, tol):
            newData.append((255, 255, 255, 0))  # transparent
        else:
            newData.append(item)

    img.putdata(newData)
    return img


def remove_background(player_img):
    # Convert to bytes
    with io.BytesIO() as buf:
        player_img.save(buf, format='PNG')
        input_bytes = buf.getvalue()

    # Remove background using rembg
    output_bytes = remove(input_bytes)

    # Convert result bytes back to PIL image with alpha channel
    return Image.open(io.BytesIO(output_bytes)).convert("RGBA")


def is_dark(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < 128
