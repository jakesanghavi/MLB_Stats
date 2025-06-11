import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import label


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
