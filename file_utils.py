import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import label
import matplotlib.colors as mcolors
import colorsys
from PIL import Image
from rembg import remove, new_session
from pathlib import Path
import json
from sklearn.cluster import DBSCAN
import cv2
import pandas as pd


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
    PLATE_LEFT = -(HOME_WIDTH / 2)
    PLATE_RIGHT = (HOME_WIDTH / 2)
    PLATE_SIDE = (HOME_WIDTH / 2)
    KNEES = 18
    CHEST = 42

    BASEBALL_AREA = np.pi * (9.125 / 2) ** 2

    V_THIRD = (PLATE_RIGHT - PLATE_LEFT) / 3
    H_THIRD = (CHEST - KNEES) / 3

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

    ls = 'k-'
    if home_plate is True:
        plt.plot([-0.708, 0.708], [0, 0], ls)
        plt.plot([-0.708, -0.708], [0, -0.3], ls)
        plt.plot([0.708, 0.708], [0, -0.3], ls)
        plt.plot([-0.708, 0], [-0.3, -0.6], ls)
        plt.plot([0.708, 0], [-0.3, -0.6], ls)

    plt.axis('equal')
    plt.xlim([-2, 2])
    plt.ylim([-1.2, 5])
    plt.show()


def draw_strike_zone_rect():
    HOME_WIDTH = 17  # inches
    KNEES = 18  # inches
    CHEST = 42  # inches

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


def fast_remove(img):
    session = new_session("u2net_human_seg")
    output = remove(img, session=session)

    return output


def is_dark(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < 128


def _m(hsv, lo, hi):
    return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))


# Orange-red used for ball path. Used for batter/runner path too but I think not working
def mask_red_orange(hsv):
    return cv2.bitwise_or(_m(hsv, [0, 100, 100], [18, 255, 255]),
                          _m(hsv, [158, 100, 100], [180, 255, 255]))


# Red for batter/runner END dots. Needs better calibration
def mask_pure_red(hsv):
    return cv2.bitwise_or(_m(hsv, [0, 180, 70], [10, 255, 255]),
                          _m(hsv, [168, 180, 70], [180, 255, 255]))


# Blue throw path/s color
def mask_blue(hsv):
    return _m(hsv, [95, 80, 80], [128, 255, 255])


# Light-gray fielder dots/paths: low saturation.
def mask_gray(hsv):
    return _m(hsv, [0, 0, 140], [180, 55, 255])


# White baselines/general to ignore coor
def mask_white(hsv):
    return _m(hsv, [0, 0, 200], [180, 30, 255])


# Field mask. Grass = green, dirt = brown, slight dilation too
def field_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = _m(hsv, [40, 25, 70],  [100, 120, 175])
    brown = _m(hsv, [12, 60, 100], [32,  230, 210])
    fm = cv2.bitwise_or(green, brown)
    fm = cv2.morphologyEx(fm, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    fm = cv2.morphologyEx(fm, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(fm, 8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fm = (lbl == largest).astype(np.uint8) * 255
    # Dilate so dots/lines sitting on the boundary are included for help
    fm = cv2.dilate(fm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    return fm


# Close morph
def close(m, k):
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, el)


# Open morph
def open_(m, k):
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, el)


# Dilate elipse
def dilate(m, k, it=1):
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(m, el, iterations=it)


# Euclidean distance helper
def euc(a, b):
    return float(np.linalg.norm(np.array(a, float) - np.array(b, float)))


# Deduplicate via dbscan
def dedup(pts, eps=12):
    if not pts: return []
    arr = np.array(pts, float)
    labels = DBSCAN(eps=eps, min_samples=1).fit(arr).labels_
    return [tuple(arr[labels == l].mean(0).round().astype(int).tolist())
            for l in np.unique(labels)]


# Find nearest candidate to point
def nearest(pt, candidates):
    return min(candidates, key=lambda c: euc(pt, c)) if candidates else None


# Find pair w greatest dist among point collection
def farthest_pair(pts):
    if len(pts) < 2: return (pts[0] if pts else None), None
    best, p1, p2 = -1, pts[0], pts[1]
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d = euc(pts[i], pts[j])
            if d > best:
                best, p1, p2 = d, pts[i], pts[j]
    return p1, p2


# FMT
def fmt(pt):
    return f"({pt[0]}, {pt[1]})" if pt is not None else None


# Detect blob centroids over mask area
def blob_centroids(mask, min_area=10, max_area=5000, min_circ=0.3):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if not (min_area <= a <= max_area):
            continue
        p = cv2.arcLength(c, True)
        if 4 * np.pi * a / (p*p + 1e-6) < min_circ:
            continue
        M = cv2.moments(c)
        if M['m00'] == 0: continue
        out.append((int(M['m10']/M['m00']), int(M['m01']/M['m00'])))
    return out


# Skeletonize from mask
def skeletonize(mask):
    skel = np.zeros_like(mask)
    el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = mask.copy()
    while True:
        eroded = cv2.erode(img, el)
        temp = cv2.subtract(img, cv2.dilate(eroded, el))
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


# Given skeleton, get endpoints
def skel_endpoints(skel):
    k = np.ones((3, 3), np.uint8)
    nb = cv2.filter2D((skel > 0).astype(np.uint8), -1, k)
    ep = np.argwhere((skel > 0) & (nb == 2))
    return [(int(x), int(y)) for y, x in ep]


# From mask, get endpoints
def line_endpoints(mask, min_comp=40):
    mask = close(mask, 7)
    mask = open_(mask, 3)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None, None
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]
    all_ep = []
    for idx in order[:3]:
        if stats[idx+1, cv2.CC_STAT_AREA] < min_comp: continue
        comp = ((lbl == idx+1)*255).astype(np.uint8)
        all_ep.extend(skel_endpoints(skeletonize(comp)))
    return farthest_pair(all_ep) if all_ep else (None, None)


# Involved-fielder labeled circle detection
# Uses HoughCircles + interior pixel analysis (dark ring + white text inside).
def detect_labeled_circles(gray, fm_dilated, min_r=6, max_r=13):
    """
    Detect small circles (fielder position labels: LF, SS, C, 1B, …) that have:
      - A very dark region inside (the black fill)
      - Bright pixels inside (the white position-text)
      - Located on the field
    Returns list of (cx, cy).
    """
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=15,
        param1=50, param2=15, minRadius=min_r, maxRadius=max_r
    )
    if circles is None:
        return []
    circles = np.round(circles[0]).astype(int)
    H, W = gray.shape
    found = []
    for (cx, cy, r) in circles:
        # Must be within the (dilated) field mask
        y_, x_ = max(0, min(H-1, cy)), max(0, min(W-1, cx))
        if fm_dilated[y_, x_] == 0:
            continue
        # Sample all pixels inside the circle
        circ_mask = np.zeros(gray.shape, np.uint8)
        cv2.circle(circ_mask, (cx, cy), r, 255, -1)
        pixels = gray[circ_mask > 0]
        if len(pixels) == 0:
            continue
        # Must have both very dark pixels (black fill) and very bright (white text)
        if pixels.min() < 20 and pixels.max() > 150:
            found.append((int(cx), int(cy)))
    return dedup(found, eps=10)


# Main Tracker Class to run things above
class BaseballTracker:
    # Position shortcuts
    POSITIONS = ['C', '1B', '2B', 'SS', '3B', 'P', 'LF', 'CF', 'RF']

    # Normalised (x%, y%) reference coords for a top-down ~400x400 baseball diagram.
    # y increases downward; CF at top-center, C at bottom-center.
    # POS_REF = {
    #     'P':  (0.500, 0.430),
    #     'C':  (0.505, 0.590),
    #     '1B': (0.615, 0.510),
    #     '2B': (0.500, 0.388),
    #     'SS': (0.385, 0.450),
    #     '3B': (0.385, 0.515),
    #     'LF': (0.220, 0.290),
    #     'CF': (0.490, 0.200),
    #     'RF': (0.710, 0.300),
    # }

    # Corrected refs
    POS_REF = {
        'P': (0.500, 0.718),
        'C': (0.500, 0.805),
        '1B': (0.583, 0.683),
        '2B': (0.573, 0.603),
        'SS': (0.458, 0.603),
        '3B': (0.420, 0.663),
        'LF': (0.273, 0.370),
        'CF': (0.500, 0.280),
        'RF': (0.728, 0.360),
    }

    # Init class from our image
    def __init__(self, image_path):
        self.path = str(image_path)
        self.img = cv2.imread(self.path)
        if self.img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        self.H, self.W = self.img.shape[:2]
        self.hsv  = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        print(f"  Image: {self.W}x{self.H} px")
        self.fm = field_mask(self.img)

    def _on(self, m):
        return cv2.bitwise_and(m, self.fm)

    # Ball path method
    def _ball_path(self):
        m = mask_red_orange(self.hsv)
        m = self._on(m)
        m = open_(m, 2)
        # Remove thick dot blobs (erode to kill them, then subtract from original)
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        big = cv2.dilate(cv2.erode(m, el), el)
        line_only = cv2.subtract(m, big)
        # Reconnect dashes from ex. skipped throw
        line_only = dilate(line_only, 5, 2)
        return line_endpoints(line_only)

    # Throw path/s method
    def _throw_paths(self):
        m = mask_blue(self.hsv)
        m = self._on(m)
        m = dilate(m, 5, 2)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        segs = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 40: continue
            comp = ((lbl == i)*255).astype(np.uint8)
            ep = skel_endpoints(skeletonize(comp))
            if len(ep) >= 2:
                p1, p2 = farthest_pair(ep)
                segs.append((p1, p2, area))
        segs.sort(key=lambda s: s[2], reverse=True)
        return [(s[0], s[1]) for s in segs[:2]]

    # Light gray dots (uninvolved fielders)
    def _gray_dots(self):
        """
        Gray circular dots at the start and end of dashed fielder paths.
        We dilate the combined gray+white mask to bridge the dashes, then find
        connected components that contain multiple dots = (start, end) pairs.
        """
        gm = mask_gray(self.hsv)
        gm = self._on(gm)
        wm = mask_white(self.hsv)
        wm = self._on(wm)
        path_m = cv2.bitwise_or(gm, wm)

        dots_m = open_(gm, 2)
        dot_pts = dedup(blob_centroids(dots_m, min_area=18, max_area=600, min_circ=0.50), 10)

        path_d = dilate(path_m, 9, 3)
        n, lbl, _, _ = cv2.connectedComponentsWithStats(path_d, 8)

        comp_dots = {}
        for pt in dot_pts:
            x, y = int(pt[0]), int(pt[1])
            y = max(0, min(self.H-1, y)); x = max(0, min(self.W-1, x))
            cid = int(lbl[y, x])
            if cid > 0:
                comp_dots.setdefault(cid, []).append(pt)

        pairs, paired = [], set()
        for cid, pts in comp_dots.items():
            if len(pts) == 2:
                pairs.append((pts[0], pts[1]))
                paired.update(pts)
            elif len(pts) > 2:
                p1, p2 = farthest_pair(pts)
                if p1 and p2:
                    pairs.append((p1, p2))
                    paired.update([p1, p2])

        # If not paired, player did not move
        for pt in dot_pts:
            if pt not in paired:
                pairs.append((pt, pt))

        return pairs, dot_pts

    # Actually invovled (black) fielder stuff
    def _involved_fielder_circles(self):
        """
        Detect the small black circles with white position labels (LF, SS, …).
        Returns list of (cx, cy) = END positions of involved fielders.
        Also tries to find small black START dots near each circle.
        """
        # Use the dilated field mask so border circles are included
        fm_big = dilate(self.fm, 25)
        end_pts = detect_labeled_circles(self.gray, fm_big)

        # Small black start dots: look for very dark isolated blobs on the field
        # that are NOT the background and NOT near any labeled circle
        dark_m = _m(self.hsv, [0, 0, 0], [180, 255, 50])
        dark_m = self._on(dark_m)
        dark_m = open_(dark_m, 2)
        small_pts = dedup(blob_centroids(dark_m, min_area=8, max_area=100, min_circ=0.25), 8)
        # Exclude any small dot that is at the same location as a labeled circle end
        start_pts = [s for s in small_pts
                     if not any(euc(s, e) < 18 for e in end_pts)]
        return end_pts, start_pts

    # Batter/Runner ends
    def _batter_runner_ends(self):
        pm = mask_pure_red(self.hsv)
        pm = open_(pm, 2)
        return dedup(blob_centroids(pm, min_area=25, max_area=5000, min_circ=0.30), 12)

    # Batter/Runner starts
    def _batter_runner_starts(self):
        m = mask_red_orange(self.hsv)
        m = self._on(m)
        m = open_(m, 2)
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        big = cv2.dilate(cv2.erode(m, el), el)
        small_only = cv2.subtract(m, big)
        return dedup(blob_centroids(small_only, min_area=5, max_area=200, min_circ=0.15), 10)

    # Assing positions of fielders given candidates compared to our position refs
    # Need to be careful about savant changing image stuff in future, this is a liability and coud break
    def _assign_positions(self, end_candidates, start_candidates):
        ref = {pos: (int(fx*self.W), int(fy*self.H))
               for pos, (fx, fy) in self.POS_REF.items()}

        remaining = list(end_candidates)
        pos_end = {}

        # Min distance between point and reamining points
        def min_d(pos):
            rp = ref[pos]
            return min((euc(rp, ep) for ep in remaining), default=9999)

        for pos in sorted(self.POSITIONS, key=min_d):
            if not remaining:
                pos_end[pos] = None
                continue
            best = min(remaining, key=lambda ep: euc(ref[pos], ep))
            if euc(ref[pos], best) < 100:
                pos_end[pos] = best
                remaining.remove(best)
            else:
                pos_end[pos] = None

        pos_start, used = {}, set()
        for pos in self.POSITIONS:
            ep = pos_end[pos]
            if ep is None:
                pos_start[pos] = None
                continue
            cands = [s for s in start_candidates if s not in used]
            sp = nearest(ep, cands) if cands else None
            if sp and euc(ep, sp) < 140:
                pos_start[pos] = sp; used.add(sp)
            else:
                pos_start[pos] = ep   # player didn't move

        return pos_end, pos_start

    # Driver method
    def run(self):
        # Ball path
        # I manually set start here, seemed to be having some issues detecting but its always same I think
        ball_e, ball_s = self._ball_path()
        ball_s = (200, 325)

        # Get throw paths
        throws = self._throw_paths()

        # Uninvolved fieldrs
        gray_pairs, gray_dots = self._gray_dots()

        # Invoolved fielders
        inv_ends, inv_starts = self._involved_fielder_circles()

        # Batter/runners ends
        # Sorta broken for now
        bat_ends = self._batter_runner_ends()

        # Batter/runners starts
        # Sorta broken for now
        bat_starts = self._batter_runner_starts()

        # Pool all fielder end/start candidates and assign positions after batters/runners removed
        gray_ends = [p[1] for p in gray_pairs]
        gray_starts = [p[0] for p in gray_pairs]
        all_ends = dedup(gray_ends + inv_ends,   eps=15)
        all_starts = dedup(gray_starts + inv_starts, eps=12)
        all_ends = [e for e in all_ends if not any(euc(e, be) < 22 for be in bat_ends)]
        all_starts = [s for s in all_starts if not any(euc(s, bs) < 18 for bs in bat_starts)]

        pos_end, pos_start = self._assign_positions(all_ends, all_starts)

        bat_ends_s = sorted(bat_ends,   key=lambda p: p[1])
        bat_starts_s = sorted(bat_starts, key=lambda p: p[1])

        def bat_entry(idx):
            end = bat_ends_s[idx] if idx < len(bat_ends_s) else None
            if end is None: return None, None
            start = nearest(end, bat_starts_s) if bat_starts_s else None
            if start and euc(end, start) < 140:
                return start, end
            return end, end

        b_s,  b_e = bat_entry(0)
        r1_s, r1_e = bat_entry(1)
        r2_s, r2_e = bat_entry(2)
        r3_s, r3_e = bat_entry(3)

        def throw(idx):
            return (throws[idx][0], throws[idx][1]) if idx < len(throws) else (None, None)
        t1s, t1e = throw(0)
        t2s, t2e = throw(1)

        result = {
            "ball_path_start": fmt(ball_s), "ball_path_end": fmt(ball_e),
            "throw_1_start":   fmt(t1s),    "throw_1_end":   fmt(t1e),
            "throw_2_start":   fmt(t2s),    "throw_2_end":   fmt(t2e),
            "B_start":  fmt(b_s),   "B_end":  fmt(b_e),
            "1R_start": fmt(r1_s),  "1R_end": fmt(r1_e),
            "2R_start": fmt(r2_s),  "2R_end": fmt(r2_e),
            "3R_start": fmt(r3_s),  "3R_end": fmt(r3_e),
        }
        for pos in self.POSITIONS:
            result[f"{pos}_start"] = fmt(pos_start.get(pos))
            result[f"{pos}_end"] = fmt(pos_end.get(pos))

        # Store for viz
        self.result = result
        self._pos_end = pos_end
        self._pos_start = pos_start
        self._bat_ends = bat_ends_s
        self._bat_starts = bat_starts_s
        self._throws = throws
        self._ball = (ball_s, ball_e)
        self._inv_ends = inv_ends
        return result


# Render viz
def visualize(tracker):
    img = tracker.img.copy()

    def dot(pt, clr, r=6, fill=True):
        if pt:
            cv2.circle(img, pt, r, clr, -1 if fill else 2)

    def seg(a, b, clr, t=2):
        if a and b: cv2.line(img, a, b, clr, t)

    def tag(pt, txt, clr):
        if pt:
            cv2.putText(img, txt, (pt[0]+4, pt[1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, clr, 1, cv2.LINE_AA)

    # Color defs to match savant
    ORNG = (0, 160, 255)
    BLUE = (200, 80, 20)
    RED = (30, 30, 220)
    LGRAY = (180, 180, 180)

    bs, be = tracker._ball
    dot(bs, ORNG, 5); dot(be, ORNG, 5); seg(bs, be, ORNG)
    tag(bs, "ball_s", ORNG); tag(be, "ball_e", ORNG)

    for i, (ts, te) in enumerate(tracker._throws):
        dot(ts, BLUE, 5); dot(te, BLUE, 5); seg(ts, te, BLUE)
        tag(ts, f"T{i+1}s", BLUE); tag(te, f"T{i+1}e", BLUE)

    bat_labels = ['B', '1R', '2R', '3R']
    bat_colors = [RED, (0,180,80), (0,80,200), (180,0,180)]
    for idx, (lbl, clr) in enumerate(zip(bat_labels, bat_colors)):
        e = tracker._bat_ends[idx] if idx < len(tracker._bat_ends) else None
        s = tracker._bat_starts[idx] if idx < len(tracker._bat_starts) else None
        dot(e, clr, 7); dot(s, clr, 4); seg(s, e, clr)
        tag(e, f"{lbl}_e", clr); tag(s, f"{lbl}_s", clr)

    for pt in tracker._inv_ends:
        cv2.circle(img, pt, 9, (0, 255, 0), 2)

    for pos in tracker.POSITIONS:
        ep = tracker._pos_end.get(pos)
        sp = tracker._pos_start.get(pos)
        dot(ep, LGRAY, 7, fill=False); dot(sp, LGRAY, 4)
        tag(ep, pos, LGRAY)

    out = Path(tracker.path).stem + "_annotated.png"
    cv2.imwrite(out, img)
    print(f"  Annotated image -> {out}")
    return out


# Main method
def play_tracker(img_path=None):
    # Default if not given
    image_path = img_path if img_path is not None else "img.png"

    tracker = BaseballTracker(image_path)
    result = tracker.run()

    groups = [
        ("Ball Path",   ["ball_path_start", "ball_path_end"]),
        ("Throw 1",     ["throw_1_start",   "throw_1_end"]),
        ("Throw 2",     ["throw_2_start",   "throw_2_end"]),
        ("Batter  B",   ["B_start",         "B_end"]),
        ("Runner 1R",   ["1R_start",        "1R_end"]),
        ("Runner 2R",   ["2R_start",        "2R_end"]),
        ("Runner 3R",   ["3R_start",        "3R_end"]),
    ]
    for pos in tracker.POSITIONS:
        groups.append((f"Fielder {pos}", [f"{pos}_start", f"{pos}_end"]))

    for label, keys in groups:
        vals = " | ".join(
            f"{'START' if k.endswith('_start') else 'END'}: {result.get(k) or 'null'}"
            for k in keys
        )

    stem = Path(image_path).stem
    pd.DataFrame([result]).to_csv(stem + "_results.csv", index=False)
    with open(stem + "_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  CSV  -> {stem}_results.csv")
    print(f"  JSON -> {stem}_results.json")
    visualize(tracker)
    print("\nDone.")
