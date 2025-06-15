import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import networkx as nx
from math import hypot


def is_black_mode_near_top(kp, gray_img, neighborhood_size=150, black_threshold=20):
    x, y = int(kp.pt[0]), int(kp.pt[1])
    r = int(kp.size / 2)
    top_y = max(y - r, 0)
    top_x = x

    # Extract neighborhood pixels around (top_x, top_y)
    h, w = gray_img.shape
    # Collect pixel coords around (top_x, top_y) within a radius big enough to include 100 pixels
    radius = int(np.sqrt(neighborhood_size / np.pi)) + 1

    x_min = max(top_x - radius, 0)
    x_max = min(top_x + radius, w - 1)
    y_min = max(top_y - radius, 0)
    y_max = min(top_y + radius, h - 1)

    neighborhood = []
    for ny in range(y_min, y_max + 1):
        for nx in range(x_min, x_max + 1):
            # Check if pixel inside circle radius
            if (nx - top_x) ** 2 + (ny - top_y) ** 2 <= radius ** 2:
                neighborhood.append(gray_img[ny, nx])

    if len(neighborhood) == 0:
        return False  # No pixels to analyze, assume not black mode

    mode_pixel_val = stats.mode(neighborhood, keepdims=True)[0][0]

    # If mode pixel value is below black_threshold, consider it black mode
    return mode_pixel_val < black_threshold


def get_fielder_positions(image, type='start', plot=False):
    # Convert image to RGB and Grayscale
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # gray = clahe.apply(gray)
    # === Gray Dot Detector ===
    params_gray = cv2.SimpleBlobDetector_Params()
    params_gray.filterByColor = False
    params_gray.filterByArea = True
    if type == 'start':
        params_gray.minArea = 12
        params_gray.maxArea = 30
    elif type == 'end':
        params_gray.minArea = 100
        params_gray.maxArea = 200
    else:
        params_gray.minArea = 5
        params_gray.minArea = 10
    params_gray.filterByCircularity = True
    params_gray.minCircularity = 0.01
    params_gray.filterByInertia = True
    params_gray.minInertiaRatio = 0.01
    detector_gray = cv2.SimpleBlobDetector_create(params_gray)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for gray
    lg = [0, 0, 50] if type == 'start' else [0, 0, 100]
    lower_gray = np.array(lg)
    ug = [180, 40, 220]
    upper_gray = np.array(ug)
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

    # Apply mask to grayscale image before blob detection
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask_gray)

    # Use gray_masked instead of gray in the gray dot detector
    keypoints_gray = detector_gray.detect(gray_masked)

    # === Black Dot Detector ===
    params_black = cv2.SimpleBlobDetector_Params()
    params_black.filterByColor = True
    params_black.blobColor = 0  # Detect dark blobs
    params_black.filterByArea = True
    if type == 'start':
        params_black.minArea = 100
        params_black.maxArea = 150
    elif type == 'end':
        params_black.minArea = 300
        params_black.maxArea = 1000
    else:
        params_black.minArea = 5
        params_black.maxArea = 50
    params_black.filterByCircularity = True
    params_black.minCircularity = 0.5
    params_black.filterByInertia = True
    params_black.minInertiaRatio = 0.5
    detector_black = cv2.SimpleBlobDetector_create(params_black)
    keypoints_black = detector_black.detect(gray)

    filtered_black_keypoints = []
    for kp in keypoints_black:
        if type == 'start':
            if not is_black_mode_near_top(kp, gray):
                filtered_black_keypoints.append(kp)
        else:
            filtered_black_keypoints.append(kp)

    # Combine and deduplicate keypoints
    keypoints = list(keypoints_gray) + filtered_black_keypoints
    starting_positions = [(int(k.pt[0]), int(k.pt[1])) for k in keypoints]

    if plot:
        image_with_keypoints = cv2.drawKeypoints(
            image_rgb, keypoints, None, (255, 0, 0),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # Display result
        plt.figure(figsize=(6, 6))
        plt.imshow(image_with_keypoints)
        plt.title(f"Detected {type} Positions (Gray + Black Filtered)")
        plt.axis("on")
        plt.show()

    return starting_positions


def filter_positions(starting_positions, image):
    # Convert image to HSV for color-based filtering
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red color range in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red regions
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Filter out red keypoints
    non_red_positions = []
    for (x, y) in starting_positions:
        if mask_red[y, x] < 128:  # if pixel is not red
            non_red_positions.append((x, y))

    # Use clustering to remove near-duplicate detections (e.g., average those within 10 pixels)
    from sklearn.cluster import DBSCAN

    clustering = DBSCAN(eps=10, min_samples=1).fit(non_red_positions)
    labels = clustering.labels_

    # Compute cluster centers
    filtered_positions = []
    for label in np.unique(labels):
        cluster_points = np.array([non_red_positions[i] for i in range(len(labels)) if labels[i] == label])
        center = tuple(np.mean(cluster_points, axis=0).astype(int))
        filtered_positions.append(center)

    # Display the cleaned-up points
    filtered_positions.sort(key=lambda pt: pt[1])  # sort by y for easier manual inspection
    return filtered_positions


def filter_positions_final(filtered_positions):
    # Filter points by plausible y-range for fielders (exclude points too close to bottom or top)
    candidate_positions = [pt for pt in filtered_positions if 50 < pt[1] < 250]

    # If more than 9 points, keep the 9 closest to the average field center
    if len(candidate_positions) > 9:
        field_center = np.array([200, 150])  # approximate center of field
        candidate_positions.sort(key=lambda pt: np.linalg.norm(np.array(pt) - field_center))
        candidate_positions = candidate_positions[:9]

    # Sort to make labeling easier later
    candidate_positions.sort(key=lambda pt: pt[1])  # Sort by y-coordinate

    return candidate_positions


def get_dotted_line_mask(image, color='gray'):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if color == 'gray':
        # Gray lines are low saturation, mid-to-high brightness
        lower = np.array([0, 0, 170])   # H=any, S=low, V=medium
        upper = np.array([180, 20, 220])
    elif color == 'black':
        # Black lines are very low brightness, but we must avoid the black background
        # So we target dark blobs *not* touching the image edges (later filtering)
        lower = np.array([0, 0, 0])     # H=any, S=any, V=low
        upper = np.array([180, 255, 50])
    else:
        raise ValueError("Color must be 'gray' or 'black'")

    mask = cv2.inRange(hsv, lower, upper)

    # Optional: clean the mask to close small gaps and remove specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Second step: only dilate with smaller kernel (no erosion)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=5)

    # Component analysis to filter by size
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned_mask = np.zeros_like(mask)

    for i in range(1, num_labels):  # Skip background
        x, y, w, h, area = stats[i]
        diag = hypot(w, h)

        if color == 'gray':
            if w >= 5 and h >= 2:  # adjust these as needed
                cleaned_mask[labels == i] = 255

        elif color == 'black':
            if area < 2000 and x > 0 and y > 0 and x + w < mask.shape[1] - 1 and y + h < mask.shape[0] - 1:
                cleaned_mask[labels == i] = 255

    return cleaned_mask


def mask_to_graph(mask):
    G = nx.Graph()
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:
                node = (x, y)
                G.add_node(node)
                # Connect to 8 neighbors if they exist and are part of mask
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx_ = x + dx
                        ny_ = y + dy
                        if 0 <= nx_ < w and 0 <= ny_ < h and mask[ny_, nx_] > 0:
                            neighbor = (nx_, ny_)
                            if neighbor != node:
                                G.add_edge(node, neighbor)
    return G


def link_positions(init_pos, end_pos, dotted_mask):
    G = mask_to_graph(dotted_mask)
    plt.figure(figsize=(10, 10))
    pos = {node: node for node in G.nodes}  # Use (x, y) as positions
    nx.draw(G, pos, node_size=5, node_color='blue', edge_color='gray')
    plt.gca().invert_yaxis()  # Match image coordinates
    plt.title("Dotted Line Graph")
    plt.show()

    linked_positions = []
    # Convert start positions to set of nodes for quick lookup
    start_nodes = set(init_pos)

    for end_pt in end_pos:
        # Find nearest dotted node to end_pt
        nearest_node = None
        min_dist = float('inf')
        for node in G.nodes:
            dist = (node[0] - end_pt[0]) ** 2 + (node[1] - end_pt[1]) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        # BFS from nearest_node trying to reach any start node
        linked_start = None
        if nearest_node is not None:
            try:
                # Find shortest path lengths from nearest_node to all start_nodes
                lengths = nx.multi_source_dijkstra_path_length(G, start_nodes)
                paths = {}
                for s in start_nodes:
                    if nx.has_path(G, nearest_node, s):
                        path_length = nx.shortest_path_length(G, nearest_node, s)
                        paths[s] = path_length
                if paths:
                    linked_start = min(paths, key=paths.get)
            except Exception as e:
                linked_start = None

        # If no linked start found, assign start = end
        if linked_start is None:
            linked_start = end_pt

        linked_positions.append((linked_start, end_pt))

    return linked_positions


image = cv2.imread('img3.png')

gray_mask = get_dotted_line_mask(image, 'gray')
black_mask = get_dotted_line_mask(image, 'black')
combined_mask = cv2.bitwise_or(gray_mask, black_mask)

init_pos = get_fielder_positions(image, 'start', True)
end_pos = get_fielder_positions(image, 'end', True)
# filtered_pos = filter_positions(init_pos, image)
# final_pos = filter_positions_final(filtered_pos)
# print(final_pos)

# dotted_mask = get_dotted_line_mask(image)
linked_pairs = link_positions(init_pos, end_pos, combined_mask)

# linked_pairs is list of tuples: (start_pos, end_pos)
# If no start_pos connected, start_pos = end_pos

for start, end in linked_pairs:
    print(f"Start: {start}  --> End: {end}")


def plot_all_pos(image, init_pos, end_pos):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.axis('off')

    # Plot initial positions in blue
    if init_pos:
        xs, ys = zip(*init_pos)
        plt.scatter(xs, ys, c='blue', label='Start Positions', s=40, marker='o', edgecolors='w')

    # Plot end positions in red
    if end_pos:
        xs, ys = zip(*end_pos)
        plt.scatter(xs, ys, c='red', label='End Positions', s=40, marker='X', edgecolors='w')

    plt.legend()
    plt.title("Start (blue) and End (red) Positions")
    plt.show()


def plot_linked_pairs(image, linked_pairs):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.axis('off')

    for start, end in linked_pairs:
        plt.plot([start[0], end[0]], [start[1], end[1]], 'g-', linewidth=2)  # line linking
        plt.scatter(*start, c='blue', s=50, label='Start')
        plt.scatter(*end, c='red', s=50, label='End')

    # avoid duplicate labels
    plt.scatter([], [], c='blue', label='Start Positions')
    plt.scatter([], [], c='red', label='End Positions')
    plt.legend()
    plt.title('Linked Start and End Positions')
    plt.show()

# plot_all_pos(image, init_pos, end_pos)

plot_linked_pairs(image, linked_pairs)
