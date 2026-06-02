import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import numpy as np  # <-- Need numpy for xy_to_hex


# -----------------------------
# 1️⃣ Convert (x, y) to hex_id
# (This is needed to convert your new logic's centers)
# -----------------------------
def xy_to_hex(x, y, size):
    """Convert (x, y) to a hex grid coordinate."""
    q = (np.sqrt(3) / 3 * x - 1 / 3 * y) / size
    r = (2 / 3 * y) / size
    # Round to nearest hex
    rq = round(q)
    rr = round(r)
    rs = round(-q - r)

    # Correct for rounding errors
    q_diff = abs(rq - q)
    r_diff = abs(rr - r)
    s_diff = abs(rs - (-q - r))

    if q_diff > r_diff and q_diff > s_diff:
        rq = -rr - rs
    elif r_diff > s_diff:
        rr = -rq - rs

    return f"{int(rq)}_{int(rr)}"


# -----------------------------
# 2️⃣ Convert hex_id back to (x, y)
# (Used by visualization)
# -----------------------------
def hex_to_xy(hex_id, size):
    """Convert hex axial coordinate (q_r) back to (x, y) for visualization."""
    try:
        q, r = map(int, hex_id.split("_"))
    except (ValueError, TypeError):
        # print(f"Warning: Skipping invalid hex_id '{hex_id}'")
        return None, None

    x = size * math.sqrt(3) * (q + r / 2)
    y = size * 3 / 2 * r
    return x, y


# -----------------------------
# 3️⃣ Helper: compute hex corners
# (Used by visualization)
# -----------------------------
def hex_corners(center_x, center_y, size):
    """Return 6 (x,y) corner coordinates of a pointy-top hex."""
    corners = []
    for i in range(6):
        angle_deg = 60 * i - 30
        angle_rad = math.radians(angle_deg)
        corners.append((
            center_x + size * math.cos(angle_rad),
            center_y + size * math.sin(angle_rad)
        ))
    return corners


# -----------------------------
# 4️⃣ Visualization function
#    (No changes needed. This function is correct.)
# -----------------------------
def visualize_traffic_from_csv(
        zone_stats_df,
        hex_size,
        all_hex_ids,  # This list will now be the 182 zones
        output_dir="traffic_maps_from_csv",
        add_labels=True,
        label_column='label',
        start_time=None,
        end_time=None
):
    """
    Draw colored hexagons for each zone at each timestep from the DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)

    color_map = {
        1: "#4CAF50",  # Green
        2: "#FFEB3B",  # Yellow
        3: "#FF9800",  # Orange
        4: "#F44336",  # Red
        5: "#380000",  # Dark Red/Maroon
    }
    DEFAULT_COLOR = "#E0E0E0"  # A light gray for inactive hexes

    all_timesteps = sorted(zone_stats_df["time"].unique())
    if start_time is None: start_time = min(all_timesteps) if all_timesteps else 0
    if end_time is None: end_time = max(all_timesteps) if all_timesteps else 0

    timesteps_to_process = [
        t for t in all_timesteps if t >= start_time and t <= end_time
    ]

    if not timesteps_to_process:
        print(f"No timesteps found in range [{start_time}, {end_time}].")
        return

    print(f"Processing {len(timesteps_to_process)} timesteps in range [{start_time}, {end_time}]...")

    # Pre-calculate all 182 hex geometries
    map_geometry = {}
    all_corners_x, all_corners_y = [], []
    for hex_id in all_hex_ids:
        cx, cy = hex_to_xy(hex_id, hex_size)
        if cx is None: continue
        corners = hex_corners(cx, cy, hex_size)
        poly_x, poly_y = list(zip(*corners))
        map_geometry[hex_id] = (cx, cy, poly_x, poly_y)
        all_corners_x.extend(poly_x)
        all_corners_y.extend(poly_y)

    map_xlim, map_ylim = None, None
    if all_corners_x:
        margin = hex_size
        map_xlim = (min(all_corners_x) - margin, max(all_corners_x) + margin)
        map_ylim = (min(all_corners_y) - margin, max(all_corners_y) + margin)

    # --- Main Visualization Loop ---
    for t in timesteps_to_process:
        timestep_data = zone_stats_df[zone_stats_df["time"] == t]
        active_hex_data = timestep_data.set_index('hex_id').to_dict('index')

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect("equal")
        ax.set_title(f"Traffic Map — Timestep {t}", fontsize=16)
        ax.axis("off")

        # Loop through the FULL MAP (all 182 zones)
        for hex_id, (cx, cy, poly_x, poly_y) in map_geometry.items():
            label_text = None
            if hex_id in active_hex_data:
                # This hex has data, color it
                row_data = active_hex_data[hex_id]
                try:
                    label_value = int(row_data["label"])
                except (ValueError, TypeError):
                    label_value = "default"
                color_hex = color_map.get(label_value, DEFAULT_COLOR)

                if add_labels:
                    try:
                        label_text = str(row_data[label_column])
                    except KeyError:
                        add_labels = False
            else:
                # This hex has no data, color it gray
                color_hex = DEFAULT_COLOR

            ax.fill(poly_x, poly_y, color=color_hex, edgecolor="black", alpha=0.8)

            if add_labels and label_text is not None:
                text_color = 'white' if color_hex in ["#F44336", "#380000"] else 'black'
                ax.text(cx, cy, label_text,
                        ha='center', va='center',
                        fontsize=6, color=text_color, fontweight='bold')

        if map_xlim and map_ylim:
            ax.set_xlim(map_xlim)
            ax.set_ylim(map_ylim)

        out_path = os.path.join(output_dir, f"traffic_map_timestep_{t:04d}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"\n✅ All {len(timesteps_to_process)} maps saved to '{output_dir}/' folder.")


# -----------------------------
# 5️⃣ Main block to run the visualization
#    (❗️ NEW LOGIC: Replicates your code snippet)
# -----------------------------
if __name__ == "__main__":

    # --- ⚙️ User Configuration ---
    CSV_FILE_PATH = "final.csv"
    OUTPUT_FOLDER = "traffic_maps_by_label"
    COLUMN_FOR_TEXT_LABEL = "label"
    START_TIMESTEP = 1
    END_TIMESTEP = 10

    # --- 🗺️ Map Definition (from your new code) ---
    MAP_WIDTH = 4300
    MAP_HEIGHT = 3400
    HEX_SIZE = 200  # This is the 'radius' in your example

    # --- ❗️ Re-implementing your grid logic ---

    # 1. Calculate hex dimensions (for pointy-top hexes)
    hex_width = math.sqrt(3) * HEX_SIZE
    hex_height = 2 * HEX_SIZE
    row_spacing = 0.75 * hex_height  # (This is 1.5 * HEX_SIZE)

    # 2. Calculate max rows and columns
    max_rows = int(math.ceil(MAP_HEIGHT / row_spacing))
    max_cols = int(math.ceil(MAP_WIDTH / hex_width))

    # 3. Generate all hexes
    all_hex_ids_on_map = set()
    print(f"Generating hex grid with max_row={max_rows}, max_col={max_cols}...")

    for row in range(0, max_rows + 1):
        row_is_even = (row % 2 == 0)
        start_x = 0.0 if row_is_even else (hex_width / 2.0)
        center_y = row * row_spacing

        for col in range(0, max_cols + 1):
            center_x = start_x + col * hex_width

            # Convert the (x, y) center to a "q_r" hex_id
            hex_id = xy_to_hex(center_x, center_y, HEX_SIZE)
            all_hex_ids_on_map.add(hex_id)

    # --- End of New Logic ---

    # Check: (12+1) * (13+1) = 13 * 14 = 182 zones. This matches your number.
    print(f"Created a complete grid with {len(all_hex_ids_on_map)} total hex zones.")

    # --- Run Visualization ---
    try:
        print(f"Loading data from {CSV_FILE_PATH}...")
        all_stats_df = pd.read_csv(CSV_FILE_PATH)
        print(f"Loaded {len(all_stats_df)} records.")
        print("Starting visualization...")

        visualize_traffic_from_csv(
            all_stats_df,
            hex_size=HEX_SIZE,
            all_hex_ids=list(all_hex_ids_on_map),  # Pass the new 182-zone grid
            output_dir=OUTPUT_FOLDER,
            add_labels=True,
            label_column=COLUMN_FOR_TEXT_LABEL,
            start_time=START_TIMESTEP,
            end_time=END_TIMESTEP
        )

    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
    except KeyError as e:
        print(f"Error: A required column is missing. {e}")
    except Exception as e:
        print(f"An error occurred: {e}")