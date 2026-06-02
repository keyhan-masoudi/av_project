import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math

# -----------------------------
# 1️⃣ Parse XML
# -----------------------------
def parse_xml_to_df(xml_path, max_time=None):
    """Parse vehicle data from XML up to max_time."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    records = []
    for timestep in root.findall("timestep"):
        time = int(float(timestep.get("time")))
        if max_time is not None and time > max_time:
            break
        for vehicle in timestep.findall("vehicle"):
            records.append((
                time,
                vehicle.get("id"),
                float(vehicle.get("x")),
                float(vehicle.get("y")),
                float(vehicle.get("speed"))
            ))
    df = pd.DataFrame(records, columns=["time", "vehicle_id", "x", "y", "speed"])
    return df

# -----------------------------
# 2️⃣ Hex zone conversion
# -----------------------------
def xy_to_hex(x, y, size):
    """Convert (x, y) to a hex grid coordinate."""
    q = (np.sqrt(3)/3 * x - 1/3 * y) / size
    r = (2/3 * y) / size
    rq = round(q)
    rr = round(r)
    return f"{rq}_{rr}"

def add_hex_zones(df, hex_size):
    """Add hex_id column to dataframe."""
    df["hex_id"] = df.apply(lambda row: xy_to_hex(row.x, row.y, hex_size), axis=1)
    return df

# -----------------------------
# 3️⃣ Aggregate per timestep & zone
# -----------------------------
def aggregate_zone_stats(df):
    """Aggregate number of vehicles and avg speed per zone per timestep."""
    return (
        df.groupby(["time", "hex_id"])
        .agg(num_vehicles=("vehicle_id", "count"), avg_speed=("speed", "mean"))
        .reset_index()
    )

# -----------------------------
# 4️⃣ Determine traffic color
# -----------------------------
def traffic_color(count):
    """Assign traffic color based on number of vehicles."""
    if count <= 15:
        return "green"
    elif count <= 30:
        return "yellow"
    elif count <= 45:
        return "orange"
    else:
        return "red"

def add_traffic_colors(df):
    """Add traffic_color column."""
    df["traffic_color"] = df["num_vehicles"].apply(traffic_color)
    return df

# -----------------------------
# 5️⃣ Main function for analysis
# -----------------------------
def analyze_traffic(xml_path, hex_size, target_timestep=None):
    """Full pipeline: parse → hex map → aggregate → colorize."""
    df = parse_xml_to_df(xml_path, max_time=target_timestep)
    print(f"✅ Parsed up to timestep {df['time'].max()} with {len(df)} records")

    df = add_hex_zones(df, hex_size)
    zone_stats = aggregate_zone_stats(df)
    zone_stats = add_traffic_colors(zone_stats)

    # Only print the target timestep if provided
    if target_timestep is not None:
        sample = zone_stats[zone_stats["time"] == target_timestep]
        print(f"\n🚦 Traffic at timestep {target_timestep}")
        print(sample[["hex_id", "num_vehicles", "traffic_color"]])

    return zone_stats  # <-- return ALL timesteps, not just one

# -----------------------------
# 🔷 Helper: compute hex corners
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
# 🔷 Convert hex_id back to coordinates
# -----------------------------
def hex_to_xy(hex_id, size):
    """Convert hex axial coordinate (q_r) back to (x, y) for visualization."""
    q, r = map(int, hex_id.split("_"))
    x = size * math.sqrt(3) * (q + r/2)
    y = size * 3/2 * r
    return x, y

# -----------------------------
# 🔷 Visualization function
# -----------------------------
def visualize_traffic_map(zone_stats, hex_size, output_dir="traffic_maps"):
    """
    Draw colored hexagons for each zone at each timestep.
    Saves PNG per timestep to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define traffic color mapping
    color_map = {
        "green": "#4CAF50",
        "yellow": "#FFEB3B",
        "orange": "#FF9800",
        "red": "#F44336"
    }

    timesteps = sorted(zone_stats["time"].unique())

    for t in timesteps:
        if t <= 130:
            continue  # skip timesteps <= 30

        timestep_data = zone_stats[zone_stats["time"] == t]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect("equal")
        ax.set_title(f"Traffic Map — Timestep {t}", fontsize=14)
        ax.axis("off")

        for _, row in timestep_data.iterrows():
            cx, cy = hex_to_xy(row["hex_id"], hex_size)
            corners = hex_corners(cx, cy, hex_size)
            poly_x, poly_y = zip(*corners)
            color = color_map.get(row["traffic_color"], "gray")
            ax.fill(poly_x, poly_y, color=color, edgecolor="black", alpha=0.7)

        # Save one image per timestep (only > 30)
        out_path = os.path.join(output_dir, f"traffic_map_timestep_{t}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"✅ Saved: {out_path}")


# -----------------------------
# 6️⃣ Run example
# -----------------------------
if __name__ == "__main__":
    xml_file = "../data/vehicles/chunk_0.xml"
    hex_size = 250
    max_time = 150  # choose how many timesteps to parse
    # get all timesteps
    all_stats = analyze_traffic(xml_file, hex_size, target_timestep=max_time)
    # save overall data
    all_stats.to_csv("traffic_zones_all.csv", index=False)
    print("✅ Saved all timesteps to traffic_zones_all.csv")
    # visualize every timestep
    visualize_traffic_map(all_stats, hex_size, output_dir="traffic_maps")
    print("\n✅ All timestep maps saved to 'traffic_maps/' folder.")
