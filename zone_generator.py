# ==========================================================
# PART 1 — Parse SUMO XML → aggregate zone stats → fixed label → CSV
# ==========================================================
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import math

# -----------------------------
# Config
# -----------------------------
xml_path = "chunk_0.xml"     # path to your SUMO XML file
output_csv = "new_traffic_dataset.csv"
hex_size = 250                # adjust to your map size
# label thresholds
THRESHOLDS = [15, 30, 45, 60]   # <=15->1, <=30->2, <=45->3, <=60->4, >60->5

# -----------------------------
# Helpers
# -----------------------------
def point_to_axial(x, y, size):
    """Convert (x, y) to axial hex coordinates."""
    q = (math.sqrt(3)/3 * x - 1/3 * y) / size
    r = (2/3 * y) / size
    xh, zh, yh = q, r, -q - r
    rx, ry, rz = round(xh), round(yh), round(zh)
    xdiff, ydiff, zdiff = abs(rx - xh), abs(ry - yh), abs(rz - zh)
    if xdiff > ydiff and xdiff > zdiff:
        rx = -ry - rz
    elif ydiff > zdiff:
        ry = -rx - rz
    return int(rx), int(rz)

def axial_key(q, r):
    return f"{q}_{r}"

def label_from_count(n):
    """Map vehicle count to discrete traffic level 1–5."""
    if n <= THRESHOLDS[0]:
        return 1  # green
    elif n <= THRESHOLDS[1]:
        return 2  # yellow
    elif n <= THRESHOLDS[2]:
        return 3  # orange
    elif n <= THRESHOLDS[3]:
        return 4  # red
    else:
        return 5  # black (heavy congestion)

# -----------------------------
# Parse XML
# -----------------------------
def parse_xml_to_df(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rec = []
    for ts in root.findall("timestep"):
        t = int(float(ts.get("time", "0")))
        for v in ts.findall("vehicle"):
            try:
                x = float(v.get("x", "0"))
                y = float(v.get("y", "0"))
                sp = float(v.get("speed", "0"))
                ang = float(v.get("angle", "0"))
            except:
                continue
            rec.append((t, v.get("id"), x, y, sp, ang))
    df = pd.DataFrame(rec, columns=["time","vehicle_id","x","y","speed","angle"])
    return df

# -----------------------------
# Aggregate per timestep & zone
# -----------------------------
def aggregate_zone_stats(df, hex_size):
    hex_ids = []
    for _, row in df.iterrows():
        q, r = point_to_axial(row.x, row.y, hex_size)
        hex_ids.append(axial_key(q, r))
    df["hex_id"] = hex_ids
    df["angle_rad"] = np.radians(df["angle"])
    df["angle_sin"] = np.sin(df["angle_rad"])
    df["angle_cos"] = np.cos(df["angle_rad"])

    zone_stats = (
        df.groupby(["time", "hex_id"])
          .agg(num_vehicles=("vehicle_id", "count"),
               avg_speed=("speed", "mean"),
               avg_sin=("angle_sin", "mean"),
               avg_cos=("angle_cos", "mean"))
          .reset_index()
          .sort_values(["time", "hex_id"])
          .reset_index(drop=True)
    )

    # assign label using fixed thresholds
    zone_stats["label"] = zone_stats["num_vehicles"].apply(label_from_count)
    return zone_stats

# -----------------------------
# Main
# -----------------------------
print(f"Parsing XML file: {xml_path}")
df = parse_xml_to_df(xml_path)
print(f"→ Parsed {len(df)} vehicle entries across {df['time'].nunique()} timesteps.")

zone_stats = aggregate_zone_stats(df, hex_size)
print(f"→ Aggregated {len(zone_stats)} zone-timestep rows over {zone_stats['hex_id'].nunique()} zones.")

# ensure sorted integer timesteps
zone_stats["time"] = zone_stats["time"].astype(int)
zone_stats = zone_stats.sort_values(["time","hex_id"]).reset_index(drop=True)

# save
zone_stats.to_csv(output_csv, index=False)
print(f"\n✅ Saved processed dataset → {output_csv}")