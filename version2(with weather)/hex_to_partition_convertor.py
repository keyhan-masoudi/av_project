import math
import numpy as np
import pandas as pd
import pickle

# --- New Map Configuration Bounding Box ---
MIN_X = 29.24
MAX_X = 2128.64
MIN_Y = 30.38
MAX_Y = 1586.68

HEX_SIZE = 200  # Your hexagon radius

def _xy_to_axial(x, y, size):
    """
    Converts (x, y) cartesian coordinates to (q, r) axial hex coordinates.
    Matches your visualizer's internal geometry engine perfectly.
    """
    q_f = (np.sqrt(3) / 3 * x - 1 / 3 * y) / size
    r_f = (2 / 3 * y) / size

    # Round to nearest hex
    q = round(q_f)
    r = round(r_f)
    s = round(-q_f - r_f)

    # Correct for rounding errors
    q_diff = abs(q - q_f)
    r_diff = abs(r - r_f)
    s_diff = abs(s - (-q_f - r_f))

    if q_diff > r_diff and q_diff > s_diff:
        q = -r - s
    elif r_diff > s_diff:
        r = -q - s

    return (int(q), int(r))

def build_index_lookup():
    """
    Generates the exact same node_to_idx dictionary as your build_hex_graph
    function using your map's active minimum and maximum boundaries.
    """
    unique_hex_coords_set = set()

    hex_width = math.sqrt(3) * HEX_SIZE
    row_spacing = 1.5 * HEX_SIZE

    # Determine structural grid loops with a padding buffer
    min_row = int(math.floor(MIN_Y / row_spacing)) - 1
    max_row = int(math.ceil(MAX_Y / row_spacing)) + 1
    
    min_col = int(math.floor(MIN_X / hex_width)) - 1
    max_col = int(math.ceil(MAX_X / hex_width)) + 1

    for row in range(min_row, max_row + 1):
        row_is_even = (row % 2 == 0)
        start_x = 0.0 if row_is_even else (hex_width / 2.0)
        center_y = row * row_spacing

        for col in range(min_col, max_col + 1):
            center_x = start_x + col * hex_width

            # Restrict grid generation to the map's bounding dimensions
            if (MIN_X - hex_width/2 <= center_x <= MAX_X + hex_width/2) and \
               (MIN_Y - row_spacing/2 <= center_y <= MAX_Y + row_spacing/2):
                
                q, r = _xy_to_axial(center_x, center_y, HEX_SIZE)
                unique_hex_coords_set.add((q, r))

    # Sort to create the identical integer index map sequence used by PyTorch Geometric
    unique_hex_coords = sorted(list(unique_hex_coords_set))
    node_to_idx = {h: i for i, h in enumerate(unique_hex_coords)}
    return node_to_idx

def hex_id_to_pixel(hex_id, size):
    """
    Converts historical 'col_row' string format back to geometric (x, y) pixels.
    Uses old format math to extract original centers before matching with new grid.
    """
    parts = str(hex_id).split('_')
    col = int(parts[0])
    row = int(parts[1])
    
    # Revert legacy odd-r offset coordinates back to standard axial (q, r)
    r = row
    q = col - math.floor(r / 2.0)
    
    # Revert axial coordinates back to original pixel centers
    x = size * (math.sqrt(3) * q + math.sqrt(3) / 2.0 * r)
    y = size * (3.0 / 2.0 * r)
    
    return x, y

def convert_csv_to_pickle(input_csv, output_pickle):
    # 1. Build the network map configuration matching your updated bounds
    print("Pre-calculating graph structure index mappings...")
    node_to_idx = build_index_lookup()
    print(f"Index mapping created successfully with {len(node_to_idx)} active nodes.")
    
    # 2. Load the targeted dataset file
    df = pd.read_csv(input_csv)
    partition_ids = []
    
    # 3. Translate historical hex_ids to matching graph indexes
    print("Transforming CSV records...")
    for hex_id in df['hex_id']:
        try:
            # Step A: Find its original pixel positions 
            x, y = hex_id_to_pixel(hex_id, HEX_SIZE)
            
            # Step B: Pass pixel coordinates through your updated axial math
            q, r = _xy_to_axial(x, y, HEX_SIZE)
            axial_coord = (q, r)
            
            # Step C: Match against the sorted graph index layout map
            if axial_coord in node_to_idx:
                # MODIFIED: Convert 0-indexed integer to 1-indexed "partitionX" string format
                partition_label = f"partition{node_to_idx[axial_coord] + 1}"
                partition_ids.append(partition_label)
            else:
                # Discard coordinates completely out of bounds of the downscaled map
                partition_ids.append(None)
        except Exception:
            partition_ids.append(None)
            
    # 4. Filter records outside the downscaled dimensions map layout
    df['hex_id'] = partition_ids
    total_before = len(df)
    df = df.dropna(subset=['hex_id'])
    
    # MODIFIED: Forced type as string instead of int to preserve formatting
    df['hex_id'] = df['hex_id'].astype(str)
    
    # 5. Export clean exact structural dataframe layout directly to Pickle format
    with open(output_pickle, 'wb') as f:
        pickle.dump(df, f)
        
    print(f"Success! Preserved {len(df)}/{total_before} lines inside new bounds.")
    print(f"Saved exact columns layout to: {output_pickle}")

# --- Execution ---
input_csv_file = "predictions_output300.csv"   # Path to your current CSV file
output_pkl_file = "hex_data.pkl"               # Output destination

convert_csv_to_pickle(input_csv_file, output_pkl_file)