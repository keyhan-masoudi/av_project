import pandas as pd
import xml.etree.ElementTree as ET
import sys
import numpy as np # Used for NaN
import time as a_time # To time the script
import math # <-- ADDED: Needed for the new functions

# --- 1. FILENAMES & CONFIGURATION ---

# The XML file with all vehicle movements (e.g., chunk0.xml)
XML_FILE = 'data\\vehicles\chunk_0.xml' 

# The file you just created with your model's predictions
PREDICTED_TRAFFIC_FILE = 'all_predictions_output.csv'

# The CSV file with the REAL (actual) traffic labels.
REAL_TRAFFIC_FILE = 'VANET-Copy\\final.csv' 

# The final output file that will be created
OUTPUT_FILE = 'vehicle_stats_per_timestep.csv'

TIME_WINDOW = 7  # 7-second window (t to t+7)

# --- TIME FILTER CONFIGURATION ---
START_TIMESTEP = 2600
END_TIMESTEP = 3600

# This now matches your original 'hex_size'
HEX_GRID_SIZE = 200 

# --- 2. YOUR CUSTOM FUNCTION (NOW MATCHES SCRIPT 1) ---

def point_to_axial(x, y, size):
    """
    Convert (x, y) to axial hex coordinates (q, r).
    (Copied from your 'Zone Generator' script)
    """
    q_frac = (math.sqrt(3) / 3 * x - 1 / 3 * y) / size
    r_frac = (2 / 3 * y) / size

    # Cube coordinates
    xh, zh, yh = q_frac, r_frac, -q_frac - r_frac

    # Rounding to nearest integer cube coordinate
    rx, ry, rz = round(xh), round(yh), round(zh)
    xdiff, ydiff, zdiff = abs(rx - xh), abs(ry - yh), abs(rz - zh)

    if xdiff > ydiff and xdiff > zdiff:
        rx = -ry - rz
    elif ydiff > zdiff:
        ry = -rx - rz
    else:  # zdiff is largest
        rz = -rx - ry
 
    return int(rx), int(rz)

def coords_to_hex_id(q, r):
    """
    Converts axial (q, r) coordinates to your "odd-r" offset (col, row) string ID.
    (Copied from your 'Zone Generator' script)
    """
    # col = q + floor(r / 2)
    # row = r
    col = q + int(math.floor(r / 2.0))
    row = r
    return f"{col}_{row}"


print("Step 1: Loading CSV data into memory for fast lookup...")
script_start_time = a_time.perf_counter()
try:
    # Load REAL traffic data
    real_df = pd.read_csv(REAL_TRAFFIC_FILE)
    real_df['time'] = real_df['time'].astype(int)
    real_df['hex_id'] = real_df['hex_id'].astype(str)
    real_df.set_index(['hex_id', 'time'], inplace=True)
    real_df.sort_index(inplace=True)

    # Load PREDICTED traffic data
    predicted_df = pd.read_csv(PREDICTED_TRAFFIC_FILE)
    predicted_df['time'] = predicted_df['time'].astype(int)
    predicted_df['hex_id'] = predicted_df['hex_id'].astype(str)
    predicted_df.set_index(['hex_id', 'time'], inplace=True)
    predicted_df.sort_index(inplace=True)
    
except FileNotFoundError as e:
    print(f"Error: Could not find file {e.filename}.")
    print("Please check your FILENAMES at the top of the script.")
    sys.exit(1)
except KeyError as e:
    print(f"Error: CSV file is missing a required column: {e}")
    print("Please ensure your CSVs have 'hex_id', 'time', and 'label' columns.")
    sys.exit(1)

# --- STEP A: PRE-CALCULATION OF ALL STATS ---
print(f"\nStep 2: Pre-calculating all {TIME_WINDOW+1}-second window stats...")
precalc_start_time = a_time.perf_counter()

# This dictionary will be our high-speed lookup
stats_lookup = {}
idx = pd.IndexSlice

# Get a unique list of all zones from both files
all_zones = set(real_df.index.get_level_values(0)) | set(predicted_df.index.get_level_values(0))
print(f"Found {len(all_zones)} unique zones to pre-calculate.")

# Loop ONCE for every combination of zone and time
for zone_id in all_zones:
    for time_t in range(START_TIMESTEP, END_TIMESTEP + 1):
        start_time = time_t
        end_time = time_t + TIME_WINDOW # Slice will be [t, t+1, ..., t+7]
        
        # --- Get Predicted Stats ---
        try:
            predicted_labels = predicted_df.loc[idx[zone_id, start_time:end_time], 'label']
            max_predict = predicted_labels.max() if not predicted_labels.empty else np.nan
            avg_predict = predicted_labels.mean() if not predicted_labels.empty else np.nan
        except KeyError:
            max_predict, avg_predict = np.nan, np.nan

        # --- Get Real Traffic Stats ---
        try:
            real_labels = real_df.loc[idx[zone_id, start_time:end_time], 'label']
            max_traffic = real_labels.max() if not real_labels.empty else np.nan
            avg_traffic = real_labels.mean() if not real_labels.empty else np.nan
        except KeyError:
            max_traffic, avg_traffic = np.nan, np.nan
            
        # Store the pre-calculated result in our dictionary
        stats_lookup[(zone_id, time_t)] = (max_predict, avg_predict, max_traffic, avg_traffic)

precalc_end_time = a_time.perf_counter()
print(f"Pre-calculation complete in {precalc_end_time - precalc_start_time:.2f} seconds.")

# --- STEP B: XML ITERATION & FAST LOOKUP ---
print(f"\nStep 3: Parsing XML file and merging stats...")

output_data = []
current_time = 0.0
last_printed_time = -1 # For progress updates
vehicle_row_count = 0

try:
    for event, elem in ET.iterparse(XML_FILE, events=('start', 'end')):
        if event == 'start':
            if elem.tag == 'timestep':
                current_time = float(elem.attrib['time'])
                
                # Print progress update
                if int(current_time) != last_printed_time and int(current_time) % 50 == 0:
                   print(f"  ... processing timestep {int(current_time)}")
                   last_printed_time = int(current_time)

        elif event == 'end':
            # Check if it's a vehicle in our time range
            if elem.tag == 'vehicle' and START_TIMESTEP <= current_time <= END_TIMESTEP:
                
                # 1. Get data from XML
                vehicle_id = elem.attrib['id']
                x = float(elem.attrib['x'])
                y = float(elem.attrib['y'])
                time_t = int(round(current_time)) 
                
                # --- 2. CALCULATE ZONE (USING THE CORRECT FUNCTIONS) ---
                q, r = point_to_axial(x, y, HEX_GRID_SIZE)
                zone_id = coords_to_hex_id(q, r)
                # --- END OF FIX ---
                
                # 3. Perform a *single fast lookup*
                stats = stats_lookup.get((zone_id, time_t))
                
                if stats:
                    max_predict, avg_predict, max_traffic, avg_traffic = stats
                else:
                    # Fallback if the (zone, time) was somehow not in our lookup
                    max_predict, avg_predict, max_traffic, avg_traffic = np.nan, np.nan, np.nan, np.nan

                # 4. Add the row to our output list
                output_data.append({
                    'vehicles name': vehicle_id,
                    'time': time_t,
                    'max predict': max_predict,
                    'max traffic': max_traffic,
                    'avg predict': avg_predict,
                    'avg traffic': avg_traffic,
                    'x': x,
                    'y': y,
                    'zone id': zone_id
                })
                vehicle_row_count += 1
            
            # Clear the element from memory
            if elem.tag == 'timestep' or elem.tag == 'vehicle':
                 elem.clear()

except ET.ParseError as e:
    print(f"Error parsing XML file: {e}")
    sys.exit(1)
except FileNotFoundError:
    print(f"Error: Could not find XML file: {XML_FILE}.")
    print("Please check your FILENAMES at the top of the script.")
    sys.exit(1)

print("\nStep 4: Saving final CSV file...")
# Create the final DataFrame from all the rows we collected
final_df = pd.DataFrame(output_data)

# Reorder columns to match your request
final_df = final_df[[
    'vehicles name', 
    'time',
    'max predict', 
    'max traffic', 
    'avg predict', 
    'avg traffic', 
    'x', 
    'y', 
    'zone id'
]]

# Save to CSV
final_df.to_csv(OUTPUT_FILE, index=False, na_rep='NaN')

script_end_time = a_time.perf_counter()
print(f"\n✅ Success! Data processed and saved to {OUTPUT_FILE}")
print(f"   Total rows generated: {len(final_df)} (This should match your {vehicle_row_count} vehicle lines)")
print(f"   Total time taken: {script_end_time - script_start_time:.2f} seconds")
print(final_df.head())