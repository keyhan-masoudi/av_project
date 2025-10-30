import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
from NoiseConfigs.utilsFunctions import UtilsFunc  # Import your utility functions
import os

# --- Settings ---
# Place the name of your XML file here
TRAFFIC_XML_FILE = 'chunk_0.xml'
# Name of the output Excel file
OUTPUT_EXCEL_FILE = 'traffic_analysis_report.xlsx'


def get_traffic_state_name(count: int) -> str:
    """
    Determines the name of the traffic state based on the vehicle count.
    This logic is based on the recognize_traffic_status function in utilsFunctions.py.
    """
    if count < 10:
        return 'Green'
    elif count < 20:
        return 'Orange'
    elif count < 40:
        return 'Red'
    else:
        return 'Black'


def analyze_traffic_xml(xml_path: str, partitions: list) -> pd.DataFrame:
    """
    Parses the traffic XML file and counts the occurrence of each traffic state
    for each partition.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Error: The XML file '{xml_path}' was not found.")

    # Data structure to store the counts: {partition: {'Green': 10, 'Orange': 5, ...}}
    partition_stats = defaultdict(lambda: defaultdict(int))

    print(f"Analyzing XML file: {xml_path}...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    timesteps = root.findall('timestep')
    total_timesteps = len(timesteps)

    for i, timestep in enumerate(timesteps):
        if (i + 1) % 100 == 0:
            print(f"Processing timestep {i + 1}/{total_timesteps}...")

        # 1. Collect vehicle positions in this timestep
        vehicle_positions = []
        for vehicle in timestep.findall('vehicle'):
            try:
                x = float(vehicle.get('x'))
                y = float(vehicle.get('y'))
                vehicle_positions.append((x, y))
            except (ValueError, TypeError):
                # Ignore vehicles without valid coordinates
                continue

        # --- MODIFICATION START ---
        # Only process the timestep if there is at least one vehicle on the map.
        # This prevents empty timesteps from being counted as 'Green' for all partitions.
        if vehicle_positions:
            # 2. Count vehicles per partition for this timestep
            vehicles_per_partition = defaultdict(int)
            for x, y in vehicle_positions:
                partition = UtilsFunc.find_partition(partitions, x, y)
                if partition:
                    vehicles_per_partition[partition] += 1

            # 3. Determine the state of each partition and update stats
            for partition in partitions:
                count = vehicles_per_partition.get(partition, 0)
                state = get_traffic_state_name(count)
                partition_stats[partition][state] += 1
        # If vehicle_positions is empty, we do nothing for this timestep.
        # --- MODIFICATION END ---


    print("Analysis complete. Preparing the report...")

    # 4. Convert the data to a suitable format for the DataFrame
    data_for_df = []
    for i, partition in enumerate(partitions):
        counts = partition_stats.get(partition, {})
        row = {
            'partition_id': f'Partition_{i + 1}',
            'center_x': partition.centerX,
            'center_y': partition.centerY,
            'Green_Count': counts.get('Green', 0),
            'Orange_Count': counts.get('Orange', 0),
            'Red_Count': counts.get('Red', 0),
            'Black_Count': counts.get('Black', 0)
        }
        data_for_df.append(row)

    return pd.DataFrame(data_for_df)


if __name__ == "__main__":
    # Ensure the generated partitions file exists
    if not os.path.exists("generated_hex_partitions.py"):
        print("Error: The file 'generated_hex_partitions.py' was not found.")
        print("Please run the 'HexPartitionGenerator.py' script first.")
    else:
        # Load the partition definitions
        print("Loading partition definitions...")
        all_partitions = UtilsFunc.load_partitions("generated_hex_partitions")

        if not all_partitions:
            print("No partitions were loaded. Check the 'generated_hex_partitions.py' file.")
        else:
            try:
                # Run the analysis
                df_results = analyze_traffic_xml(TRAFFIC_XML_FILE, all_partitions)

                # Save the DataFrame to an Excel file
                df_results.to_excel(OUTPUT_EXCEL_FILE, index=False, engine='openpyxl')
                print(f"🎉 Report successfully saved to '{OUTPUT_EXCEL_FILE}'")

            except FileNotFoundError as e:
                print(e)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")