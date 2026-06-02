import os
import sys
import math
import pickle
import glob
import xml.etree.ElementTree as ET
from config import Config

# Append project paths
sys.path.append(Config.VehiclesTraffic.PROJECT_ROOT)
sys.path.append(Config.VehiclesTraffic.NOISE_CONFIGS_PATH)


def main():
    # 1. Load the spatial grid cache for O(1) partition lookups
    grid_path = os.path.join(Config.VehiclesTraffic.PROJECT_ROOT, "precalculated_spatial_grid.pkl")
    print(f"Loading spatial grid from: {grid_path}")
    with open(grid_path, "rb") as f:
        spatial_grid = pickle.load(f)

    # 2. Load the Traffic Prediction data (Pandas DataFrame)
    pred_data_path = os.path.join(Config.VehiclesTraffic.PROJECT_ROOT, "HexTraffic_Prediction_data.pkl")
    print(f"Loading traffic prediction data from: {pred_data_path}")
    with open(pred_data_path, "rb") as f:
        df = pickle.load(f)

    # Convert DataFrame to a fast nested dictionary: {time: {hex_id: label}}
    print("Building fast lookup dictionary for predictions...")
    predictions_dict = {}
    for row in df.itertuples(index=False):
        t = int(row.time)
        hex_id = str(row.hex_id).lower()
        if t not in predictions_dict:
            predictions_dict[t] = {}
        predictions_dict[t][hex_id] = float(row.label)

    # 3. Parse XML files and calculate future paths
    vehicles_dir = os.path.join(Config.VehiclesTraffic.PROJECT_ROOT, "data", "vehicles")
    xml_files = glob.glob(os.path.join(vehicles_dir, "*.xml"))

    if not xml_files:
        print(f"No XML files found in {vehicles_dir}")
        return

    vehicle_predictions_cache = {}
    GRID_SIZE = 5  # Used in spatial grid calculation

    for xml_file in xml_files:
        print(f"Processing XML for future trajectory predictions: {xml_file} ...")
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for timestep_elem in root.findall('timestep'):
            time_val = int(float(timestep_elem.get('time')))

            if time_val not in vehicle_predictions_cache:
                vehicle_predictions_cache[time_val] = {}

            for vehicle_elem in timestep_elem.findall('vehicle'):
                veh_id = vehicle_elem.get('id')
                x = float(vehicle_elem.get('x'))
                y = float(vehicle_elem.get('y'))
                angle = float(vehicle_elem.get('angle'))
                speed = float(vehicle_elem.get('speed'))

                angle_rad = math.radians(angle)
                traffic_values = []

                # Project trajectory for the next 10 seconds (Look-ahead)
                for t_future in range(1, 11):
                    future_time = time_val + t_future

                    # Linear kinematics prediction (Matching RL agent's observation logic)
                    future_x = x + (speed * math.sin(angle_rad) * t_future)
                    future_y = y + (speed * math.cos(angle_rad) * t_future)

                    # Find partition using O(1) grid lookup
                    grid_x = int(future_x // GRID_SIZE)
                    grid_y = int(future_y // GRID_SIZE)
                    p_name = spatial_grid.get((grid_x, grid_y))

                    pred_val = 0.5  # Default fallback

                    if p_name:
                        p_name_lower = p_name.lower()

                        # Exact time match
                        if future_time in predictions_dict and p_name_lower in predictions_dict[future_time]:
                            label = predictions_dict[future_time][p_name_lower]
                            pred_val = min(label / 4.0, 1.0)
                        else:
                            # Fallback: Find closest available future time
                            future_times = [k for k in predictions_dict.keys() if k >= future_time]
                            if future_times:
                                closest_t = min(future_times)
                                if p_name_lower in predictions_dict[closest_t]:
                                    label = predictions_dict[closest_t][p_name_lower]
                                    pred_val = min(label / 4.0, 1.0)

                    traffic_values.append(pred_val)

                # Calculate final stats (Average and Max for the 10-second window)
                if traffic_values:
                    avg_traffic = sum(traffic_values) / len(traffic_values)
                    max_traffic = max(traffic_values)
                else:
                    avg_traffic, max_traffic = 0.5, 0.5

                # Save to vehicle cache
                vehicle_predictions_cache[time_val][veh_id] = {
                    'avg': avg_traffic,
                    'max': max_traffic
                }

    # 4. Save the finalized dictionary to Pickle
    output_file = os.path.join(Config.VehiclesTraffic.PROJECT_ROOT, "precalculated_vehicle_predictions.pkl")
    print(f"Saving vehicle predictions cache to: {output_file} ...")
    with open(output_file, "wb") as f:
        pickle.dump(vehicle_predictions_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("\n✅ Future vehicle predictions successfully cached!")


if __name__ == "__main__":
    main()