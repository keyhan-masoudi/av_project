import os
import glob
import pickle
import xml.etree.ElementTree as ET
import sys
from config import Config

sys.path.append(Config.VehiclesTraffic.PROJECT_ROOT)
sys.path.append(Config.VehiclesTraffic.NOISE_CONFIGS_PATH)

try:
    from NoiseConfigs.utilsFunctions import UtilsFunc
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def get_partition_by_location(x: float, y: float, partitions: list):
    return min(partitions, key=lambda p: (p.centerX - x) ** 2 + (p.centerY - y) ** 2)


def extract_name(obj):
    if isinstance(obj, str):
        return obj
    if hasattr(obj, '__name__'):
        return obj.__name__
    if hasattr(obj, '__class__'):
        return obj.__class__.__name__
    return str(obj)


def read_and_print_from_pickle(target_time=None):
    pickle_path = os.path.join(Config.VehiclesTraffic.PROJECT_ROOT, "precalculated_vehicle_traffic.pkl")

    if not os.path.exists(pickle_path):
        print(f"Error: Pickle file not found at {pickle_path}")
        return

    print("\n**************************************************")
    print(" READING AND PRINTING DATA FROM SAVED PICKLE FILE ")
    print("**************************************************")

    with open(pickle_path, "rb") as f:
        saved_data = pickle.load(f)

    time_steps_to_print = [target_time] if target_time is not None else sorted(saved_data.keys())

    for time_step in time_steps_to_print:
        if time_step not in saved_data:
            print(f"\n[ Time Step: {time_step}s ] : No data found in the file.")
            continue

        print(f"\n[ Time Step: {time_step}s ]")

        for veh_id, info in saved_data[time_step].items():
            zone = info['zone']
            traffic = info['traffic']
            intensity = info['intensity']
            print(f"  Vehicle: {veh_id:<8} | Zone: {zone:<12} | Traffic: {traffic:<15} | Intensity: {intensity}")


def read_and_print_dynamic_traffic(target_time=None):
    pkl_dest_dir = Config.Paths.pklPath
    pkl_filename = "traffic_processed_data.pkl"
    pickle_path = os.path.join(pkl_dest_dir, pkl_filename)

    vehicle_pickle_path = os.path.join(Config.VehiclesTraffic.PROJECT_ROOT, "precalculated_vehicle_traffic.pkl")
    vehicle_data = {}
    if os.path.exists(vehicle_pickle_path):
        with open(vehicle_pickle_path, "rb") as vf:
            vehicle_data = pickle.load(vf)

    if not os.path.exists(pickle_path):
        print(f"Error: Dynamic traffic Pickle file not found at {pickle_path}")
        return

    print("\n**************************************************")
    print(" READING DYNAMIC TRAFFIC DATA (traffic_processed_data.pkl) ")
    print("**************************************************")

    with open(pickle_path, "rb") as f:
        saved_data = pickle.load(f)

    time_steps_to_print = [target_time] if target_time is not None else sorted(saved_data.keys())

    for time_step in time_steps_to_print:
        if time_step not in saved_data:
            print(f"\n[ Time Step: {time_step}s ] : No data found in the file.")
            continue

        print(f"\n[ Time Step: {time_step}s ]")

        zone_counts = {}
        if time_step in vehicle_data:
            for veh_id, info in vehicle_data[time_step].items():
                zone_name = info['zone']
                zone_counts[zone_name] = zone_counts.get(zone_name, 0) + 1

        for partition_key, traffic_status in saved_data[time_step].items():
            p_name_str = extract_name(partition_key)
            t_status_str = extract_name(traffic_status)

            vehicle_count = zone_counts.get(p_name_str, 0)

            print(
                f"  Partition: {p_name_str:<15} | Traffic Status: {t_status_str:<27} | Vehicles Count: {vehicle_count}")

def main():
    print(f"Loading partitions exactly from: {Config.VehiclesTraffic.NOISE_CONFIGS_PATH} ...")
    partitions = UtilsFunc.load_partitions("generated_hex_partitions")

    traffic_cache_path = os.path.join(Config.Paths.pklPath, "traffic_processed_data.pkl")

    dynamic_traffic = {}
    if os.path.exists(traffic_cache_path):
        with open(traffic_cache_path, "rb") as f:
            dynamic_traffic = pickle.load(f)
        print(f"Loaded dynamic traffic cache from {traffic_cache_path}")
    else:
        raise FileNotFoundError(f"Dynamic traffic file not found at {traffic_cache_path}!")

    mapping = {
        'GreenTraffic': 0.0,
        'YellowTraffic': 0.25,
        'OrangeTraffic': 0.5,
        'RedTraffic': 0.75,
        'BlackTraffic': 1.0
    }

    vehicles_dir = os.path.join(Config.VehiclesTraffic.PROJECT_ROOT, "data", "vehicles")
    xml_files = glob.glob(os.path.join(vehicles_dir, "*.xml"))

    if not xml_files:
        print(f"No XML files found in {vehicles_dir}")
        return

    vehicle_traffic_cache = {}

    for xml_file in xml_files:
        print(f"Processing {xml_file} ...")
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for timestep_elem in root.findall('timestep'):
            time_val = int(float(timestep_elem.get('time')))

            if time_val not in vehicle_traffic_cache:
                vehicle_traffic_cache[time_val] = {}

            time_traffic_dict = None
            if time_val in dynamic_traffic:
                time_traffic_dict = dynamic_traffic[time_val]
            elif str(time_val) in dynamic_traffic:
                time_traffic_dict = dynamic_traffic[str(time_val)]
            elif float(time_val) in dynamic_traffic:
                time_traffic_dict = dynamic_traffic[float(time_val)]

            if time_traffic_dict is None:
                raise ValueError(f"CRITICAL ERROR: Time step {time_val} does NOT exist in traffic_processed_data.pkl!")

            normalized_traffic_dict = {}
            for k, v in time_traffic_dict.items():
                key_str = extract_name(k)
                val_str = extract_name(v)
                normalized_traffic_dict[key_str] = val_str

            for vehicle_elem in timestep_elem.findall('vehicle'):
                veh_id = vehicle_elem.get('id')
                x = float(vehicle_elem.get('x'))
                y = float(vehicle_elem.get('y'))

                p = get_partition_by_location(x, y, partitions)
                p_name = p.__class__.__name__

                if p_name in normalized_traffic_dict:
                    traffic_name = normalized_traffic_dict[p_name]
                else:
                    traffic_name = extract_name(p.trafficStatus)

                intensity = -1
                for k, v in mapping.items():
                    if k in traffic_name:
                        intensity = v
                        break

                vehicle_traffic_cache[time_val][veh_id] = {
                    'zone': p_name,
                    'traffic': traffic_name,
                    'intensity': intensity
                }

    output_file = os.path.join(Config.VehiclesTraffic.PROJECT_ROOT, "precalculated_vehicle_traffic.pkl")

    with open(output_file, "wb") as f:
        pickle.dump(vehicle_traffic_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✅ Vehicle traffic data successfully calculated and saved to: {output_file}")


if __name__ == "__main__":
    # main()
    read_and_print_dynamic_traffic()
    # read_and_print_from_pickle()
