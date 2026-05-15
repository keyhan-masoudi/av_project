import pickle
import os
import sys
from config import Config


sys.path.append(os.path.abspath(Config.Paths.NoiseConfigsPath))
from NoiseConfigs.utilsFunctions import UtilsFunc
from config import Config


def generate_cache_file():
    pkl_dest_dir = Config.Paths.pklPath
    csv_source_dir = os.path.join(pkl_dest_dir, "Outputs2")
    pkl_filename = "./traffic_processed_data.pkl"
    full_pkl_path = os.path.join(pkl_dest_dir, pkl_filename)

    print("Loading partitions...")
    partitions = UtilsFunc.load_partitions("generated_hex_partitions")

    traffic_cache = {}
    duration = int(Config.SimulatorConfig.SIMULATION_DURATION)

    print(f"Start processing from: {csv_source_dir}")

    for t in range(duration + 1):
        csv_file_path = os.path.join(csv_source_dir, f"dataInTime{t}.csv")

        if os.path.exists(csv_file_path):
            try:
                raw_data = UtilsFunc.recognize_traffic_status(csv_file_path, partitions)

                safe_data = {}
                if isinstance(raw_data, dict):
                    for partition_obj, status in raw_data.items():
                        key_name = partition_obj.__class__.__name__
                        safe_data[key_name] = status

                traffic_cache[t] = safe_data

                if t == 0:
                    print(f"Sample Safe Data at t=0: {list(safe_data.items())[:2]}")

            except Exception as e:
                print(f"Error processing time {t}: {e}")

        if t % 50 == 0:
            print(f"Time {t}/{duration} processed...")

    print(f"Saving cache to: {full_pkl_path}")
    with open(full_pkl_path, "wb") as f:
        pickle.dump(traffic_cache, f)
    print("Done.")


if __name__ == "__main__":
    generate_cache_file()