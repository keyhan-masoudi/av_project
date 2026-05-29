import pickle
import os
import sys
import glob
import re
from config import Config

sys.path.append(os.path.abspath(Config.Paths.NoiseConfigsPath))
from NoiseConfigs.utilsFunctions import UtilsFunc


def generate_cache_file():
    pkl_dest_dir = Config.Paths.pklPath
    csv_source_dir = os.path.join(pkl_dest_dir, "Outputs2")
    pkl_filename = "./traffic_processed_data.pkl"
    full_pkl_path = os.path.join(pkl_dest_dir, pkl_filename)

    print("Loading partitions...")
    partitions = UtilsFunc.load_partitions("generated_hex_partitions")

    traffic_cache = {}

    print(f"Searching for CSV files in: {csv_source_dir}")
    file_pattern = os.path.join(csv_source_dir, "dataInTime*.csv")
    csv_files = glob.glob(file_pattern)

    if not csv_files:
        print(f"No files found matching pattern 'dataInTime*.csv' in {csv_source_dir}")
        return

    max_t = -1
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        match = re.search(r'dataInTime(\d+)\.csv', filename)
        if match:
            t_val = int(match.group(1))
            if t_val > max_t:
                max_t = t_val

    if max_t == -1:
        print("Could not extract time from file names. Exiting.")
        return

    duration = max_t
    print(f"Found files up to t={duration}. Start processing...")

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
