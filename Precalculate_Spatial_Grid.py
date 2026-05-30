import os
import sys
import pickle
from config import Config

# Append project paths to ensure correct module loading
sys.path.append(Config.VehiclesTraffic.PROJECT_ROOT)
sys.path.append(Config.VehiclesTraffic.NOISE_CONFIGS_PATH)

try:
    from NoiseConfigs.utilsFunctions import UtilsFunc
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Resolution of the spatial grid in meters
GRID_SIZE = 5
# Safety buffer around the map boundaries to prevent out-of-bounds errors
BUFFER = 100


def main():
    print("Loading partitions for spatial grid initialization...")
    partitions = UtilsFunc.load_partitions("generated_hex_partitions")

    if not partitions:
        print("No partitions found! Please check your configuration.")
        return

    # Exact map boundaries based on the simulated environment
    min_x = 29.24 - BUFFER
    max_x = 2128.64 + BUFFER
    min_y = 30.38 - BUFFER
    max_y = 1586.68 + BUFFER

    print(f"Map boundaries (including buffer): X({min_x} to {max_x}), Y({min_y} to {max_y})")
    print(f"Grid resolution: {GRID_SIZE} meters")

    spatial_cache = {}

    # Convert real coordinates to grid indices
    start_gx = int(min_x // GRID_SIZE)
    end_gx = int(max_x // GRID_SIZE) + 1
    start_gy = int(min_y // GRID_SIZE)
    end_gy = int(max_y // GRID_SIZE) + 1

    total_cells = (end_gx - start_gx) * (end_gy - start_gy)
    print(f"Calculating nearest partitions for {total_cells} grid cells. Please wait...")

    processed = 0
    for gx in range(start_gx, end_gx):
        for gy in range(start_gy, end_gy):
            # Calculate the exact center coordinate of the current grid cell
            real_x = (gx * GRID_SIZE) + (GRID_SIZE / 2.0)
            real_y = (gy * GRID_SIZE) + (GRID_SIZE / 2.0)

            # Mathematical calculation to find the nearest partition (Euclidean distance)
            nearest_p = min(partitions, key=lambda p: (p.centerX - real_x) ** 2 + (p.centerY - real_y) ** 2)

            # Store only the class name of the partition to optimize memory usage
            spatial_cache[(gx, gy)] = nearest_p.__class__.__name__

            processed += 1
            if processed % 50000 == 0:
                print(f"Processed {processed} / {total_cells} cells...")

    # Define the output path for the Pickle file
    output_file = os.path.join(Config.VehiclesTraffic.PROJECT_ROOT, "precalculated_spatial_grid.pkl")

    # Serialize and save the dictionary
    with open(output_file, "wb") as f:
        pickle.dump(spatial_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n[SUCCESS] Spatial grid successfully calculated and saved to: {output_file}")


if __name__ == "__main__":
    main()
