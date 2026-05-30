import os
import pickle
import random
import sys
from config import Config

sys.path.append(Config.VehiclesTraffic.PROJECT_ROOT)
sys.path.append(Config.VehiclesTraffic.NOISE_CONFIGS_PATH)

try:
    from NoiseConfigs.utilsFunctions import UtilsFunc
    from NoiseConfigs.noiseConfigGeneralAttribute import NoiseConfigGeneralAttribute
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

RAIN_PERIODS = [(700, 900), (1500, 1700), (3000, 3300)]
SIMULATION_DURATION = 3600


def is_rain_period(t):
    for start, end in RAIN_PERIODS:
        if start <= t <= end:
            return True
    return False


def main():
    print("Loading partitions for weather...")
    partitions = UtilsFunc.load_partitions("generated_hex_partitions")
    p_names = [p.__class__.__name__ for p in partitions]

    weather_cache = {}

    raw_options = NoiseConfigGeneralAttribute.Rain_options
    weather_options = [s.replace("NoiseConfig.", "").replace("()", "") for s in raw_options]

    sun_weather = weather_options[0]
    rain_options = weather_options[1:]

    random.seed(12345)

    current_weather = {p: sun_weather for p in p_names}

    print("Calculating rain states for each second...")
    for t in range(SIMULATION_DURATION + 1):
        if not is_rain_period(t):
            for p in p_names:
                current_weather[p] = sun_weather
        else:
            for p in p_names:
                if random.random() < 0.30:
                    current_weather[p] = random.choice(rain_options)
                elif current_weather[p] == sun_weather:
                    current_weather[p] = random.choice(rain_options)

        weather_cache[t] = current_weather.copy()

        if t % 500 == 0:
            print(f"Processed t={t} / {SIMULATION_DURATION}")

    output_file = os.path.join(Config.VehiclesTraffic.PROJECT_ROOT, "precalculated_weather.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(weather_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✅ Weather successfully calculated and saved to: {output_file}")


if __name__ == "__main__":
    main()
