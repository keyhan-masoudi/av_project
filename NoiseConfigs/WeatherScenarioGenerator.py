# WeatherScenarioGenerator.py

import random
import json
import inspect
from config import Config
from NoiseConfigs.noiseConfigGeneralAttribute import NoiseConfigGeneralAttribute
from NoiseConfigs.noiseConfig import NoiseConfig
from NoiseConfigs.utilsFunctions import UtilsFunc
import generated_hex_partitions
from generated_hex_partitions import Partitions

RAINY_INTERVALS = [
    (300, 600),
    (700, 850)
]

NO_RAIN_STATUS_STR = "NoiseConfig.Rain0()"

def is_time_in_rainy_interval(time_step):
    for start, end in RAINY_INTERVALS:
        if start <= time_step <= end:
            return True
    return False


def get_all_partition_instances():
    instances = []
    for name, obj in inspect.getmembers(generated_hex_partitions):
        if inspect.isclass(obj) and issubclass(obj, Partitions) and obj is not Partitions:
            instances.append(obj())
    return instances


def generate_weather_scenario(partitions, neighbors_map, duration):
    scenario_data = {}

    for time_step in range(0, duration):
        if time_step % 5 != 0:
            continue

        print(f"Calculating weather for time step: {time_step}")

        current_time_state = {}
        is_rainy_period = is_time_in_rainy_interval(time_step)

        if is_rainy_period:
            print(f"Time {time_step}: Entering RAINY period logic.")
            for p in partitions:
                neighbors = neighbors_map.get(p, [])
                if not neighbors:
                    p.rainStatus = eval(random.choice(NoiseConfigGeneralAttribute.Rain_options))
                    continue

                neighbor_units = [
                    NoiseConfigGeneralAttribute.Rain_class_to_unit[n.rainStatus.__class__.__name__]
                    for n in neighbors
                ]
                min_neighbor_unit = min(neighbor_units)
                max_neighbor_unit = max(neighbor_units)

                min_allowed_unit = max(0, max_neighbor_unit - 2)
                max_allowed_unit = min(len(NoiseConfigGeneralAttribute.Rain_options) - 1, min_neighbor_unit + 2)

                allowed_options = []
                if min_allowed_unit <= max_allowed_unit:
                    for unit in range(min_allowed_unit, max_allowed_unit + 1):
                        allowed_options.append(NoiseConfigGeneralAttribute.Rain_options[unit])

                if allowed_options:
                    new_rain_status_str = random.choice(allowed_options)
                    p.rainStatus = eval(new_rain_status_str)

                status_class_name = p.rainStatus.__class__.__name__
                status_str_for_eval = f"NoiseConfig.{status_class_name}()"
                current_time_state[p.__class__.__name__] = status_str_for_eval

        else:
            for p in partitions:

                current_time_state[p.__class__.__name__] = NO_RAIN_STATUS_STR

                p.rainStatus = eval("NoiseConfig.Rain0()")

        scenario_data[str(time_step)] = current_time_state

    return scenario_data


def main():
    partitions = get_all_partition_instances()

    print(f"Found {len(partitions)} partitions.")

    neighbors_map = UtilsFunc.find_neighbors(partitions)

    duration = Config.SimulatorConfig.SIMULATION_DURATION
    scenario = generate_weather_scenario(partitions, neighbors_map, duration)

    output_filename = "weather_scenario.json"
    with open(output_filename, "w") as f:
        json.dump(scenario, f, indent=2)

    print(f"\nSuccessfully generated and saved weather scenario to '{output_filename}'")


if __name__ == "__main__":
    main()