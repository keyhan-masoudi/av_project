from concurrent.futures import ProcessPoolExecutor, as_completed


from config import Config
from controllers.loader import Loader
from controllers.simulator import Simulator, red_bg
from models.node.cloud import CloudNode
from utils.clock import Clock

from controllers.simulator_maddpg import SimulatorMADDPG
from controllers.Simulator.simulator_ddpg import SimulatorDDPG
from controllers.Simulator.simulator_ppo import SimulatorPPO
from controllers.Simulator.simulator_sac import SimulatorSAC

def run_one(params):
    algorithm, method, threshold, traffic_noise_profile, attenuationLevel, city = params
    Config.ZoneManagerConfig.DEFAULT_ALGORITHM = algorithm
    Config.NoiseMethod.DEFAULT_METHOD = method
    Config.NoiseConfig.DEFAULT_THRESHOLD = threshold
    Config.AttenuationLevel.DEFAULT_AttenuationLevel = attenuationLevel
    Config.TrafficNoise.DEFAULT_TrafficNoiseLevel = traffic_noise_profile
    Config.City.DEFAULT_CITY = city
    print(f"=====================================================")
    print(f"=== Start of : {algorithm} ===")
    print(f"=====================================================")

    if Config.City.DEFAULT_CITY == Config.City.MELBOURNE:
        Config.Directory.DEFAULT_ZON = Config.Directory.ZON_MEL
        Config.Directory.DEFAULT_FN = Config.Directory.FN_MEL
        Config.CloudConfig.CLOSEST_FOG_X = 2069.28
        Config.CloudConfig.CLOSEST_FOG_Y = 789.39
        Config.SimulatorConfig.SIMULATION_DURATION = 1300
        Config.SimulatorConfig.SIMULATION_START_TIME = 300
    else:
        Config.Directory.DEFAULT_ZON = Config.Directory.ZON_HAM
        Config.Directory.DEFAULT_FN = Config.Directory.FN_HAM
        Config.CloudConfig.CLOSEST_FOG_X = 4214.90
        Config.CloudConfig.CLOSEST_FOG_Y = 1932.26
        Config.SimulatorConfig.SIMULATION_DURATION = 1200
        Config.SimulatorConfig.SIMULATION_START_TIME = 0

    if Config.AttenuationLevel.AttenuationLevel1 == attenuationLevel:
        Config.AttenuationLevel.DEFAULT_AttenuationLevelName = Config.AttenuationLevel.AttenuationLevel1Name
    else:
        Config.AttenuationLevel.DEFAULT_AttenuationLevelName = Config.AttenuationLevel.AttenuationLevel2Name

    if traffic_noise_profile == 0:
        # Use "Noise 0" configuration
        Config.TrafficNoise.GreenTrafficNoise.DEFAULT_GreenTrafficNoise = Config.TrafficNoise.GreenTrafficNoise.GreenTrafficNoise1
        Config.TrafficNoise.YellowTrafficNoise.DEFAULT_YellowTrafficNoise = Config.TrafficNoise.YellowTrafficNoise.YellowTrafficNoise1
        Config.TrafficNoise.OrangeTrafficNoise.DEFAULT_OrangeTrafficNoise = Config.TrafficNoise.OrangeTrafficNoise.OrangeTrafficNoise1
        Config.TrafficNoise.RedTrafficNoise.DEFAULT_RedTrafficNoise = Config.TrafficNoise.RedTrafficNoise.RedTrafficNoise1
        Config.TrafficNoise.BlackTrafficNoise.DEFAULT_BlackTrafficNoise = Config.TrafficNoise.BlackTrafficNoise.BlackTrafficNoise1
    else:
        # Use "Noise 1" configuration
        Config.TrafficNoise.GreenTrafficNoise.DEFAULT_GreenTrafficNoise = Config.TrafficNoise.GreenTrafficNoise.GreenTrafficNoise2
        Config.TrafficNoise.YellowTrafficNoise.DEFAULT_YellowTrafficNoise = Config.TrafficNoise.YellowTrafficNoise.YellowTrafficNoise2
        Config.TrafficNoise.OrangeTrafficNoise.DEFAULT_OrangeTrafficNoise = Config.TrafficNoise.OrangeTrafficNoise.OrangeTrafficNoise2
        Config.TrafficNoise.RedTrafficNoise.DEFAULT_RedTrafficNoise = Config.TrafficNoise.RedTrafficNoise.RedTrafficNoise2
        Config.TrafficNoise.BlackTrafficNoise.DEFAULT_BlackTrafficNoise = Config.TrafficNoise.BlackTrafficNoise.BlackTrafficNoise2


    loader = Loader(
        zone_file=Config.Directory.DEFAULT_ZON,
        fixed_fn_file=Config.Directory.DEFAULT_FN,
        mobile_file="./data/vehicles",
        task_file="./data/tasks",
        checkpoint_path="./checkpoints",
    )
    cloud = CloudNode(
        id="CLOUD0",
        x=Config.CloudConfig.DEFAULT_X,
        y=Config.CloudConfig.DEFAULT_Y,
        power=Config.CloudConfig.DEFAULT_COMPUTATION_POWER,
        remaining_power=Config.CloudConfig.DEFAULT_COMPUTATION_POWER,
        radius=Config.CloudConfig.DEFAULT_RADIUS,
    )

    if Config.ZoneManagerConfig.DEFAULT_ALGORITHM == Config.ZoneManagerConfig.ALGORITHM_MADDPG:
        simulator = SimulatorMADDPG(loader, Clock(), cloud)
    elif Config.ZoneManagerConfig.DEFAULT_ALGORITHM == Config.ZoneManagerConfig.ALGORITHM_DDPG:
        simulator = SimulatorDDPG(loader, Clock(), cloud)
    elif Config.ZoneManagerConfig.DEFAULT_ALGORITHM == Config.ZoneManagerConfig.ALGORITHM_PPO:
        simulator = SimulatorPPO(loader, Clock(), cloud)
    elif Config.ZoneManagerConfig.DEFAULT_ALGORITHM == Config.ZoneManagerConfig.ALGORITHM_SAC:
        simulator = SimulatorSAC(loader, Clock(), cloud)
    else:
        simulator = Simulator(loader, Clock(), cloud)

    simulator.start_simulation()
    return {
        "algorithm": algorithm,
        "method": method,
        "total": simulator.metrics.total_tasks,
        "completed": simulator.metrics.completed_tasks,
        "missed": simulator.metrics.deadline_misses,
        "migrations": simulator.metrics.migrations_count,
        "cloud": simulator.metrics.cloud_tasks,
        "metrics": simulator.metrics,
    }

if __name__ == "__main__":
    algorithms = [
        Config.ZoneManagerConfig.ALGORITHM_RANDOM,
        # Config.ZoneManagerConfig.ALGORITHM_HEURISTIC,
        # Config.ZoneManagerConfig.ALGORITHM_ONLY_CLOUD,
        # Config.ZoneManagerConfig.ALGORITHM_ONLY_FOG,
        # Config.ZoneManagerConfig.ALGORITHM_DEEP_RL,
        # Config.ZoneManagerConfig.ALGORITHM_DDPG,
        # Config.ZoneManagerConfig.ALGORITHM_PPO,
        # Config.ZoneManagerConfig.ALGORITHM_SAC,
        # Config.ZoneManagerConfig.ALGORITHM_MADDPG,
    ]

    methods = [
        Config.NoiseMethod.FIRST_CHOICE,
        # Config.NoiseMethod.RANDOM_CHOICE,
        # Config.NoiseMethod.MIN_DISTANCE,
    ]

    thresholds = [
        Config.NoiseConfig.NONE
    ]

    traffic_noise_profiles = [
        # Config.TrafficNoise.TrafficNoiseLevel1,
        Config.TrafficNoise.TrafficNoiseLevel2
    ]

    attenuationLevels = [
        # Config.AttenuationLevel.AttenuationLevel1,
        Config.AttenuationLevel.AttenuationLevel2,
    ]

    cities = [
        Config.City.MELBOURNE,
        # Config.City.HAMBURG,
    ]

    all_results = []
    # for algorithm in algorithms:
    #     print(f"=====================================================")
    #     print(f"=== Start of : {algorithm} ===")
    #     print(f"=====================================================")
    #
    #     tasks_for_current_algo = [(algorithm, m, t, n, at) for m in methods for t in thresholds for n in traffic_noise_profiles for at in attenuationLevels]
    #
    #     with ProcessPoolExecutor() as executor:
    #         futures = {executor.submit(run_one, t): t for t in tasks_for_current_algo}
    #         for fut in as_completed(futures):
    #             res = fut.result()
    #             print(red_bg(f"Finished {res['algorithm']} / {res['method']}"))
    #             print("SCENARIO\tALGORITHM\tMETHOD\tTOTAL\tCOMPLETED\tMISSED\tMIGRATIONS\tCLOUD")
    #             print(
    #                 f"Rainy\t{res['algorithm']}\t{res['method']}\t"
    #                 f"{res['total']}\t{res['completed']}\t{res['missed']}\t"
    #                 f"{res['migrations']}\t{res['cloud']}"
    #             )
    #             all_results.append(res)
    #
    #     print(f"--- End of simulations for : {algorithm} ---")
    #
    #

    tasks_for_current_algo = [(algorithm, m, t, n, at, city) for algorithm in algorithms for m in methods for t in thresholds for n in
                              traffic_noise_profiles for at in attenuationLevels for city in cities]

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_one, t): t for t in tasks_for_current_algo}
        for fut in as_completed(futures):
            res = fut.result()
            print(red_bg(f"Finished {res['algorithm']} / {res['method']}"))
            print("SCENARIO\tALGORITHM\tMETHOD\tTOTAL\tCOMPLETED\tMISSED\tMIGRATIONS\tCLOUD")
            print(
                f"Rainy\t{res['algorithm']}\t{res['method']}\t"
                f"{res['total']}\t{res['completed']}\t{res['missed']}\t"
                f"{res['migrations']}\t{res['cloud']}"
            )
            all_results.append(res)