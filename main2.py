import matplotlib.pyplot as plt
import numpy as np
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


def visualize_metrics(metrics_controller):
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    overview_data = [
        metrics_controller.completed_tasks,
        metrics_controller.migrations_count,
        metrics_controller.deadline_misses,
        metrics_controller.cloud_tasks
    ]
    labels = ['Completed Tasks', 'Migrations', 'Deadline Misses', 'Cloud Tasks']
    colors = ['#00C49F', '#0088FE', '#FF8042', '#FFBB28']
    ax1.pie(overview_data, labels=labels, colors=colors, autopct='%1.1f%%')
    ax1.set_title('System Overview')

    ax2 = fig.add_subplot(gs[0, 1])
    steps = range(1, len(metrics_controller.migration_counts_per_step) + 1)
    ax2.plot(steps, metrics_controller.deadline_misses_per_step,
             label='Deadline Misses', color='#FF8042')
    ax2.plot(steps, metrics_controller.completed_task_per_step,
             label='Completed Tasks', color='#00C49F')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Count')
    ax2.set_title('Metrics Per Step')
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, :])
    categories = ['Total Tasks', 'Completed Tasks', 'Missed Tasks']
    values = [
        metrics_controller.total_tasks,
        metrics_controller.completed_tasks,
        metrics_controller.deadline_misses,
    ]
    colors = ['#8884d8', '#00C49F', '#FFBB28']
    bars = ax3.bar(categories, values, color=colors)
    ax3.set_title('Task Distribution')
    ax3.set_ylabel('Count')

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_load_differences(metrics_controller, num_points=50):
    task_ids = sorted(metrics_controller.task_load_diff.keys())
    min_loads = [metrics_controller.task_load_diff[id][0] for id in task_ids]
    max_loads = [metrics_controller.task_load_diff[id][1] for id in task_ids]

    if not task_ids:
        print("No data to plot")
        return

    if len(task_ids) > num_points:
        indices = np.linspace(0, len(task_ids) - 1, num_points, dtype=int)
        task_ids = [task_ids[i] for i in indices]
        min_loads = [min_loads[i] for i in indices]
        max_loads = [max_loads[i] for i in indices]

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(task_ids)), min_loads, 'b-', label='Min Load', alpha=0.7)
    plt.plot(range(len(task_ids)), max_loads, 'r-', label='Max Load', alpha=0.7)
    plt.fill_between(range(len(task_ids)), min_loads, max_loads, alpha=0.2)

    show_n_ticks = min(10, len(task_ids))
    tick_indices = np.linspace(0, len(task_ids) - 1, show_n_ticks, dtype=int)
    plt.xticks(tick_indices, [task_ids[i] for i in tick_indices], rotation=45)

    plt.xlabel('Task ID')
    plt.ylabel('Load')
    plt.title('Task Load Range')
    plt.legend()
    plt.grid(True)

    avg_diff = np.mean([max_loads[i] - min_loads[i] for i in range(len(task_ids))])
    plt.annotate(f'Average Load Difference: {avg_diff:.3f}',
                 xy=(0.02, 0.98), xycoords='axes fraction',
                 bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


def run_one(params):
    algorithm, method, threshold, traffic_noise_profile, attenuationLevel = params
    Config.ZoneManagerConfig.DEFAULT_ALGORITHM = algorithm
    Config.NoiseMethod.DEFAULT_METHOD = method
    Config.NoiseConfig.DEFAULT_THRESHOLD = threshold
    Config.AttenuationLevel.DEFAULT_AttenuationLevel = attenuationLevel
    Config.TrafficNoise.DEFAULT_TrafficNoiseLevel = traffic_noise_profile

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
        zone_file="./data/hamburg.zon.xml",
        fixed_fn_file="./data/hamburg.fn.xml",
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
        # Config.ZoneManagerConfig.ALGORITHM_RANDOM,
        # Config.ZoneManagerConfig.ALGORITHM_HEURISTIC,
        # Config.ZoneManagerConfig.ALGORITHM_ONLY_CLOUD,
        # Config.ZoneManagerConfig.ALGORITHM_ONLY_FOG,
        Config.ZoneManagerConfig.ALGORITHM_DEEP_RL,
        # Config.ZoneManagerConfig.ALGORITHM_DDPG,
        # Config.ZoneManagerConfig.ALGORITHM_PPO,
        # Config.ZoneManagerConfig.ALGORITHM_SAC,
        # Config.ZoneManagerConfig.ALGORITHM_MADDPG,
    ]

    methods = [
        # Config.NoiseMethod.FIRST_CHOICE,
        Config.NoiseMethod.RANDOM_CHOICE,
        # Config.NoiseMethod.MIN_DISTANCE,
        # Config.NoiseMethod.PROPOSED_METHOD
    ]

    thresholds = [  # just need for Proposed method
        Config.NoiseConfig.NONE,
        # Config.NoiseConfig.T1,
        # Config.NoiseConfig.T2,
        # Config.NoiseConfig.T3
    ]

    traffic_noise_profiles = [
        # Config.TrafficNoise.TrafficNoiseLevel1,
        Config.TrafficNoise.TrafficNoiseLevel2
    ]

    attenuationLevels = [
        # Config.AttenuationLevel.AttenuationLevel1,
        Config.AttenuationLevel.AttenuationLevel2,
    ]

    all_results = []
    for algorithm in algorithms:
        print(f"=====================================================")
        print(f"=== Start of : {algorithm} ===")
        print(f"=====================================================")

        tasks_for_current_algo = [(algorithm, m, t, n, at) for m in methods for t in thresholds for n in traffic_noise_profiles for at in attenuationLevels]

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
        
        print(f"--- End of simulations for : {algorithm} ---")


    if all_results:
        print("\nAll simulations finished. Visualizing metrics from the last run.")
        # visualize_metrics(all_results[-1]["metrics"])
        # plot_load_differences(all_results[-1]["metrics"])