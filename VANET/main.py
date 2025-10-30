import matplotlib.pyplot as plt
import numpy as np

from config import Config
from controllers.loader import Loader
from controllers.simulator import Simulator, red_bg
from controllers.zone_managers.hrl.hrl import HRLZoneManager
from models.node.cloud import CloudNode
from utils.clock import Clock

from controllers.simulator_maddpg import SimulatorMADDPG
from controllers.Simulator.simulator_ddpg import SimulatorDDPG
from controllers.Simulator.simulator_ppo import SimulatorPPO
from controllers.Simulator.simulator_sac import SimulatorSAC


def visualize_metrics(metrics_controller):
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)

    # 1. System Overview Pie Chart
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

    # 2. Metrics Per Step Line Plot
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

    # 3. Task Distribution Bar Chart
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

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def plot_load_differences(metrics_controller, num_points=50):
    task_ids = sorted(metrics_controller.task_load_diff.keys())
    min_loads = [metrics_controller.task_load_diff[id][0] for id in task_ids]
    max_loads = [metrics_controller.task_load_diff[id][1] for id in task_ids]

    if len(task_ids) == 0:
        print("No data to plot")
        return

    # If too many points, sample them
    if len(task_ids) > num_points:
        indices = np.linspace(0, len(task_ids) - 1, num_points, dtype=int)
        task_ids = [task_ids[i] for i in indices]
        min_loads = [min_loads[i] for i in indices]
        max_loads = [max_loads[i] for i in indices]

    plt.figure(figsize=(12, 6))

    # Plot loads
    plt.plot(range(len(task_ids)), min_loads, 'b-', label='Min Load', alpha=0.7)
    plt.plot(range(len(task_ids)), max_loads, 'r-', label='Max Load', alpha=0.7)
    plt.fill_between(range(len(task_ids)), min_loads, max_loads, alpha=0.2)

    # Set x-ticks to show some task IDs
    show_n_ticks = min(10, len(task_ids))
    tick_indices = np.linspace(0, len(task_ids) - 1, show_n_ticks, dtype=int)
    plt.xticks(tick_indices, [task_ids[i] for i in tick_indices], rotation=45)

    plt.xlabel('Task ID')
    plt.ylabel('Load')
    plt.title('Task Load Range')
    plt.legend()
    plt.grid(True)

    # Add average load difference annotation
    avg_diff = np.mean([max_loads[i] - min_loads[i] for i in range(len(task_ids))])
    plt.annotate(f'Average Load Difference: {avg_diff:.3f}',
                 xy=(0.02, 0.98), xycoords='axes fraction',
                 bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


algorithms_to_run = [
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

methods_to_run = [
    # Config.NoiseMethod.FIRST_CHOICE,
    # Config.NoiseMethod.RANDOM_CHOICE,
    Config.NoiseMethod.MIN_DISTANCE,
    # Config.NoiseMethod.PROPOSED_METHOD
]

for algorithm in algorithms_to_run:
    for method in methods_to_run:
        Config.ZoneManagerConfig.DEFAULT_ALGORITHM = algorithm
        Config.NoiseMethod.DEFAULT_METHOD = method
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
        print(red_bg(f"DEFAULT_ALGORITHM: {Config.ZoneManagerConfig.DEFAULT_ALGORITHM}"))

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

        for _, zone_manager in simulator.zone_managers.items():
            if isinstance(zone_manager, HRLZoneManager):
                zone_manager.save_checkpoint("./checkpoints")

        SCENARIO = "Rainy"

        print("SCENARIO\tALGORITHM\tMETHOD\tTOTAL_TASKS\tCOMPLETED\tMISSED\tMIGRATIONS\tCLOUD_TASKS")
        print("-" * 80)
        print(
            f"{SCENARIO}\t{algorithm}\t{method}\t"
            f"{simulator.metrics.total_tasks}\t{simulator.metrics.completed_tasks}\t"
            f"{simulator.metrics.deadline_misses}\t{simulator.metrics.migrations_count}\t{simulator.metrics.cloud_tasks}"
        )

# visualize_metrics(simulator.metrics)
# plot_load_differences(simulator.metrics)
