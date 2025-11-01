import os
from collections import defaultdict
from typing import List, Dict

import pandas as pd


def green_bg(text):
    return f"\033[42m{text}\033[0m"


class MetricsController:
    """Gathers and store all statistics metrics in our system."""

    def __init__(self):
        # General Metrics
        self.no_device_found_to_run_becauseOf_Noise = 0
        self.migrations_count = 0  # Total number of migrations happened in system.
        self.deadline_misses = 0  # Total number of deadline misses happened in system.
        self.no_resource_found = 0  # Total number of tasks that did not find resource to execute in system.
        self.migrate_and_miss = 0
        self.local_execution = 0
        self.fog_execution = 0
        self.total_tasks = 0  # Total number of tasks processed in system.
        self.cloud_tasks = 0  # Total number of tasks offloaded to cloud server.
        self.completed_tasks = 0  # Total number of tasks completed in system.
        self.packet_loss = 0

        # Per Node Metrics
        self.node_task_counts: dict[str, int] = defaultdict(int)

        # Per Step Metrics
        self.migration_counts_per_step: list[int] = []
        self.deadline_misses_per_step: list[int] = []
        self.completed_task_per_step: list[int] = []
        # self.transmission_delay_per_step: list[float] = []

        self.current_step_migrations = 0
        self.current_step_deadline_misses = 0
        self.current_step_completed_tasks = 0
        self.current_step_transmission = 0.0
        self.count_step_transmission = 0
        self.dataPerStep: List[Dict] = []

        # Per task metrics
        self.task_load_diff: dict[int, tuple[float, float]] = {}

        self.cloudDataRate: list[float] = []
        self.fogDataRate: list[float] = []

    def addFogDataRate(self, dataRate: float):
        self.fogDataRate.append(dataRate)

    def addCloudDataRate(self, dataRate: float):
        self.cloudDataRate.append(dataRate)

    def saveToExcel(self, filename: str):
        df = pd.DataFrame(self.fogDataRate, columns=["Fog Data Rate"])
        df.to_excel(filename, index=False)

    def inc_task_load_diff(self, task_id: int, min_load: float, max_load: float):
        self.task_load_diff[task_id] = (min_load, max_load)

    def inc_local_execution(self):
        self.local_execution += 1

    def inc_fog_execution(self):
        self.fog_execution += 1

    def inc_migrate_and_miss(self):
        self.migrate_and_miss += 1

    def inc_completed_task(self):
        self.current_step_completed_tasks += 1
        self.completed_tasks += 1

    def add_transmission(self, transmission):
        self.current_step_transmission += transmission
        self.count_step_transmission += 1

    def inc_migration(self):
        self.current_step_migrations += 1
        self.migrations_count += 1

    def inc_no_resource_found(self):
        self.current_step_deadline_misses += 1
        self.deadline_misses += 1
        self.no_resource_found += 1

    def inc_deadline_miss(self):
        self.current_step_deadline_misses += 1
        self.deadline_misses += 1

    def inc_total_tasks(self):
        self.total_tasks += 1

    def inc_node_tasks(self, node_id: str):
        self.node_task_counts[node_id] += 1

    def print_node_tasks(self):
        for node_id, count in self.node_task_counts.items():
            print(green_bg(f"Node ID: {node_id}, Count: {count}"))

    def inc_cloud_tasks(self):
        self.cloud_tasks += 1

    def inc_packet_loss(self):
        self.packet_loss += 1

    def inc_no_device_found_to_run_becauseOf_Noise(self):
        self.no_device_found_to_run_becauseOf_Noise += 1

    def flush(self):
        self.migration_counts_per_step.append(self.current_step_migrations)
        self.deadline_misses_per_step.append(self.current_step_deadline_misses)
        self.completed_task_per_step.append(self.current_step_completed_tasks)
        # if self.count_step_transmission != 0:
        #     self.transmission_delay_per_step.append(self.current_step_transmission / self.count_step_transmission)
        # else:
        #     self.transmission_delay_per_step.append(0)
        self.current_step_deadline_misses = 0
        self.current_step_migrations = 0
        self.current_step_completed_tasks = 0
        self.current_step_transmission = 0
        self.count_step_transmission = 0

    def log_metrics(self):
        print("Metrics:")
        print(f"\tTotal packet loss: {self.packet_loss}")
        print(f"\tTotal no_device_found_to_run_becauseOf_Noise: {self.no_device_found_to_run_becauseOf_Noise}")
        # print(f"\tTotal migrations: {self.migrations_count}")
        print(f"\tTotal deadline misses: {self.deadline_misses}")
        # print(f"\tTotal migrate and misses: {self.migrate_and_miss}")
        print(f"\tTotal cloud tasks: {self.cloud_tasks}")
        print(f"\tTotal local execution tasks: {self.local_execution}")
        print(f"\tTotal fog execution tasks: {self.fog_execution}")
        print(f"\tTotal completed tasks: {self.completed_tasks}")
        print(f"\tTotal tasks: {self.total_tasks}")
        # if self.count_step_transmission != 0:
        #     print(f"\tTransmission delay per step: {self.transmission_delay_per_step}")
        if self.total_tasks != 0:
            print(f"\tPacket loss ratio: {'{:.3f}'.format(self.packet_loss * 100 / self.total_tasks)}%")
            # print(f"\tMigration ratio: {'{:.3f}'.format(self.migrations_count * 100 / self.total_tasks)}%")
            print(f"\tDeadline miss ratio: {'{:.3f}'.format(self.deadline_misses * 100 / self.total_tasks)}%")
            if self.deadline_misses:
                print(
                    f"\tNo Resource found by deadline miss ratio: "
                    f"{'{:.3f}'.format(self.no_resource_found * 100 / self.deadline_misses)}%"
                )

    def add_data(self, current_time):
        deadline_miss_ratio = 0.0
        no_resource_ratio = 0.0
        packet_loss_ratio = 0.0

        if self.total_tasks != 0:
            packet_loss_ratio = self.packet_loss * 100 / self.total_tasks
            deadline_miss_ratio = self.deadline_misses * 100 / self.total_tasks

            if self.deadline_misses != 0:
                no_resource_ratio = self.no_resource_found * 100 / self.deadline_misses

        metrics_data = {
            'timeStep': current_time,
            'Total deadline misses': self.deadline_misses,
            'Total cloud tasks': self.cloud_tasks,
            'Total local execution tasks': self.local_execution,
            'Total fog execution tasks': self.fog_execution,
            'Total completed tasks': self.completed_tasks,
            'Deadline miss ratio': f"{deadline_miss_ratio:.3f}%",
            'No Resource found by deadline miss ratio': f"{no_resource_ratio:.3f}%",
            'Total packet loss': self.packet_loss,
            'Total no_device_found_to_run_becauseOf_Noise': self.no_device_found_to_run_becauseOf_Noise,
            'Packet loss ratio': f"{packet_loss_ratio:.3f}%",
            # 'Transmission delay per step': self.transmission_delay_per_step
        }
        self.dataPerStep.append(metrics_data)

    def save_to_excel(self, filename: str = "final_metrics.xlsx"):
        df = pd.DataFrame(self.dataPerStep)
        output_dir = "Results_Metrics"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory '{output_dir}' created.")

        full_path = os.path.join(output_dir, filename)

        try:
            df.to_excel(full_path, index=False)
            print(f"Successfully saved metrics to {full_path}")
        except Exception as e:
            print(f"Error saving metrics to Excel file: {e}")