from dataclasses import dataclass, field
from typing import Optional, Any, List
from VANET.models.task import Task
from VANET.models.node.fog import FixedFogNode

def edf_scheduler(task_list: List[Task]) -> List[Task]:
    edf_queue = sorted(task_list, key=lambda task: task.deadline)
    return edf_queue


def wfd_scheduler(task_list: List[Task], fog_node: FixedFogNode) -> None:
    """
    Assigns tasks to a FogNode's cores using the WFD algorithm.

    This function modifies the fog_node object in-place, adding tasks
    to its core1, core2, core3, and core4 lists.

    1. Sorts tasks by execution time (decreasing).
    2. Calculates the *current* load of each core (in case they
       already have tasks).
    3. Assigns each new task, one by one, to the core with the
       *minimum* current load.

    Args:
        task_list: A list of new Task objects to be scheduled.
        fog_node: The FogNode object whose cores will be assigned.

    Returns:
        None. The fog_node object is modified directly.
    """

    # 1. Sort tasks by exec_time, largest first (Decreasing)
    sorted_tasks = sorted(task_list, key=lambda t: t.exec_time, reverse=True)

    # 2. Create a list to manage the cores and their loads
    # This calculates the *initial* load, in case the cores
    # already have tasks assigned from a previous run.
    cores_data = [
        {'tasks_list': fog_node.core1, 'load': sum(t.exec_time for t in fog_node.core1)},
        {'tasks_list': fog_node.core2, 'load': sum(t.exec_time for t in fog_node.core2)},
        {'tasks_list': fog_node.core3, 'load': sum(t.exec_time for t in fog_node.core3)},
        {'tasks_list': fog_node.core4, 'load': sum(t.exec_time for t in fog_node.core4)},
    ]

    # 3. Assign new tasks
    for task in sorted_tasks:
        # 4. Find the core with the minimum current load
        least_loaded_core_data = min(cores_data, key=lambda c: c['load'])

        # 5. Assign the task to that core's list
        least_loaded_core_data['tasks_list'].append(task)

        # 6. Update the load in our manager list
        least_loaded_core_data['load'] += task.exec_time

    # No return value needed, as we've modified the fog_node's lists
    # (which were passed by reference).
# --- Example Usage ---

if __name__ == "__main__":
    # 1. Create a list of tasks
    # (creator_id is used as a simple identifier here)
    tasks = [
        Task(release_time=0, deadline=8, exec_time=3, power=10, creator_id="T1", dataSize=100),
        Task(release_time=1, deadline=5, exec_time=2, power=5, creator_id="T2", dataSize=50),
        Task(release_time=2, deadline=10, exec_time=4, power=12, creator_id="T3", dataSize=200),
        Task(release_time=3, deadline=5, exec_time=1, power=3, creator_id="T4", dataSize=30),
    ]

    print("--- Original Task List ---")
    for task in tasks:
        print(task)

    # 2. Generate the EDF queue
    scheduled_queue = edf_scheduler(tasks)

    print("\n--- EDF Scheduled Queue (Earliest Deadline First) ---")
    for task in scheduled_queue:
        print(task)

    # Example of tie-breaking (T2 and T4 have the same deadline)
    # The default sort is stable, so T2 (which appeared first in the
    # original list) will appear before T4.
    # If we had used the (deadline, release_time) key, T2 (release 1)
    # would also come before T4 (release 3).
    print("\nNote: T2 and T4 have the same deadline of 5.")
    print("T2 comes first because it appeared earlier in the original list (stable sort).")
