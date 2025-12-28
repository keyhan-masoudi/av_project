import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from collections import deque
from typing import List, Dict, Tuple
import abc


class ModelBaseABC:
    pass


class MobileNodeABC(ModelBaseABC):
    pass


class NodeABC(ModelBaseABC, abc.ABC):
    def __init__(self, node_id, power, x=0, y=0, bandwidth=10, latency=0.5):
        self.id = node_id
        self.x = x
        self.y = y
        self.power = power  # Processing speed
        self.bandwidth = bandwidth  # Network speed
        self.latency = latency  # Network latency

        # Simulating 2 cores per node for this example
        self.cores = [[], []]
        self.tasks = deque()
        self.finished_tasks = deque()

    def __repr__(self):
        return f"Node({self.id})"


class Task(ModelBaseABC):
    def __init__(self, t_id, release, exec_time, deadline, data_size, creator_id="Car"):
        self.id = t_id
        self.release_time = release
        self.deadline = deadline
        self.exec_time = exec_time
        self.dataSize = data_size
        self.creator_id = creator_id

        # Runtime attributes
        self.start_time = 0
        self.finish_time = 0
        self.executor = None
        self.remaining_time = exec_time

    def __repr__(self):
        return f"Task({self.id} | Rel:{self.release_time} | D:{self.deadline})"


# ==========================================
# 2. SCHEDULER IMPLEMENTATION
# ==========================================

@dataclass
class HardTaskBlueprint:
    """Defines the pattern for a Hard Periodic Task."""
    type_id: str
    period: float
    wcet: float  # Worst Case Execution Time (Load)
    deadline: float


class HybridScheduler:
    def __init__(self):
        # Stores offline decision: hard_type_id -> (Node, CoreIndex)
        self.offline_map: Dict[str, Tuple[NodeABC, int]] = {}
        # Stores the blueprints to check schedulability
        self.node_allocations: Dict[Tuple[str, int], List[HardTaskBlueprint]] = {}

    def get_execution_time(self, task_load, node_power):
        return task_load / node_power

    # --- PHASE 1: OFFLINE (Hard Tasks) ---
    def perform_offline_scheduling(self, blueprints: List[HardTaskBlueprint], vehicle_nodes: List[NodeABC]):
        """
        Assigns hard task Types to Vehicle Nodes permanently.
        """
        print(f"\n--- [Phase 1] Offline Scheduling for {len(blueprints)} Hard Blueprints ---")

        # Sort by Rate Monotonic (Period)
        blueprints.sort(key=lambda x: x.period)

        for bp in blueprints:
            assigned = False
            for node in vehicle_nodes:
                for core_idx in range(len(node.cores)):
                    if self._check_schedulability(bp, node, core_idx):
                        # Successful Assignment
                        self.offline_map[bp.type_id] = (node, core_idx)

                        # Record allocation for future checks
                        key = (node.id, core_idx)
                        if key not in self.node_allocations: self.node_allocations[key] = []
                        self.node_allocations[key].append(bp)

                        print(f"  [Offline] Assigned '{bp.type_id}' to {node.id} Core {core_idx}")
                        assigned = True
                        break
                if assigned: break

            if not assigned:
                print(f"  [CRITICAL] Could not assign Hard Task '{bp.type_id}' offline!")

    def _check_schedulability(self, new_bp, node, core_idx):
        """Demand-Supply Analysis / Response Time Analysis"""
        existing_bps = self.node_allocations.get((node.id, core_idx), [])

        # Calculate C (Time) based on Node Power
        C_new = self.get_execution_time(new_bp.wcet, node.power)

        # Check if new task fits
        w = C_new
        while True:
            interference = 0
            for existing in existing_bps:
                # Higher Priority (Shorter Period) tasks interfere
                if existing.period <= new_bp.period:
                    C_exist = self.get_execution_time(existing.wcet, node.power)
                    interference += math.ceil(w / existing.period) * C_exist

            w_next = C_new + interference
            if w_next > new_bp.deadline: return False
            if w_next == w: return True
            w = w_next

    # --- PHASE 2: ONLINE (Soft Tasks) ---
    def schedule_online_job(self, task: Task, vehicle_nodes, edge_nodes, cloud_nodes):
        """
        Algorithm 4: Offloads soft tasks if local is busy.
        """
        current_time = task.release_time
        abs_deadline = task.release_time + task.deadline

        # 1. Try Local Vehicle (Checking slack after hard tasks)
        for node in vehicle_nodes:
            for core_idx in range(len(node.cores)):
                if self._check_online_slack(task, node, core_idx):
                    task.executor = node
                    print(f"  [Online] Soft Task '{task.id}' accepted on LOCAL {node.id} (Core {core_idx})")
                    return

        # 2. Try Offloading (Compare Edge vs Cloud)
        best_node = None
        min_finish = float('inf')

        # Check Edge & Cloud
        candidates = edge_nodes + cloud_nodes
        for node in candidates:
            # Finish = Now + Tx_Time + Exec_Time + Latency
            tx_time = task.dataSize / node.bandwidth
            exec_time = self.get_execution_time(task.exec_time, node.power)
            finish_time = current_time + tx_time + exec_time + node.latency

            if finish_time <= abs_deadline:
                if finish_time < min_finish:
                    min_finish = finish_time
                    best_node = node

        if best_node:
            task.executor = best_node
            print(f"  [Online] Soft Task '{task.id}' OFFLOADED to {best_node.id} (Est Finish: {min_finish:.2f})")
        else:
            print(f"  [Online] Soft Task '{task.id}' DISCARDED (Deadline Miss)")

    def _check_online_slack(self, soft_task, node, core_idx):
        """Checks if soft task fits in gaps of hard tasks"""
        key = (node.id, core_idx)
        hard_bps = self.node_allocations.get(key, [])

        C_soft = self.get_execution_time(soft_task.exec_time, node.power)
        w = C_soft

        # Response Time Analysis assuming Soft Task has lowest priority
        while True:
            interference = 0
            for bp in hard_bps:
                C_hard = self.get_execution_time(bp.wcet, node.power)
                interference += math.ceil(w / bp.period) * C_hard

            w_next = C_soft + interference
            if w_next > soft_task.deadline: return False
            if w_next == w: return True
            w = w_next


# ==========================================
# 3. XML PARSER (MOCKED)
# ==========================================

def load_soft_tasks_from_xml(xml_string):
    root = ET.fromstring(xml_string)
    tasks = []
    for t in root.findall('task'):
        new_task = Task(
            t_id=t.get('id'),
            release=float(t.get('release')),
            exec_time=float(t.get('exec')),
            deadline=float(t.get('deadline')),
            data_size=float(t.get('data'))
        )
        tasks.append(new_task)
    return tasks


# ==========================================
# 4. MAIN EXECUTION FLOW
# ==========================================

if __name__ == "__main__":
    # --- A. Setup Infrastructure ---
    # Vehicle: Medium Power, 2 Cores
    car = NodeABC("Vehicle_1", power=10)
    # Edge: High Power, High Bandwidth
    edge = NodeABC("Edge_Server", power=30, bandwidth=100, latency=2)
    # Cloud: Massive Power, Low Bandwidth, High Latency
    cloud = NodeABC("Cloud_AWS", power=100, bandwidth=20, latency=20)

    scheduler = HybridScheduler()

    # --- B. Define 3 Hard Periodic Task Types ---
    # 1. Steering: Very frequent, low compute
    # 2. SensorFusion: Medium frequency, medium compute
    # 3. Navigation: Low frequency, high compute
    hard_blueprints = [
        HardTaskBlueprint("HARD_Steering", period=20, wcet=50, deadline=20),
        HardTaskBlueprint("HARD_Sensor", period=50, wcet=100, deadline=50),
        HardTaskBlueprint("HARD_Nav", period=200, wcet=400, deadline=200)
    ]

    # --- C. Run Offline Scheduling ---
    # This locks hard task types to specific vehicle cores
    scheduler.perform_offline_scheduling(hard_blueprints, [car])

    # --- D. Load Soft Tasks from XML ---
    # Example XML content
    # xml_data = """
    # <tasks>
    #     <task id="SOFT_Music" release="15" exec="60" deadline="100" data="500" />
    #     <task id="SOFT_Video" release="16" exec="500" deadline="50" data="2000" />
    #     <task id="SOFT_Update" release="105" exec="200" deadline="300" data="100" />
    # </tasks>
    # """
    soft_tasks_queue = deque(load_soft_tasks_from_xml(xml_data))

    # --- E. Simulation Loop (Time 0 to 250) ---
    print("\n--- [Phase 2] Starting Simulation (t=0 to 250) ---")

    simulation_duration = 250

    # This loop generates hard jobs and processes soft jobs as time moves
    for current_time in range(simulation_duration):

        # 1. Generate Hard Task Instances for this timestep
        for bp in hard_blueprints:
            # If current time is a multiple of period, release a new job
            if current_time % bp.period == 0:
                # Create the actual runtime Task object
                hard_job = Task(
                    t_id=f"{bp.type_id}_{current_time}",
                    release=current_time,
                    exec_time=bp.wcet,
                    deadline=bp.deadline,
                    data_size=0
                )

                # Retrieve Offline Assignment
                if bp.type_id in scheduler.offline_map:
                    assigned_node, core_idx = scheduler.offline_map[bp.type_id]
                    hard_job.executor = assigned_node
                    # In a real sim, you would add to assigned_node.cores[core_idx].append(hard_job)
                    # print(f"  [t={current_time}] Generated {hard_job.id} -> {assigned_node.id} Core {core_idx}")

        # 2. Check for Soft Task Arrivals from XML Queue
        while soft_tasks_queue and soft_tasks_queue[0].release_time == current_time:
            soft_job = soft_tasks_queue.popleft()

            # Run the Online Algorithm (Algorithm 4)
            scheduler.schedule_online_job(
                soft_job,
                vehicle_nodes=[car],
                edge_nodes=[edge],
                cloud_nodes=[cloud]
            )