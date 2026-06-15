import xml.etree.ElementTree as ET
import os
from typing import Dict, List, Unpack, Any, Optional
from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from models.node.fog import FogLayerABC
from models.node.base import MobileNodeABC
from models.task import Task

class GreedyZoneManager(ZoneManagerABC):
    def __init__(self, zone):
        super().__init__(zone)
        self.simulator = None
        self.__target_node: Optional[Any] = None
        self.task_configs: Dict[str, Dict[str, float]] = {}

    def set_simulator(self, simulator):
        self.simulator = simulator

    # todo add this function to create dictionary
    # config file reader to get D and C for WCET
    def load_task(self, xml_path: str = "task_config.xml"):
        if not os.path.exists(xml_path):
            print(f"Warning: Configuration file {xml_path} not found!")
            return

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for task_elem in root.findall('.//task'): 
            task_id = task_elem.get('id') 
            if task_id:
                data_size_str = task_elem.get('dataSize')
                cycles_str = task_elem.get('cycles_per_bit')
                
                if data_size_str is not None and cycles_str is not None:
                    self.task_configs[task_id] = {'D': float(data_size_str), 'C': float(cycles_str)}
                    
        print(f"Successfully loaded {len(self.task_configs)} task configurations from XML.")
    
    # ==========================================================
    # HELPER MATH METHODS (Replaces modifying Task/NodeABC)
    # ==========================================================
    def _calculate_wcet(self, task: Task, node: MobileNodeABC) -> float:
        """
        Calculates Worst-Case Execution Time based on the XML task config.
        WCET = (Data Size * Cycles per bit) / f
        Note: Assumes your node object has an attribute for CPU frequency (e.g., node.f).
        """
        config = self.task_configs.get(str(task.id))
        
        if config:
            d_val = config['D']
            c_val = config['C']
        else:
            # Fallback if the task isn't in the XML
            d_val = task.dataSize
            c_val = task.cycles_per_bit
            print(f"Warning: Task {task.id} not found in XML. Using default values.")

        # Get the node's CPU frequency (defaulting to 1e9 if 'f' isn't found to avoid division by zero) 
        frequency = node.frequency
        return (d_val * c_val) / frequency
        
    def _calculate_real_exec_time(self, task: Task, node: MobileNodeABC) -> float:
        """
        Calculates the real online execution time dynamically based on the paper's formula.
        Formula used: (Data Size * Cycles per bit) / CPU Frequency
        """
        # Get the node's CPU frequency. Using getattr safely handles if 'freq' is missing.
        # Fallback to 'f' or 1e9 just to prevent division by zero errors.
        frequency = node.frequency
        return (task.dataSize * task.cycles_per_bit) / frequency
    
    def _get_period(self, task: Task) -> float:
        """Infers the task period dynamically."""
        return max(task.deadline - task.release_time, 0.001)

    def _get_utilization(self, task: Task, node: MobileNodeABC) -> float:
        """Calculates CPU utilization for a specific task on a specific node."""
        return self._calculate_wcet(task, node) / self._get_period(task)

    # ==========================================================
    # NODE ACCOMMODATION METHODS (Replaces modifying NodeABC)
    # ==========================================================
    def _can_accommodate_utilization(self, node: MobileNodeABC, task: Task) -> bool:
        utilization = self._get_utilization(task, node)
        for i in range(node.num_cores):
            if node.core_Up[i] + utilization <= 0.693:
                return True
        return False

    def _assign_task_offline(self, node: MobileNodeABC, task: Task):
        utilization = self._get_utilization(task, node)
        best_core = 0
        lowest_util = float('inf')

        for i in range(node.num_cores):
            if node.core_Up[i] < lowest_util:
                lowest_util = node.core_Up[i]
                best_core = i

        node.core_Up[best_core] += utilization
        node.periodic_allocation[best_core].append(task)
        if task.is_hard:
            node.local_hard_tasks.append(task)


    # ==========================================================
    # OFFLINE STAGE (Run this once before the simulation starts)
    # ==========================================================
    def offline_planning(self, all_hard_tasks: List[Task], all_soft_tasks: List[Task]) -> List[Task]:
        # --- Algorithm 1 & 2: Hard Real-Time Tasks (Creator ECU ONLY) ---
        unassigned_hard = self._algorithm_1_utilization_test(all_hard_tasks)
        
        # Check System Load: Did Algorithm 1 perfectly assign all hard tasks?
        alg1_perfect_success = (len(unassigned_hard) == 0)

        if unassigned_hard:
            success = self._algorithm_2_demand_supply(unassigned_hard)
            if not success:
                print("CRITICAL: System safety cannot be guaranteed. Hard tasks failed offline local mapping.")

        # --- Algorithm 3: Soft Real-Time Tasks (Creator ECU First) ---
        final_unassigned_soft = []
        
        if alg1_perfect_success:
            # Fast Test: Alg 1 was completely successful, so try it on soft tasks first
            unassigned_soft_fast = self._algorithm_1_utilization_test(all_soft_tasks)
            
            # Deep Analysis: Process the leftovers
            if unassigned_soft_fast:
                final_unassigned_soft = self._algorithm_3_demand_supply_soft(unassigned_soft_fast)
        else:
            # Deep Analysis: Alg 1 struggled with hard tasks, skip fast test and route all soft tasks here
            final_unassigned_soft = self._algorithm_3_demand_supply_soft(all_soft_tasks)
            
        # Pool of Unassigned Tasks: Returned to be handled on-the-fly during runtime (Algorithm 4)
        return final_unassigned_soft
    
    
    def _algorithm_1_utilization_test(self, tasks: List[Task]) -> List[Task]:
        """
        Algorithm 1: Binary Search + Linear Integer Program (LIP) equivalent.
        Finds the maximum index 'q' where tasks 0 through q can be scheduled 
        without exceeding the 0.693 utilization limit on any core.
        """
        unassigned_all = []

        # Step 1: Group tasks by their creator (vehicle) so each vehicle solves its own cores
        tasks_by_creator = {}
        for task in tasks:
            if task.creator not in tasks_by_creator:
                tasks_by_creator[task.creator] = []
            tasks_by_creator[task.creator].append(task)

        for local_vehicle, vehicle_tasks in tasks_by_creator.items():
            # Step 2: Sort tasks by required frequency (1 / period) in
            sorted_tasks = sorted(vehicle_tasks, key=lambda t: 1.0 / self._get_period(t), reverse=True)

            # Step 3: Binary Search for cutoff index 'q'
            low = 0
            high = len(sorted_tasks)
            best_q = 0
            best_assignment = None

            while low <= high:
                mid = (low + high) // 2
                if mid == 0:
                    low = mid + 1
                    continue

                # Step 4: Linear Integer Program (LIP) feasibility test for tasks 0 to mid-1
                success, assignment = self._solve_lip_feasibility(sorted_tasks[:mid], local_vehicle)

                if success:
                    best_q = mid               # This index is valid
                    best_assignment = assignment # Save the mapping
                    low = mid + 1              # Try to pack more tasks
                else:
                    high = mid - 1             # Failed, try fewer tasks

            # Step 5: Permanently assign the successful tasks (0 through best_q - 1)
            if best_assignment:
                for task_idx, core_idx in enumerate(best_assignment):
                    task_to_assign = sorted_tasks[task_idx]
                    self._commit_task_to_core(local_vehicle, task_to_assign, core_idx)

            # Step 6: Collect the remaining unassigned tasks (best_q to end) for Algorithm 2
            unassigned_all.extend(sorted_tasks[best_q:])

        return unassigned_all
    
    # ==========================================================
    # ALGORITHM 2: Demand-Supply Analysis (Hard Real-Time)
    # ==========================================================
    def _algorithm_2_demand_supply(self, tasks: List[Task]) -> bool:
        """
        Algorithm 2: Tries to assign remaining hard tasks using exact SBF/DBF math.
        """
        for task in tasks:
            local_vehicle = task.creator
            assigned = False

            # 1. Sort Processors: Get all cores and sort them by current utilization (ascending)
            # This ensures the emptiest ECUs (cores) are checked first.
            core_utils = [(i, local_vehicle.core_Up[i]) for i in range(local_vehicle.num_cores)]
            sorted_cores = sorted(core_utils, key=lambda x: x[1])

            # Calculate the task's strict requirements
            dbf = self._calculate_wcet(task, local_vehicle)  # Demand-Bound Function
            period = self._get_period(task)
            task_utilization = self._get_utilization(task, local_vehicle)

            # 2. Calculate Supply vs. Demand for each processor (core)
            for core_idx, core_util in sorted_cores:
                
                # Supply-Bound Function: Time available on this core over the task's period
                sbf_utilization_available = 1.0 - core_util
                sbf_time_available = sbf_utilization_available * period

                # Compare Supply vs Demand
                if sbf_time_available >= dbf:
                    # 3. Assignment: The supply meets the demand, so assign safely.
                    local_vehicle.core_Up[core_idx] += task_utilization
                    local_vehicle.periodic_allocation[core_idx].append(task)
                    
                    if task.is_hard:
                        local_vehicle.local_hard_tasks.append(task)

                    assigned = True
                    break # Stop checking cores for this task

            # 4. Safety Check: If it couldn't fit on ANY sorted processor, we must fail.
            if not assigned:
                print(f"Algorithm 2 Failure: Hard task {task.id} cannot fit on its creator vehicle {local_vehicle.id}.")
                return False

        # If we successfully loop through all tasks without triggering the fail state:
        return True
    

    def _algorithm_3_demand_supply_soft(self, tasks: List[Task]) -> List[Task]:
        """
        Algorithm 3 Deep Analysis: Demand-Supply check for soft tasks.
        Uses the same strict math as Algorithm 2, but gracefully handles unassigned tasks.
        """
        unassigned = []
        
        for task in tasks:
            local_vehicle = task.creator
            assigned = False

            # Sort Processors: Get all cores and sort them by current utilization (ascending)
            core_utils = [(i, local_vehicle.core_Up[i]) for i in range(local_vehicle.num_cores)]
            sorted_cores = sorted(core_utils, key=lambda x: x[1])

            # Calculate Demand
            dbf = self._calculate_wcet(task, local_vehicle)
            period = self._get_period(task)
            task_utilization = self._get_utilization(task, local_vehicle)

            # Calculate Supply vs. Demand for each processor (core)
            for core_idx, core_util in sorted_cores:
                sbf_utilization_available = 1.0 - core_util
                sbf_time_available = sbf_utilization_available * period

                if sbf_time_available >= dbf:
                    # Assignment: Supply meets demand, assign safely
                    local_vehicle.core_Up[core_idx] += task_utilization
                    local_vehicle.periodic_allocation[core_idx].append(task)
                    
                    assigned = True
                    break # Stop checking cores for this task

            # Pool of Unassigned Tasks
            if not assigned:
                unassigned.append(task)

        return unassigned

    # ==========================================================
    # ONLINE STAGE (Algorithm 4 - Dynamic Assignment)
    # ==========================================================
    def can_offload_task(self, task: Task) -> bool:
        """
        Algorithm 4: Handles unassigned soft tasks dynamically at runtime.
        """
        # Hard tasks must be guaranteed offline.
        if task.is_hard:
            return False

        local_vehicle = task.creator
        best_node = None
        best_finish_time = float('inf')

        # 1. Try local slack reclaiming on the CREATOR'S ECU first
        if self._test_local_ecus_with_dynamic_bounds(local_vehicle, task):
            best_node = local_vehicle
            best_finish_time = task.release_time + self._calculate_real_exec_time(task, local_vehicle)
        else:
            # 2 & 3. Local ECU is full. Offload to Edge/Cloud.
            all_fog_nodes: Dict[str, FogLayerABC] = {**self.fixed_fog_nodes, **self.mobile_fog_nodes}
            
            # Check Edge/Roadside Servers
            for node in all_fog_nodes.values():
                if node.can_offload_task(task):
                    real_exec = self._calculate_real_exec_time(task, node)
                    est_finish = task.release_time + real_exec
                    
                    if est_finish <= task.deadline and est_finish < best_finish_time:
                        best_finish_time = est_finish
                        best_node = node

            # Check Remote Cloud Server
            if self.simulator and self.simulator.cloud_node:
                cloud = self.simulator.cloud_node
                if cloud.can_offload_task(task):
                    real_exec = self._calculate_real_exec_time(task, cloud)
                    est_finish = task.release_time + real_exec
                    
                    if est_finish <= task.deadline and est_finish < best_finish_time:
                        best_finish_time = est_finish
                        best_node = cloud

        # 4. Final Assignment & Saving State
        if best_node:
            self.__target_node = best_node
            
            # Record the actual calculated execution time into your Task object
            task.real_exec_time_base = self._calculate_real_exec_time(task, best_node)
            return True

        # 5. Discard: Task cannot fit anywhere before its deadline
        return False

    def _test_local_ecus_with_dynamic_bounds(self, node: MobileNodeABC, new_task: Task) -> bool:
        """
        Recalculates Demand-Bound and Supply-Bound functions using exact online information.
        """
        # Calculate dynamic demand using the updated formula
        dynamic_demand = self._calculate_real_exec_time(new_task, node)
        
        # Exact active time window
        available_window = new_task.deadline - new_task.release_time
        if available_window <= 0:
            return False

        # Iterate through the vehicle's ECUs (cores)
        for i in range(node.num_cores):
            # Recalculate dynamic Supply-Bound Function (SBF)
            dynamic_supply = (1.0 - node.core_Up[i]) * available_window

            # Check if dynamic slack covers the real execution demand
            if dynamic_demand <= dynamic_supply:
                
                # Check Total Bandwidth Server (TBS) capability
                if hasattr(node, 'last_tbs_deadline') and hasattr(node, 'core_Us'):
                    new_deadline = max(new_task.release_time, node.last_tbs_deadline[i]) + (dynamic_demand / node.core_Us[i])
                    if new_deadline <= new_task.deadline:
                        node.last_tbs_deadline[i] = new_deadline
                        return True
                else:
                    return True

        return False
    
    def _solve_lip_feasibility(self, tasks_to_pack: List[Task], node: MobileNodeABC) -> tuple[bool, Optional[List[int]]]:
        """
        Solves the Linear Integer Program (LIP) equivalent using exact backtracking.
        """
        num_cores = node.num_cores
        core_utilizations = list(node.core_Up) 
        task_utils = [self._get_utilization(t, node) for t in tasks_to_pack]
        
        sorted_indices = sorted(range(len(task_utils)), key=lambda i: task_utils[i], reverse=True)
        optimized_task_utils = [task_utils[i] for i in sorted_indices]
        optimized_assignment = [-1] * len(tasks_to_pack)

        def exact_backtrack(opt_idx: int) -> bool:
            if opt_idx == len(optimized_task_utils):
                return True
                
            for core_idx in range(num_cores):
                if core_utilizations[core_idx] + optimized_task_utils[opt_idx] <= 0.693:
                    core_utilizations[core_idx] += optimized_task_utils[opt_idx]
                    optimized_assignment[opt_idx] = core_idx
                    
                    if exact_backtrack(opt_idx + 1):
                        return True
                        
                    core_utilizations[core_idx] -= optimized_task_utils[opt_idx]
            return False

        if exact_backtrack(0):
            final_assignment = [-1] * len(tasks_to_pack)
            for opt_idx, original_idx in enumerate(sorted_indices):
                final_assignment[original_idx] = optimized_assignment[opt_idx]
            return True, final_assignment
            
        return False, None

    def _commit_task_to_core(self, node: MobileNodeABC, task: Task, core_idx: int):
        """Finalizes the assignment variables inside the node."""
        utilization = self._get_utilization(task, node)
        node.core_Up[core_idx] += utilization
        node.periodic_allocation[core_idx].append(task)
        if task.is_hard:
            node.local_hard_tasks.append(task)
    
    def assign_task(self, task: Task) -> Any:
        return self.__target_node

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        pass