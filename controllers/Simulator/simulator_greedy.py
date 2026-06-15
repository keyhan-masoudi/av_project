from controllers.simulator import Simulator
from models.task import Task
from models.node.base import MobileNodeABC, findExecTimeInEachKindOfNode
from typing import Dict, List
from config import Config


class SimulatorGreedy(Simulator):
    def __init__(self, loader, clock, cloud):
        super().__init__(loader, clock, cloud)
        
    def load_hard_tasks(self, current_time: float) -> int:
        """
        Dynamically loads hard tasks at each clock tick.
        Combines Algorithm 1 (Fast Test) and Algorithm 2 (Deep Test) into a single pipeline.
        """
        loaded_count = 0
        new_hard_tasks = self.loader.load_nodes_hard_tasks(current_time)
        
        for creator_id, new_tasks in new_hard_tasks.items():
            creator = self._resolve_task_creator(creator_id)
            if creator is None:
                print(f"There is no creator for hard task: {creator_id}\n")
                continue
                
            # 1. Sort the incoming tasks for this tick by frequency (1/period) descending
            # todo : use task.frequency instead of calculating it if added to the Task model
            new_tasks.sort(
                key=lambda t: 1.0 / max(t.deadline - t.release_time, 0.001), 
                reverse=True
            )
            
            for task in new_tasks:
                # todo : use task.frequency instead of calculating it if added to the Task model
                period = max(task.deadline - task.release_time, 0.001)
                task_utilization = task.exec_time / period
                assigned = False
                
                # ==========================================================
                # ALGORITHM 1: Fast Utilization Test (Limit <= 0.693)
                # ==========================================================
                for core_idx in range(creator.num_cores):
                    if creator.core_Up[core_idx] + task_utilization <= 0.693:
                        # Success: Assign via Alg 1
                        self._assign_hard_task_locally(task, creator, current_time, core_idx)
                        self.metrics.inc_total_tasks()
                        loaded_count += 1
                        assigned = True
                        break # Stop checking cores
                        
                # If Algorithm 1 succeeded, move immediately to the next task
                if assigned:
                    continue
                    
                # ==========================================================
                # ALGORITHM 2: Demand-Supply Analysis (SBF >= DBF)
                # ==========================================================
                # If Alg 1 failed, we immediately try the rigorous math.
                # Paper Rule: Sort cores by current utilization (emptiest first).
                core_utils = [(i, creator.core_Up[i]) for i in range(creator.num_cores)]
                sorted_cores = sorted(core_utils, key=lambda x: x[1])
                
                # Demand-Bound Function (DBF) is the absolute execution time needed
                dbf = task.exec_time 
                
                for core_idx, core_util in sorted_cores:
                    # Supply-Bound Function (SBF) is the free time left on this core
                    sbf_utilization_available = 1.0 - core_util
                    sbf_time_available = sbf_utilization_available * period
                    
                    if sbf_time_available >= dbf:
                        # Success: Assign via Alg 2
                        self._assign_hard_task_locally(task, creator, current_time, core_idx)
                        self.metrics.inc_total_tasks()
                        loaded_count += 1
                        assigned = True
                        break # Stop checking cores
                
                # ==========================================================
                # SAFETY CHECK: Task cannot be scheduled
                # ==========================================================
                if not assigned:
                    print(f"[CRITICAL FAILURE] Hard task {task.id} failed both Alg 1 and Alg 2 on {creator.id}.")
                    # Task is discarded. In a real vehicle, this means the ECU is overloaded.

        return loaded_count

    def _assign_hard_task_locally(
            self,
            task: Task,
            creator: MobileNodeABC,
            current_time: float,
            core_idx: int
    ) -> None:
        """
        Finalizes the assignment of a hard task to the specific core mathematically 
        chosen by Algorithm 1 or Algorithm 2.
        """
        import heapq
        
        # 1. Initialize Task State
        task.creator = creator
        task.executor = creator
        task.creator_id = creator.id
        task.release_time = current_time
        task.start_time = current_time
        task.is_hard = True
        
        exec_time_calculated = findExecTimeInEachKindOfNode(task)
        task.exec_time = exec_time_calculated if exec_time_calculated > 0 else task.exec_time
        task.remaining_time = task.exec_time
        
        # Calculate Utilization
        # todo : use task.frequency instead of calculating it if added to the Task model
        # todo : change task.exec_time to task.WCET
        period = max(task.deadline - task.release_time, 0.001)
        task_utilization = task.exec_time / period
        
        # 2. Update Core RM Utilization (Preventing Core Overload)
        creator.core_Up[core_idx] += task_utilization
        creator.core_loads[core_idx] += task.exec_time
        
        # 3. Update CORE-SPECIFIC Task Lists
        creator.periodic_allocation[core_idx].append(task)
        
        # Re-sort this core's list by required frequency (1/period) to maintain RM priority
        creator.periodic_allocation[core_idx].sort(
            key=lambda t: 1.0 / max(t.deadline - t.release_time, 0.001),
            reverse=True
        )
        
        if task not in creator.local_hard_tasks:
            creator.local_hard_tasks.append(task)
            
        # 4. Push to the Execution Engine using strict RATE MONOTONIC (RM) Priority!
        # Because RM always runs the task with the shortest period first, 
        # we put 'period' as the first item in the heap tuple, NOT deadline.
        heapq.heappush(creator.cores[core_idx], (period, task.release_time, task))   
    
    def load_soft_tasks(self, current_time: float) -> Dict[str, List[Task]]:
        """
        Overwritten: Fetches soft tasks and executes Algorithm 3 (Local Soft Packing).
        Returns unassigned tasks if GreedyZoneManager is active, otherwise returns all tasks.
        """
        from collections import defaultdict
        
        # 1. Fetch the raw data (Mimicking the parent class)
        raw_tasks: Dict[str, List[Task]] = defaultdict(list)
        for creator_id, creator_tasks in self.loader.load_nodes_tasks(current_time).items():
            creator = self._resolve_task_creator(creator_id)
            if creator is None:
                print(f"There is no creator for soft task: {creator_id}\n")
                continue
            for task in creator_tasks:
                task.creator = creator
                task.is_hard = False
                raw_tasks[creator_id].append(task)
                
        # 2. Check if we are running the Paper Baseline
        # If not, just return the raw tasks so your other algorithms work normally.
        if self.zone_manager.__class__.__name__ != "GreedyZoneManager":
            return raw_tasks
            
        # ==========================================================
        # 3. ALGORITHM 3 LOGIC (For GreedyZoneManager)
        # ==========================================================
        unassigned_dict: Dict[str, List[Task]] = defaultdict(list)
        alg1_perfect_success = getattr(self, 'alg1_perfect_success', True)
        
        for creator_id, tasks in raw_tasks.items():
            creator = self._resolve_task_creator(creator_id)
            
            # Note: No sorting here because soft tasks are aperiodic
            for task in tasks:
                window = max(task.deadline - task.release_time, 0.001)
                task_utilization = task.exec_time / window
                assigned = False
                
                # --- Step 1: Fast Utilization Test ---
                if alg1_perfect_success:
                    for core_idx in range(creator.num_cores):
                        if creator.core_Up[core_idx] + task_utilization <= 0.693:
                            self._assign_soft_task_locally(task, creator, current_time, core_idx)
                            self.metrics.inc_total_tasks()
                            assigned = True
                            break
                            
                if assigned:
                    continue
                    
                # --- Step 2: Deep Demand-Supply Test ---
                # Sort cores by current utilization (emptiest first)
                core_utils = [(i, creator.core_Up[i]) for i in range(creator.num_cores)]
                sorted_cores = sorted(core_utils, key=lambda x: x[1])
                dbf = task.exec_time 
                
                for core_idx, core_util in sorted_cores:
                    sbf_time_available = (1.0 - core_util) * window
                    
                    if sbf_time_available >= dbf:
                        self._assign_soft_task_locally(task, creator, current_time, core_idx)
                        self.metrics.inc_total_tasks()
                        assigned = True
                        break
                
                # --- Step 3: Leftover Pool ---
                if not assigned:
                    unassigned_dict[creator_id].append(task)
                    
        return unassigned_dict
        
    def _assign_soft_task_locally(
            self,
            task: Task,
            creator: MobileNodeABC,
            current_time: float,
            core_idx: int
    ) -> None:
        """
        Finalizes the assignment of an APERIODIC SOFT task mathematically chosen by Algorithm 3.
        """
        import heapq
        
        task.creator = creator
        task.executor = creator
        task.creator_id = creator.id
        task.release_time = current_time
        task.start_time = current_time
        task.is_hard = False
        
        exec_time_calculated = findExecTimeInEachKindOfNode(task)
        task.exec_time = exec_time_calculated if exec_time_calculated > 0 else task.exec_time
        task.remaining_time = task.exec_time
        
        window = max(task.deadline - task.release_time, 0.001)
        task_utilization = task.exec_time / window
        
        # 1. Update Core Math (Consume the slack)
        creator.core_Up[core_idx] += task_utilization
        creator.core_loads[core_idx] += task.exec_time
        
        # 2. Add to allocation list (DO NOT SORT IT, it is aperiodic)
        creator.periodic_allocation[core_idx].append(task)
            
        # 3. Push to Execution Engine
        # Since aperiodic tasks don't have a repeating period, we push them into the heap
        # using their absolute deadline so they run in the background safely.
        heapq.heappush(creator.cores[core_idx], (task.deadline, task.release_time, task))
        