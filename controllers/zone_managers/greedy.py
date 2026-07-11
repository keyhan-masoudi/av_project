from typing import Dict, Unpack, Any, Optional
from controllers.zone_managers.base import ZoneManagerABC, ZoneManagerUpdate
from models.node.fog import FogLayerABC
from models.task import Task
from models.node.base import findExecTimeInEachKindOfNode
from models.node.base import NodeABC


class GreedyZoneManager(ZoneManagerABC):
    def __init__(self, zone):
        super().__init__(zone)
        self.__target_node: Optional[NodeABC] = None

    def _estimate_finish_time(self, task: Task, target_node: NodeABC) -> float:
        """
        Calculates the estimated turnaround time if the task is sent to this node.
        Finish Time = Release Time + Network Delay + Real Execution Time
        """
        # Temporarily assign executor to calculate precise hardware execution time
        original_executor = getattr(task, 'executor', None)
        task.executor = target_node
        
        delay = 0.0
        if target_node != task.creator:
            delay = getattr(target_node, 'get_transmission_time', lambda t, f: 0.0)(task, self.fixed_fog_nodes)
        
        exec_time = findExecTimeInEachKindOfNode(task)
        if exec_time <= 0:
            exec_time = task.exec_time
            
        task.executor = original_executor 
        
        return task.release_time + delay + exec_time

    def can_offload_task(self, task: Task) -> bool:
        """
        ONLINE PHASE: Handles unassigned soft tasks using strict hierarchical priority.
        Priority: User (Local) > MobileFog > FixedFog > Cloud
        """
        if getattr(task, 'is_hard', False):
            return False

        # ==========================================================
        # PRIORITY 1: USER NODE (Local Vehicle)
        # ==========================================================
        local_node = task.creator
        task_window = max(task.deadline - task.release_time, 0.001)
        
        # Isolate executor temporarily to fetch precise frequency scaling
        original_executor = getattr(task, 'executor', None)
        task.executor = local_node
        exec_time = findExecTimeInEachKindOfNode(task)
        if exec_time <= 0:
            exec_time = getattr(task, 'total_exec_time', task.exec_time)
        task.executor = original_executor
            
        soft_utilization = exec_time / task_window
        has_local_slack = False
        
        for core_idx in range(local_node.num_cores):
            # Check against real-time utilization. 0.95 acts as an interference safety buffer.
            if local_node.core_Up[core_idx] + soft_utilization <= 0.95: 
                self.__target_node = local_node
                self._lock_task_parameters(task, self.__target_node)
                
                # Lock the capacity so concurrent offloading attempts register the load
                local_node.core_Up[core_idx] += soft_utilization
                has_local_slack = True
                break
                
        if has_local_slack:
            return True

        # ==========================================================
        # PRIORITY 2: MOBILE FOG NODES
        # ==========================================================
        best_mobile = None
        best_mobile_time = float('inf')
        
        for node_id, node in self.mobile_fog_nodes.items():
            node_finish = self._estimate_finish_time(task, node)
            if node_finish <= task.deadline and node_finish < best_mobile_time:
                best_mobile_time = node_finish
                best_mobile = node
                
        if best_mobile:
            self.__target_node = best_mobile
            self._lock_task_parameters(task, self.__target_node)
            return True

        # ==========================================================
        # PRIORITY 3: FIXED FOG NODES
        # ==========================================================
        # Cleanly iterates only over actual Fixed Fog nodes (Cloud is not here)
        best_fixed = None
        best_fixed_time = float('inf')
        
        for node_id, node in self.fixed_fog_nodes.items():
            node_finish = self._estimate_finish_time(task, node)
            if node_finish <= task.deadline and node_finish < best_fixed_time:
                best_fixed_time = node_finish
                best_fixed = node
                
        if best_fixed:
            self.__target_node = best_fixed
            self._lock_task_parameters(task, self.__target_node)
            return True

        # ==========================================================
        # PRIORITY 4: CLOUD NODE
        # ==========================================================
        # Fetch the cloud node directly from the simulator reference
        if hasattr(self, 'simulator') and getattr(self.simulator, 'cloud_node', None):
            cloud_node = self.simulator.cloud_node
            cloud_finish = self._estimate_finish_time(task, cloud_node)
            
            if cloud_finish <= task.deadline:
                self.__target_node = cloud_node
                self._lock_task_parameters(task, self.__target_node)
                return True
        
        return False

    def _lock_task_parameters(self, task: Task, target_node: NodeABC):
        """Helper to lock in the final assignment variables."""
        task.executor = target_node
        task.exec_time = findExecTimeInEachKindOfNode(task)
        if task.exec_time <= 0:
            task.exec_time = getattr(task, 'total_exec_time', task.exec_time)
        task.remaining_time = task.exec_time

    def assign_task(self, task: Task) -> FogLayerABC:
        return self.__target_node

    def update(self, **kwargs: Unpack[ZoneManagerUpdate]):
        pass