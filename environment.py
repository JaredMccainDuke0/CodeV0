import numpy as np
import torch
from utils.metrics import calculate_time_delay, calculate_energy_consumption

class VehicularEdgeEnvironment:
    """Simulation environment for vehicular edge computing system"""
    
    def __init__(self, vehicles, rsus, alpha=0.5, beta=0.5):
        """
        Initialize environment
        
        Args:
            vehicles: List of vehicle objects
            rsus: List of RSU objects
            alpha: Weight for time in objective function
            beta: Weight for energy in objective function
        """
        self.vehicles = vehicles
        self.rsus = rsus
        self.alpha = alpha
        self.beta = beta
        self.tasks = []
        self.cache = {rsu.id: {} for rsu in rsus}  # Cache for each RSU
        self.time = 0  # System time
        
        # Calculate distances between vehicles and RSUs
        self.distances = np.zeros((len(vehicles), len(rsus)))
        for i, vehicle in enumerate(vehicles):
            for j, rsu in enumerate(rsus):
                self.distances[i, j] = np.sqrt((vehicle.x - rsu.x)**2 + (vehicle.y - rsu.y)**2)
    
    def reset(self, tasks):
        """Reset environment with new tasks"""
        self.tasks = tasks
        self.time = 0
        
        # Reset RSU resources
        for rsu in self.rsus:
            rsu.reset_resources()
        
        # Reset vehicle resources
        for vehicle in self.vehicles:
            vehicle.reset_resources()
    
    def execute_decision(self, task, decision):
        """
        Execute offloading decision and calculate performance metrics
        
        Args:
            task: DAG task to be executed
            decision: Tuple of (offload_decision, comp_resource, bandwidth)
                offload_decision: 0 for local, rsu_id for offloading
                comp_resource: Allocated computation resource
                bandwidth: Allocated bandwidth
        
        Returns:
            time_cost: Task completion time
            energy_cost: Energy consumption
            obj_value: Objective function value
        """
        offload_decision, comp_resource, bandwidth = decision
        vehicle_id = task.vehicle_id
        vehicle = self.vehicles[vehicle_id]
        
        # Local execution
        if offload_decision == 0:
            time_cost = calculate_time_delay(task, vehicle, None, is_local=True)
            energy_cost = calculate_energy_consumption(task, vehicle, None, is_local=True)
        
        # Offload to RSU
        else:
            rsu_id = offload_decision - 1  # Convert to 0-indexed
            rsu = self.rsus[rsu_id]
            
            # Check for computation reuse opportunities
            computation_reused = False
            if hasattr(task, 'subtasks'):
                for subtask in task.subtasks:
                    # 将特征向量转换为元组形式
                    subtask_features = tuple(subtask.get_feature_vector())
                    
                    # Check cache for similar tasks
                    for cached_task, cached_result in self.cache[rsu.id].items():
                        if self._is_similar(subtask_features, cached_task):
                            # Reuse computation result
                            computation_reused = True
                            break
            
            # Allocate resources
            rsu.allocate_comp_resource(comp_resource)
            rsu.allocate_bandwidth(bandwidth)
            
            # Calculate metrics considering computation reuse
            reuse_factor = 0.05 if computation_reused else 1.0  # 95% reduction if reused (原来是30%)
            time_cost = calculate_time_delay(task, vehicle, rsu, is_local=False, 
                                            distance=self.distances[vehicle_id, rsu_id],
                                            bandwidth=bandwidth, 
                                            comp_resource=comp_resource,
                                            reuse_factor=reuse_factor)
            energy_cost = calculate_energy_consumption(task, vehicle, rsu, is_local=False,
                                                    distance=self.distances[vehicle_id, rsu_id],
                                                    bandwidth=bandwidth,
                                                    reuse_factor=reuse_factor)
            
            # Update cache with task result
            if hasattr(task, 'subtasks'):
                for subtask in task.subtasks:
                    feature_key = tuple(subtask.get_feature_vector())
                    
                    # 如果已存在，增加访问计数
                    if feature_key in self.cache[rsu.id]:
                        self.cache[rsu.id][feature_key]['access_count'] += 1
                    else:
                        # 新增缓存条目
                        self.cache[rsu.id][feature_key] = {
                            'result': f"Result for subtask {subtask.id}",
                            'access_count': 1
                        }
            
            # Release resources
            rsu.release_comp_resource(comp_resource)
            rsu.release_bandwidth(bandwidth)
        
        # Calculate objective function value
        obj_value = self.alpha * time_cost + self.beta * energy_cost
        
        return time_cost, energy_cost, obj_value
    
    def _is_similar(self, feature_vector1, feature_vector2, similarity_threshold=0.60):
        """Check if two feature vectors are similar"""
        # Calculate cosine similarity
        vec1 = np.array(feature_vector1)
        vec2 = np.array(feature_vector2)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # 避免除零错误
        if norm1 == 0 or norm2 == 0:
            return False
        
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        
        return similarity > similarity_threshold
    
    def get_state(self, task, vehicle_id):
        """Get environment state for a specific task and vehicle"""
        vehicle = self.vehicles[vehicle_id]
        
        # Task features
        task_features = task.get_feature_vector()
        
        # RSU available resources
        rsu_resources = []
        for rsu in self.rsus:
            rsu_resources.append([
                rsu.available_comp_resource / rsu.max_comp_resource,
                rsu.available_bandwidth / rsu.max_bandwidth
            ])
        
        # Vehicle-RSU channel conditions
        channel_conditions = []
        for i, rsu in enumerate(self.rsus):
            distance = self.distances[vehicle_id, i]
            channel_gain = self._calculate_channel_gain(distance)
            channel_conditions.append(channel_gain)
        
        # 确保vehicle_resources是列表而不是单个值
        vehicle_resources = [vehicle.available_comp_resource / vehicle.max_comp_resource]
        
        # Combine all state information
        state = {
            'task_features': task_features,
            'vehicle_resources': vehicle_resources,
            'rsu_resources': rsu_resources,
            'channel_conditions': channel_conditions
        }
        
        return state
    
    def _calculate_channel_gain(self, distance):
        """Calculate channel gain based on distance"""
        path_loss_exponent = 4.0
        reference_distance = 1.0
        reference_gain = 1.0
        
        if distance < reference_distance:
            return reference_gain
        
        return reference_gain * (reference_distance / distance) ** path_loss_exponent