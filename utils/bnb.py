import numpy as np
from queue import PriorityQueue

class BranchAndBound:
    """Branch and Bound algorithm for expert strategy generation"""
    
    def __init__(self, env):
        self.env = env
    
    def solve(self, task):
        """
        Generate expert strategy for a task using B&B
        
        Args:
            task: Task to solve
        
        Returns:
            best_decision: Optimal (offload_decision, comp_resource, bandwidth) tuple
        """
        # Initialize priority queue with root node
        pq = PriorityQueue()
        
        # Root node: (lower_bound, depth, partial_decision, allocated_resources)
        root_lower_bound = self._calculate_lower_bound(task, None)
        root = (root_lower_bound, 0, None, {})
        pq.put(root)
        
        best_decision = None
        best_cost = float('inf')
        
        # B&B main loop
        while not pq.empty():
            lb, depth, partial_decision, allocated_resources = pq.get()
            
            # Pruning: if lower bound exceeds best cost, skip this branch
            if lb >= best_cost:
                continue
            
            # Complete solution found
            if depth == 3:  # All three decisions made (offload, compute, bandwidth)
                # Calculate actual cost
                offload_decision, comp_resource, bandwidth = partial_decision
                time_cost, energy_cost, obj_value = self._calculate_cost(task, partial_decision)
                
                # Update best solution if better
                if obj_value < best_cost:
                    best_cost = obj_value
                    best_decision = partial_decision
                
                continue
            
            # Branch: generate child nodes
            if depth == 0:
                # Offloading decision
                for offload_option in range(len(self.env.rsus) + 1):  # 0 for local, 1+ for RSUs
                    new_decision = (offload_option, 0, 0)  # Placeholder for resource and bandwidth
                    new_lb = self._calculate_lower_bound(task, new_decision)
                    
                    # Skip if infeasible
                    if new_lb == float('inf'):
                        continue
                    
                    # Create child node
                    child = (new_lb, depth + 1, new_decision, allocated_resources.copy())
                    pq.put(child)
            
            elif depth == 1:
                # Resource allocation decision
                offload_decision = partial_decision[0]
                
                # Skip resource allocation for local execution
                if offload_decision == 0:
                    new_decision = (offload_decision, 0, 0)  # No resources needed for local
                    child = (lb, depth + 1, new_decision, allocated_resources.copy())
                    pq.put(child)
                else:
                    # RSU execution - allocate computation resources
                    rsu_id = offload_decision - 1
                    rsu = self.env.rsus[rsu_id]
                    
                    # Try different resource levels
                    for resource_level in range(5):  # 5 discrete levels
                        comp_resource = (resource_level + 1) * 0.2 * rsu.max_comp_resource
                        
                        # Skip if resource not available
                        if comp_resource > rsu.available_comp_resource:
                            continue
                        
                        # Update decision and resources
                        new_decision = (offload_decision, comp_resource, 0)  # Placeholder for bandwidth
                        new_allocated = allocated_resources.copy()
                        new_allocated[f'comp_{rsu_id}'] = comp_resource
                        
                        # Calculate new lower bound
                        new_lb = self._calculate_lower_bound(task, new_decision)
                        
                        # Create child node
                        child = (new_lb, depth + 1, new_decision, new_allocated)
                        pq.put(child)
            
            elif depth == 2:
                # Bandwidth allocation decision
                offload_decision, comp_resource, _ = partial_decision
                
                # Skip bandwidth allocation for local execution
                if offload_decision == 0:
                    new_decision = (offload_decision, comp_resource, 0)  # No bandwidth needed for local
                    child = (lb, depth + 1, new_decision, allocated_resources.copy())
                    pq.put(child)
                else:
                    # RSU execution - allocate bandwidth
                    rsu_id = offload_decision - 1
                    rsu = self.env.rsus[rsu_id]
                    
                    # Try different bandwidth levels
                    for bandwidth_level in range(5):  # 5 discrete levels
                        bandwidth = (bandwidth_level + 1) * 0.2 * rsu.max_bandwidth
                        
                        # Skip if bandwidth not available
                        if bandwidth > rsu.available_bandwidth:
                            continue
                        
                        # Update decision and resources
                        new_decision = (offload_decision, comp_resource, bandwidth)
                        new_allocated = allocated_resources.copy()
                        new_allocated[f'bw_{rsu_id}'] = bandwidth
                        
                        # Calculate actual cost (complete decision)
                        time_cost, energy_cost, obj_value = self._calculate_cost(task, new_decision)
                        
                        # Create child node
                        child = (obj_value, depth + 1, new_decision, new_allocated)
                        pq.put(child)
        
        # If no solution found, return default (local execution)
        if best_decision is None:
            return (0, 0, 0)
        
        return best_decision
    
    def _calculate_lower_bound(self, task, partial_decision):
        """Calculate lower bound cost for a partial decision"""
        if partial_decision is None:
            # Root node - use simple heuristic
            return 0.0
        
        offload_decision = partial_decision[0]
        
        # Local execution - estimate based on vehicle specs
        if offload_decision == 0:
            vehicle = self.env.vehicles[task.vehicle_id]
            task_complexity = task.get_feature_vector()[0]
            
            # Estimate time and energy
            time_est = task_complexity / (vehicle.max_comp_resource * 1e9)
            energy_est = vehicle.energy_coeff * (vehicle.max_comp_resource ** 2) * time_est
            
            return self.env.alpha * time_est + self.env.beta * energy_est
        
        # RSU execution - more complex estimation
        rsu_id = offload_decision - 1
        rsu = self.env.rsus[rsu_id]
        
        # If resources already specified in partial decision
        if len(partial_decision) > 1 and partial_decision[1] > 0:
            comp_resource = partial_decision[1]
        else:
            # Assume best case - all available resources
            comp_resource = rsu.available_comp_resource
        
        if len(partial_decision) > 2 and partial_decision[2] > 0:
            bandwidth = partial_decision[2]
        else:
            # Assume best case - all available bandwidth
            bandwidth = rsu.available_bandwidth
        
        # Calculate metrics - this is a lower bound estimate
        task_complexity = task.get_feature_vector()[0]
        task_data_size = task.get_feature_vector()[1]
        
        # Distance between vehicle and RSU
        vehicle = self.env.vehicles[task.vehicle_id]
        distance = np.sqrt((vehicle.x - rsu.x)**2 + (vehicle.y - rsu.y)**2)
        
        # Transmission rate
        transmission_rate = bandwidth * 1e6  # Simplified, ignoring channel conditions
        
        # Upload time
        upload_time = task_data_size / transmission_rate
        
        # Execution time
        execution_time = task_complexity / (comp_resource * 1e9)
        
        # Total time
        time_est = upload_time + execution_time
        
        # Energy (only transmission energy for offloading)
        transmission_power = 0.5 * (distance / 100.0) ** 2
        energy_est = transmission_power * upload_time
        
        return self.env.alpha * time_est + self.env.beta * energy_est
    
    def _calculate_cost(self, task, decision):
        """Calculate actual cost for a complete decision"""
        offload_decision, comp_resource, bandwidth = decision
        vehicle_id = task.vehicle_id
        vehicle = self.env.vehicles[vehicle_id]
        
        # Local execution
        if offload_decision == 0:
            time_cost = self._calculate_local_time(task, vehicle)
            energy_cost = self._calculate_local_energy(task, vehicle)
        
        # RSU execution
        else:
            rsu_id = offload_decision - 1
            rsu = self.env.rsus[rsu_id]
            
            # Calculate metrics
            time_cost = self._calculate_offload_time(task, vehicle, rsu, comp_resource, bandwidth)
            energy_cost = self._calculate_offload_energy(task, vehicle, rsu, bandwidth)
        
        # Calculate objective value
        obj_value = self.env.alpha * time_cost + self.env.beta * energy_cost
        
        return time_cost, energy_cost, obj_value
    
    def _calculate_local_time(self, task, vehicle):
        """Calculate local execution time"""
        task_complexity = task.get_feature_vector()[0]
        return task_complexity / (vehicle.max_comp_resource * 1e9)
    
    def _calculate_local_energy(self, task, vehicle):
        """Calculate local execution energy"""
        task_complexity = task.get_feature_vector()[0]
        execution_time = task_complexity / (vehicle.max_comp_resource * 1e9)
        power = vehicle.energy_coeff * (vehicle.max_comp_resource ** 2)
        
        return power * execution_time
    
    def _calculate_offload_time(self, task, vehicle, rsu, comp_resource, bandwidth):
        """Calculate offloading time"""
        task_complexity = task.get_feature_vector()[0]
        task_data_size = task.get_feature_vector()[1]
        
        # Distance between vehicle and RSU
        distance = np.sqrt((vehicle.x - rsu.x)**2 + (vehicle.y - rsu.y)**2)
        
        # Transmission rate calculation
        transmission_rate = self._calculate_transmission_rate(distance, bandwidth)
        
        # Upload time
        upload_time = task_data_size / transmission_rate
        
        # Execution time
        execution_time = task_complexity / (comp_resource * 1e9)
        
        return upload_time + execution_time
    
    def _calculate_offload_energy(self, task, vehicle, rsu, bandwidth):
        """Calculate offloading energy (transmission only)"""
        task_data_size = task.get_feature_vector()[1]
        
        # Distance between vehicle and RSU
        distance = np.sqrt((vehicle.x - rsu.x)**2 + (vehicle.y - rsu.y)**2)
        
        # Transmission rate
        transmission_rate = self._calculate_transmission_rate(distance, bandwidth)
        
        # Upload time
        upload_time = task_data_size / transmission_rate
        
        # Transmission power
        transmission_power = 0.5 * (distance / 100.0) ** 2
        
        return transmission_power * upload_time
    
    def _calculate_transmission_rate(self, distance, bandwidth):
        """Calculate transmission rate based on distance and bandwidth"""
        # Path loss model
        path_loss_exponent = 4.0
        reference_distance = 1.0
        reference_gain = 1.0
        
        # Calculate channel gain
        if distance < reference_distance:
            channel_gain = reference_gain
        else:
            channel_gain = reference_gain * (reference_distance / distance) ** path_loss_exponent
        
        # Calculate SNR (simplified)
        transmission_power = 0.1  # W
        noise_power = 1e-10  # W
        snr = transmission_power * channel_gain / noise_power
        
        # Shannon capacity
        rate = bandwidth * 1e6 * np.log2(1 + snr)  # Convert Mbps to bps
        
        return rate