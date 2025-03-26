import random
import numpy as np

class RandomAlgo:
    """
    Random algorithm: Randomly decides between local execution and offloading tasks.
    Each task has 50% probability of being executed locally and 50% probability of being offloaded.
    """
    
    def __init__(self, env):
        """
        Initialize Random algorithm with environment.
        
        Args:
            env: The vehicular edge computing environment
        """
        self.env = env
        self.name = "Random"
    
    def reset(self):
        """Reset the algorithm state"""
        pass  # No state to reset for Random algorithm
    
    def make_decision(self, task):
        """
        Make offloading decision for the given task.
        
        Args:
            task: Task to make decision for
            
        Returns:
            Tuple (offload_decision, comp_resource, bandwidth)
            - offload_decision: 0 for local execution, 1-N for offloading to RSU 1-N
            - comp_resource: Allocated computation resource (only used when offloading)
            - bandwidth: Allocated bandwidth (only used when offloading)
        """
        # 50% probability for local execution, 50% for offloading
        if random.random() < 0.5:
            # Local execution
            return (0, 0, 0)
        else:
            # Offload to a random RSU
            rsu_index = random.randint(0, len(self.env.rsus) - 1)
            rsu = self.env.rsus[rsu_index]
            
            # Allocate random resources (50%-100%范围内)
            comp_resource = random.uniform(0.5, 1.0) * rsu.max_comp_resource
            bandwidth = random.uniform(0.5, 1.0) * rsu.max_bandwidth
            
            return (rsu_index + 1, comp_resource, bandwidth) 