import numpy as np

class LocalOnly:
    """
    Local-Only: All tasks are executed locally on vehicles
    """
    
    def __init__(self, env):
        self.env = env
    
    def make_decision(self, task):
        """
        Make offloading and resource allocation decision for a task
        Always executes locally (offload_decision = 0)
        
        Args:
            task: DAG task to be executed
        
        Returns:
            Tuple of (offload_decision, comp_resource, bandwidth)
        """
        # Always execute locally
        offload_decision = 0
        
        # Local execution doesn't need RSU resources
        comp_resource = 0
        bandwidth = 0
        
        return offload_decision, comp_resource, bandwidth
    
    def reset(self):
        """Reset the model for a new episode"""
        pass  # No state to reset