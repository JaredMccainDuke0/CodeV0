import numpy as np
import networkx as nx

class Vehicle:
    """Vehicle class for simulation"""
    
    def __init__(self, vehicle_id, x, y, max_comp_resource=2.0, energy_coeff=0.5):
        self.id = vehicle_id
        self.x = x
        self.y = y
        self.max_comp_resource = max_comp_resource  # GHz
        self.available_comp_resource = max_comp_resource
        self.energy_coeff = energy_coeff  # Energy consumption coefficient
    
    def reset_resources(self):
        """Reset available resources"""
        self.available_comp_resource = self.max_comp_resource

class RSU:
    """Road-Side Unit class for simulation"""
    
    def __init__(self, rsu_id, x, y, max_comp_resource=10.0, max_bandwidth=100.0):
        self.id = rsu_id
        self.x = x
        self.y = y
        self.max_comp_resource = max_comp_resource  # GHz
        self.available_comp_resource = max_comp_resource
        self.max_bandwidth = max_bandwidth  # Mbps
        self.available_bandwidth = max_bandwidth
    
    def reset_resources(self):
        """Reset available resources"""
        self.available_comp_resource = self.max_comp_resource
        self.available_bandwidth = self.max_bandwidth
    
    def allocate_comp_resource(self, amount):
        """Allocate computation resource"""
        if amount <= self.available_comp_resource:
            self.available_comp_resource -= amount
            return True
        return False
    
    def release_comp_resource(self, amount):
        """Release computation resource"""
        self.available_comp_resource = min(self.available_comp_resource + amount, self.max_comp_resource)
    
    def allocate_bandwidth(self, amount):
        """Allocate bandwidth resource"""
        if amount <= self.available_bandwidth:
            self.available_bandwidth -= amount
            return True
        return False
    
    def release_bandwidth(self, amount):
        """Release bandwidth resource"""
        self.available_bandwidth = min(self.available_bandwidth + amount, self.max_bandwidth)

class Subtask:
    """Subtask class for DAG task"""
    
    def __init__(self, subtask_id, task_id, comp_complexity, data_size, priority=1.0):
        self.id = subtask_id
        self.task_id = task_id
        self.comp_complexity = comp_complexity  # Computation complexity in cycles
        self.data_size = data_size  # Data size in bits
        self.priority = priority  # Task priority
        self.dependencies = []  # List of dependency subtask IDs
    
    def get_feature_vector(self):
        """Get feature vector for the subtask"""
        return [self.comp_complexity, self.data_size, self.priority]
    
    def add_dependency(self, subtask_id):
        """Add dependency to the subtask"""
        if subtask_id not in self.dependencies:
            self.dependencies.append(subtask_id)

class DAGTask:
    """DAG Task class for simulation"""
    
    def __init__(self, task_id, vehicle_id, num_subtasks=5, max_comp=1e9, max_data=1e6):
        self.id = task_id
        self.vehicle_id = vehicle_id
        
        # Generate random DAG structure
        G = nx.DiGraph(nx.gnp_random_graph(num_subtasks, 0.3, directed=True))
        
        # Make it a DAG by removing cycles
        attempts = 0
        while not nx.is_directed_acyclic_graph(G) and attempts < 100:
            cycles = list(nx.simple_cycles(G))
            for cycle in cycles:
                if len(cycle) > 1:
                    # 安全地移除环中的一条边
                    u, v = cycle[0], cycle[1]
                    if G.has_edge(u, v):
                        G.remove_edge(u, v)
                    else:
                        # 如果第一条边不存在，尝试移除环中的另一条边
                        for i in range(1, len(cycle)):
                            u, v = cycle[i], cycle[(i+1) % len(cycle)]
                            if G.has_edge(u, v):
                                G.remove_edge(u, v)
                                break
            attempts += 1
        
        # 如果仍然有环，使用更严格的方法
        if not nx.is_directed_acyclic_graph(G):
            # 直接创建一个无环有向图
            G = nx.DiGraph()
            G.add_nodes_from(range(num_subtasks))
            # 只添加从小节点到大节点的边，确保无环
            for i in range(num_subtasks):
                for j in range(i+1, num_subtasks):
                    if np.random.random() < 0.3:  # 与原来的边概率保持一致
                        G.add_edge(i, j)
        
        # Compute a topological sort
        self.topo_order = list(nx.topological_sort(G))
        
        # Generate subtasks
        self.subtasks = []
        for i in range(num_subtasks):
            comp_complexity = np.random.uniform(1e8, max_comp)
            data_size = np.random.uniform(1e5, max_data)
            priority = np.random.uniform(0.5, 1.0)
            
            subtask = Subtask(i, task_id, comp_complexity, data_size, priority)
            self.subtasks.append(subtask)
        
        # Add dependencies based on the DAG
        for u, v in G.edges():
            self.subtasks[v].add_dependency(u)
    
    def get_feature_vector(self):
        """Get feature vector for the task"""
        # Aggregate subtask features
        total_comp = sum(subtask.comp_complexity for subtask in self.subtasks)
        total_data = sum(subtask.data_size for subtask in self.subtasks)
        avg_priority = np.mean([subtask.priority for subtask in self.subtasks])
        
        return [total_comp, total_data, avg_priority]

def generate_vehicles(num_vehicles, area_size=1000):
    """Generate random vehicles"""
    vehicles = []
    for i in range(num_vehicles):
        x = np.random.uniform(0, area_size)
        y = np.random.uniform(0, area_size)
        # 适当提高车辆的计算频率, 从0.1-0.3 GHz提高到0.2-0.5 GHz
        max_comp = np.random.uniform(0.2, 0.5)  # GHz
        # 适当降低本地计算的能耗系数，从2.0-3.0降低到1.5-1.7
        energy_coeff = np.random.uniform(1.5, 1.7)
        
        vehicle = Vehicle(i, x, y, max_comp, energy_coeff)
        vehicles.append(vehicle)
    
    return vehicles

def generate_rsus(num_rsus, area_size=1000):
    """Generate random RSUs"""
    rsus = []
    for i in range(num_rsus):
        x = np.random.uniform(0, area_size)
        y = np.random.uniform(0, area_size)
        # 增强RSU计算能力, 从5.0-15.0提高到15.0-25.0 GHz
        max_comp = np.random.uniform(15.0, 25.0)  # GHz
        max_bandwidth = np.random.uniform(50.0, 150.0)  # Mbps
        
        rsu = RSU(i, x, y, max_comp, max_bandwidth)
        rsus.append(rsu)
    
    return rsus

def generate_dag_tasks(num_tasks, max_subtasks=10, num_vehicles=10):
    """Generate random DAG tasks"""
    tasks = []
    for i in range(num_tasks):
        vehicle_id = np.random.randint(0, num_vehicles)  # 使用参数控制vehicle_id范围
        num_subtasks = np.random.randint(3, max_subtasks + 1)
        
        task = DAGTask(i, vehicle_id, num_subtasks)
        tasks.append(task)
    
    return tasks