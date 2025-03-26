import numpy as np

def calculate_time_delay(task, vehicle, rsu, is_local=True, distance=None, bandwidth=None, comp_resource=None, reuse_factor=1.0):
    """
    Calculate time delay for task execution
    
    Args:
        task: Task to be executed
        vehicle: Vehicle executing or offloading the task
        rsu: RSU receiving the task (None if local execution)
        is_local: Whether the task is executed locally
        distance: Distance between vehicle and RSU (if offloading)
        bandwidth: Allocated bandwidth (if offloading)
        comp_resource: Allocated computation resource (if offloading)
        reuse_factor: Computation reuse factor (1.0 means no reuse)
    
    Returns:
        time_delay: Time delay for task execution
    """
    if is_local:
        # Local execution time
        task_complexity = task.get_feature_vector()[0]  # Get computation complexity
        # 添加本地处理额外开销因子，模拟资源受限设备的额外处理开销
        local_overhead_factor = 1.5  # 本地处理额外开销因子
        return local_overhead_factor * task_complexity / (vehicle.max_comp_resource * 1e9)  # Convert GHz to Hz
    else:
        # Offloading time
        task_data_size = task.get_feature_vector()[1]  # Get data size
        
        # Calculate upload time
        transmission_rate = calculate_transmission_rate(distance, bandwidth)
        upload_time = task_data_size / transmission_rate
        
        # Calculate execution time at RSU
        task_complexity = task.get_feature_vector()[0] * reuse_factor  # Apply reuse factor
        execution_time = task_complexity / (comp_resource * 1e9)  # Convert GHz to Hz
        
        # Add fixed finding time for reused computation (if reuse_factor < 1.0)
        finding_time = 0.01 if reuse_factor < 1.0 else 0.0
        
        return upload_time + finding_time + execution_time

def calculate_energy_consumption(task, vehicle, rsu, is_local=True, distance=None, bandwidth=None, reuse_factor=1.0):
    """
    Calculate energy consumption for task execution
    
    Args:
        task: Task to be executed
        vehicle: Vehicle executing or offloading the task
        rsu: RSU receiving the task (None if local execution)
        is_local: Whether the task is executed locally
        distance: Distance between vehicle and RSU (if offloading)
        bandwidth: Allocated bandwidth (if offloading)
        reuse_factor: Computation reuse factor (1.0 means no reuse)
    
    Returns:
        energy: Energy consumption for task execution
    """
    if is_local:
        # Local execution energy
        task_complexity = task.get_feature_vector()[0]  # Get computation complexity
        execution_time = task_complexity / (vehicle.max_comp_resource * 1e9)  # Convert GHz to Hz
        
        # 修正功率模型：计算性能越低，单位时间能耗越高
        # 使用性能倒数的平方，这样性能减半，能耗增加4倍
        power = vehicle.energy_coeff * ((1.0 / vehicle.max_comp_resource) ** 2)
        
        return power * execution_time
    else:
        # Offloading energy (only transmission energy)
        task_data_size = task.get_feature_vector()[1]  # Get data size
        
        # Calculate upload energy
        transmission_rate = calculate_transmission_rate(distance, bandwidth)
        upload_time = task_data_size / transmission_rate
        
        # 进一步降低传输功率系数，从0.2降低到0.1
        transmission_power = 0.1 * (distance / 100.0) ** 2
        
        return transmission_power * upload_time

def calculate_transmission_rate(distance, bandwidth):
    """
    Calculate transmission rate based on distance and bandwidth
    
    Args:
        distance: Distance between vehicle and RSU
        bandwidth: Allocated bandwidth in Mbps
    
    Returns:
        rate: Transmission rate in bps
    """
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
    # 降低传输功率，从0.1降低到0.05 W
    transmission_power = 0.05  # W
    noise_power = 1e-10  # W
    snr = transmission_power * channel_gain / noise_power
    
    # Shannon capacity
    rate = bandwidth * 1e6 * np.log2(1 + snr)  # Convert Mbps to bps
    
    return rate

def compute_metrics(env, tasks, decisions):
    """
    Compute performance metrics for a set of tasks and decisions
    
    Args:
        env: Environment
        tasks: List of tasks
        decisions: List of decisions for each task
    
    Returns:
        avg_time: Average task completion time
        avg_energy: Average energy consumption
        avg_obj: Average objective function value
    """
    total_time = 0
    total_energy = 0
    total_obj = 0
    
    for i, task in enumerate(tasks):
        decision = decisions[i]
        time_cost, energy_cost, obj_value = env.execute_decision(task, decision)
        
        total_time += time_cost
        total_energy += energy_cost
        total_obj += obj_value
    
    # Calculate averages
    avg_time = total_time / len(tasks)
    avg_energy = total_energy / len(tasks)
    avg_obj = total_obj / len(tasks)
    
    return avg_time, avg_energy, avg_obj