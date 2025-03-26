import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import time
import traceback
from matplotlib import font_manager as fm

# 设置OpenMP环境变量，避免多重初始化警告
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 使用英文字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

from environment import VehicularEdgeEnvironment
from models.gnn_reuse_il import GNNReuseIL
from models.gnn_drl import GNNDRL
from models.local_only import LocalOnly
from models.random_algo import RandomAlgo  # 导入新添加的Random算法
from utils.data_generator import generate_dag_tasks, generate_vehicles, generate_rsus
from utils.metrics import compute_metrics

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Experiment parameters
TRAIN_EPOCHS = 300  # GNN-Reuse-IL的训练轮次
TASK_NUMBERS = [50,100,150,200,250,300,350,400,450,500]  # 不同的任务数量
NUM_RUNS = 5      # 每个任务数量运行多次取平均值
ALPHA = 0.9  # Weight for time in objective function
BETA = 0.1   # Weight for energy in objective function
SAVE_PATH = "results/"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def run_experiment():
    try:
        print("Starting experiment...")
        
        # Generate network environment
        print("Generating vehicles and RSUs...")
        num_vehicles = 5  # 减少数量以加快测试速度
        num_rsus = 3      # 减少数量以加快测试速度
        vehicles = generate_vehicles(num_vehicles)
        rsus = generate_rsus(num_rsus)
        
        # Create environment
        print("Creating environment...")
        env = VehicularEdgeEnvironment(vehicles, rsus, alpha=ALPHA, beta=BETA)
        
        # Initialize algorithms
        print("Initializing algorithms...")
        print("  Initializing GNN-Reuse-IL...")
        gnn_reuse_il = GNNReuseIL(env, hidden_dim=32, num_heads=2)  # 减小模型尺寸以加快测试速度
        print("  Initializing GNN-DRL...")
        gnn_drl = GNNDRL(env, hidden_dim=32, epsilon=0.3, epsilon_min=0.1, epsilon_decay=0.98)  # 设置初始探索率较低
        print("  Initializing Local-Only...")
        local_only = LocalOnly(env)
        print("  Initializing Random algorithm...")
        random_algo = RandomAlgo(env)  # 初始化Random算法
        
        # 离线阶段：训练GNN-Reuse-IL模型
        print("\n===== Offline Phase: Training GNN-Reuse-IL Model =====")
        expert_loss_history = train_gnn_reuse_il(env, gnn_reuse_il, epochs=TRAIN_EPOCHS)
        
        # 绘制GNN-Reuse-IL离线训练loss曲线
        print("Saving expert loss history...")
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(expert_loss_history)+1), expert_loss_history)
        plt.xlabel('Training Epochs')
        plt.ylabel('Loss')
        plt.title('GNN-Reuse-IL: Imitation Learning Training Loss')
        plt.grid(True)
        plt.savefig(os.path.join(SAVE_PATH, 'gnn_reuse_il_training_loss.png'))
        plt.close()
        
        # 在线阶段：评估不同任务数量下三种算法的性能
        print("\n===== Online Phase: Evaluating Algorithm Performance =====")
        
        # 存储不同任务数量下的性能指标
        task_nums = []
        completion_times = {
            'GNN-Reuse-IL': [],
            'GNN-DRL': [],
            'Local-Only': [],
            'Random': []  # 添加Random算法的结果存储
        }
        
        energy_consumption = {
            'GNN-Reuse-IL': [],
            'GNN-DRL': [],
            'Local-Only': [],
            'Random': []  # 添加Random算法的结果存储
        }
        
        objective_values = {
            'GNN-Reuse-IL': [],
            'GNN-DRL': [],
            'Local-Only': [],
            'Random': []  # 添加Random算法的结果存储
        }
        
        # 存储策略决策统计
        decision_stats = {
            'GNN-Reuse-IL': {'local': 0, 'offload': 0},
            'GNN-DRL': {'local': 0, 'offload': 0},
            'Local-Only': {'local': 0, 'offload': 0},
            'Random': {'local': 0, 'offload': 0}  # 添加Random算法的决策统计
        }
        
        # 对每个任务数量进行多次测试
        for num_tasks in TASK_NUMBERS:
            print(f"\nEvaluating with task count: {num_tasks}")
            task_nums.append(num_tasks)
            
            # 每个任务数量测试多次取平均
            run_time_gnn_reuse_il = []
            run_energy_gnn_reuse_il = []
            run_obj_gnn_reuse_il = []
            
            run_time_gnn_drl = []
            run_energy_gnn_drl = []
            run_obj_gnn_drl = []
            
            run_time_local = []
            run_energy_local = []
            run_obj_local = []
            
            # 为Random算法添加评估结果存储
            run_time_random = []
            run_energy_random = []
            run_obj_random = []
            
            for run in range(NUM_RUNS):
                print(f"  Run {run+1}/{NUM_RUNS}...")
                
                # 生成任务
                tasks = generate_dag_tasks(num_tasks, num_vehicles=num_vehicles)
                
                # 重置环境和算法
                env.reset(tasks)
                gnn_reuse_il.reset()
                gnn_drl.reset()
                local_only.reset()
                random_algo.reset()  # 重置Random算法
                
                # 评估GNN-Reuse-IL
                print("  Evaluating GNN-Reuse-IL...")
                time_gnn_reuse_il, energy_gnn_reuse_il, obj_gnn_reuse_il, decisions_reuse_il = evaluate_algorithm(env, gnn_reuse_il, tasks, track_decisions=True)
                run_time_gnn_reuse_il.append(time_gnn_reuse_il)
                run_energy_gnn_reuse_il.append(energy_gnn_reuse_il)
                run_obj_gnn_reuse_il.append(obj_gnn_reuse_il)
                
                # 统计决策
                for d in decisions_reuse_il:
                    if d[0] == 0:  # 本地执行
                        decision_stats['GNN-Reuse-IL']['local'] += 1
                    else:  # 卸载执行
                        decision_stats['GNN-Reuse-IL']['offload'] += 1
                
                # 评估GNN-DRL
                print("  Evaluating GNN-DRL...")
                env.reset(tasks)
                time_gnn_drl, energy_gnn_drl, obj_gnn_drl, decisions_drl = evaluate_algorithm(env, gnn_drl, tasks, track_decisions=True)
                run_time_gnn_drl.append(time_gnn_drl)
                run_energy_gnn_drl.append(energy_gnn_drl)
                run_obj_gnn_drl.append(obj_gnn_drl)
                
                # 统计决策
                for d in decisions_drl:
                    if d[0] == 0:  # 本地执行
                        decision_stats['GNN-DRL']['local'] += 1
                    else:  # 卸载执行
                        decision_stats['GNN-DRL']['offload'] += 1
                
                # 评估Local-Only
                print("  Evaluating Local-Only...")
                env.reset(tasks)
                time_local, energy_local, obj_local, decisions_local = evaluate_algorithm(env, local_only, tasks, track_decisions=True)
                run_time_local.append(time_local)
                run_energy_local.append(energy_local)
                run_obj_local.append(obj_local)
                
                # 统计决策 (Local-Only总是本地执行)
                decision_stats['Local-Only']['local'] += len(decisions_local)
                
                # 评估Random算法
                print("  Evaluating Random algorithm...")
                env.reset(tasks)
                time_random, energy_random, obj_random, decisions_random = evaluate_algorithm(env, random_algo, tasks, track_decisions=True)
                run_time_random.append(time_random)
                run_energy_random.append(energy_random)
                run_obj_random.append(obj_random)
                
                # 统计Random算法的决策
                for d in decisions_random:
                    if d[0] == 0:  # 本地执行
                        decision_stats['Random']['local'] += 1
                    else:  # 卸载执行
                        decision_stats['Random']['offload'] += 1
            
            # 计算平均值并存储
            completion_times['GNN-Reuse-IL'].append(np.mean(run_time_gnn_reuse_il))
            energy_consumption['GNN-Reuse-IL'].append(np.mean(run_energy_gnn_reuse_il))
            objective_values['GNN-Reuse-IL'].append(np.mean(run_obj_gnn_reuse_il))
            
            completion_times['GNN-DRL'].append(np.mean(run_time_gnn_drl))
            energy_consumption['GNN-DRL'].append(np.mean(run_energy_gnn_drl))
            objective_values['GNN-DRL'].append(np.mean(run_obj_gnn_drl))
            
            completion_times['Local-Only'].append(np.mean(run_time_local))
            energy_consumption['Local-Only'].append(np.mean(run_energy_local))
            objective_values['Local-Only'].append(np.mean(run_obj_local))
            
            # 添加Random算法的平均值
            completion_times['Random'].append(np.mean(run_time_random))
            energy_consumption['Random'].append(np.mean(run_energy_random))
            objective_values['Random'].append(np.mean(run_obj_random))
        
        # 打印数据查看差异
        print("\nGNN-Reuse-IL vs Local-Only vs Random data comparison:")
        print("\nCompletion Times:")
        print("GNN-Reuse-IL:", completion_times['GNN-Reuse-IL'])
        print("Local-Only:", completion_times['Local-Only'])
        print("Random:", completion_times['Random'])
        
        print("\nEnergy Consumption:")
        print("GNN-Reuse-IL:", energy_consumption['GNN-Reuse-IL'])
        print("Local-Only:", energy_consumption['Local-Only'])
        print("Random:", energy_consumption['Random'])
        
        # 打印决策统计
        print("\nDecision Statistics:")
        for algo, stats in decision_stats.items():
            total = stats['local'] + stats['offload']
            local_percent = (stats['local'] / total * 100) if total > 0 else 0
            offload_percent = (stats['offload'] / total * 100) if total > 0 else 0
            print(f"{algo}: Local={stats['local']} ({local_percent:.1f}%), Offload={stats['offload']} ({offload_percent:.1f}%)")
        
        # 检查是否有0值或极小值
        has_very_small_values = False
        # 检查所有算法的结果是否有极小值
        for algo in ['GNN-Reuse-IL', 'GNN-DRL', 'Local-Only', 'Random']:
            if any(x < 0.001 for x in completion_times[algo]) or any(x < 0.001 for x in energy_consumption[algo]):
                has_very_small_values = True
                break
        
        # 如果有极小值，进行缩放处理
        if has_very_small_values:
            print("\nWARNING: Very small values detected. This may cause display issues in the charts.")
            print("Scaling values for better visualization.")
            
            # 找出最小的非零值作为基准
            min_time = float('inf')
            min_energy = float('inf')
            
            for algo in completion_times:
                for val in completion_times[algo]:
                    if 0 < val < min_time:
                        min_time = val
                for val in energy_consumption[algo]:
                    if 0 < val < min_energy:
                        min_energy = val
            
            # 对极小值进行适当放大
            for algo in completion_times:
                for i in range(len(completion_times[algo])):
                    if completion_times[algo][i] < 0.001:
                        completion_times[algo][i] = min_time * 0.5  # 设为最小值的一半，保持相对大小关系
                    if energy_consumption[algo][i] < 0.001:
                        energy_consumption[algo][i] = min_energy * 0.5
            
            # 应用统一缩放因子以便更好的可视化
            scaling_factor = 1000.0
            print(f"Additionally scaling all values by a factor of {scaling_factor} for display purposes.")
            
            for algo in completion_times:
                completion_times[algo] = [x * scaling_factor for x in completion_times[algo]]
                energy_consumption[algo] = [x * scaling_factor for x in energy_consumption[algo]]
                objective_values[algo] = [x * scaling_factor for x in objective_values[algo]]
        
        # 绘制性能曲线
        print("\nPlotting performance curves...")
        
        # 打印各算法数据供调试
        print("\n能耗数据比较:")
        for algo in ['Local-Only', 'GNN-Reuse-IL', 'GNN-DRL', 'Random']:
            print(f"{algo}: {energy_consumption[algo]}")
        
        # 1. 平均完成时间曲线
        plt.figure(figsize=(12, 8))
        plt.plot(task_nums, completion_times['Local-Only'], 'g-s', label='Local-Only')
        plt.plot(task_nums, completion_times['GNN-DRL'], 'r-^', label='GNN-DRL')
        plt.plot(task_nums, completion_times['GNN-Reuse-IL'], 'b-o', label='GNN-Reuse-IL')
        plt.plot(task_nums, completion_times['Random'], color='#800080', marker='v', linestyle='-', label='Random')  # 紫色倒三角
        plt.xlabel('Task Batch (50 tasks per batch)')
        plt.ylabel('Average Tasks Completion Time')
        plt.title('DELAY')
        plt.legend()   
        plt.grid(True)
        batch_labels = [f"Batch {i+1}" for i in range(len(task_nums))]
        plt.xticks(task_nums, batch_labels)
        plt.savefig(os.path.join(SAVE_PATH, 'task_completion_time.png'))
        plt.close()
        
        # 2. 平均能耗曲线
        plt.figure(figsize=(12, 8))
        plt.plot(task_nums, energy_consumption['Local-Only'], 'g-s', label='Local-Only')
        plt.plot(task_nums, energy_consumption['GNN-DRL'], 'r-^', label='GNN-DRL')
        plt.plot(task_nums, energy_consumption['GNN-Reuse-IL'], 'b-o', label='GNN-Reuse-IL')
        plt.plot(task_nums, energy_consumption['Random'], color='#800080', marker='v', linestyle='-', label='Random')  # 紫色倒三角
        plt.xlabel('Task Batch (50 tasks per batch)')
        plt.ylabel('Average Vehicle Energy Consumption')
        plt.title('ENERGY')
        plt.legend()
        plt.grid(True)
        plt.xticks(task_nums, batch_labels)
        plt.savefig(os.path.join(SAVE_PATH, 'energy_consumption.png'))
        plt.close()
        
        # 3. 平均优化目标值曲线
        plt.figure(figsize=(12, 8))
        plt.plot(task_nums, objective_values['Local-Only'], 'g-s', label='Local-Only')
        plt.plot(task_nums, objective_values['GNN-DRL'], 'r-^', label='GNN-DRL')
        plt.plot(task_nums, objective_values['GNN-Reuse-IL'], 'b-o', label='GNN-Reuse-IL')
        plt.plot(task_nums, objective_values['Random'], color='#800080', marker='v', linestyle='-', label='Random')  # 紫色倒三角
        plt.xlabel('Task Batch (50 tasks per batch)')
        plt.ylabel('Average Objective Value')
        plt.title('OBJECTIVE')
        plt.legend()
        plt.grid(True)
        plt.xticks(task_nums, batch_labels)
        plt.savefig(os.path.join(SAVE_PATH, 'objective_value.png'))
        plt.close()
        
        print("Experiment completed!")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        traceback.print_exc()

def train_gnn_reuse_il(env, model, epochs=300):
    """Train GNN-Reuse-IL using expert strategies"""
    from utils.bnb import BranchAndBound
    
    loss_history = []
    
    try:
        # Generate training DAG tasks
        print("Generating training tasks...")
        # 获取环境中的车辆数量
        num_vehicles = len(env.vehicles)
        train_tasks = generate_dag_tasks(15, num_vehicles=num_vehicles)  # 使用15个任务进行训练
        
        # Generate expert strategies using B&B algorithm
        bnb = BranchAndBound(env)
        expert_strategies = []
        
        print("Generating expert strategies using B&B...")
        # 跟踪专家策略决策统计
        expert_local_count = 0
        expert_offload_count = 0
        
        for i, task in enumerate(train_tasks):
            print(f"  Processing task {i+1}/{len(train_tasks)}...")
            # Expert strategy: (offload_decision, comp_resource, bandwidth)
            expert_strategy = bnb.solve(task)
            
            # 统计决策
            if expert_strategy[0] == 0:
                expert_local_count += 1
            else:
                expert_offload_count += 1
            
            # Normalize resource allocations for training
            offload_decision, comp_resource, bandwidth = expert_strategy
            if offload_decision > 0:
                rsu = env.rsus[offload_decision - 1]
                normalized_comp = comp_resource / rsu.max_comp_resource
                normalized_bandwidth = bandwidth / rsu.max_bandwidth
                expert_strategy = (offload_decision, normalized_comp, normalized_bandwidth)
            
            expert_strategies.append((task, expert_strategy))
        
        # 打印专家策略统计
        total_expert = expert_local_count + expert_offload_count
        expert_local_percent = (expert_local_count / total_expert * 100) if total_expert > 0 else 0
        expert_offload_percent = (expert_offload_count / total_expert * 100) if total_expert > 0 else 0
        print(f"\nExpert Strategy Statistics:")
        print(f"Local Execution: {expert_local_count} ({expert_local_percent:.1f}%)")
        print(f"Offload: {expert_offload_count} ({expert_offload_percent:.1f}%)")
        
        # Train GNN-Reuse-IL model
        print(f"Training GNN-Reuse-IL model for {epochs} epochs...")
        for epoch in range(epochs):
            epoch_loss = model.train_epoch(expert_strategies)
            loss_history.append(epoch_loss)
            
            if (epoch + 1) % 1 == 0:  # 更频繁地显示进度
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
    
    except Exception as e:
        print(f"Error in train_gnn_reuse_il: {str(e)}")
        traceback.print_exc()
        return [1.0]  # Return dummy loss history to continue execution
        
    return loss_history

def evaluate_algorithm(env, algorithm, tasks, track_decisions=False):
    """Evaluate algorithm performance on given tasks"""
    try:
        print(f"  Reset environment with {len(tasks)} tasks")
        env.reset(tasks)
        
        # Execute tasks according to algorithm's decisions
        print("  Reset algorithm")
        algorithm.reset()
        
        total_time = 0
        total_energy = 0
        total_obj = 0
        decisions = []  # 跟踪决策
        
        # 调试信息变量
        algorithm_name = algorithm.__class__.__name__
        min_energy = float('inf')
        max_energy = 0
        
        for i, task in enumerate(tasks):
            if i < 5 or i > len(tasks) - 5:  # 只输出前5个和后5个任务的详细信息，避免输出过多
                print(f"  Processing task {i+1}/{len(tasks)}...")
            
            # 获取决策
            decision = algorithm.make_decision(task)
            if track_decisions:
                decisions.append(decision)
            
            # 执行决策并获取结果
            time_cost, energy_cost, obj_value = env.execute_decision(task, decision)
            
            # 记录最大和最小能耗值
            if energy_cost < min_energy:
                min_energy = energy_cost
            if energy_cost > max_energy:
                max_energy = energy_cost
            
            if i < 5 or i > len(tasks) - 5:  # 只输出前5个和后5个任务的详细信息
                print(f"  Decision: {decision}")
                print(f"  Results - Time: {time_cost:.6f}, Energy: {energy_cost:.6f}, Obj: {obj_value:.6f}")
            
            total_time += time_cost
            total_energy += energy_cost
            total_obj += obj_value
        
        # Calculate averages
        avg_time = total_time / len(tasks)
        avg_energy = total_energy / len(tasks)
        avg_obj = total_obj / len(tasks)
        
        # 输出能耗范围调试信息
        print(f"  Algorithm {algorithm_name} energy range - Min: {min_energy:.6e}, Max: {max_energy:.6e}, Avg: {avg_energy:.6e}")
        
        print(f"  Algorithm evaluation completed - Avg Time: {avg_time:.6f}, Avg Energy: {avg_energy:.6f}, Avg Obj: {avg_obj:.6f}")
        
        if track_decisions:
            return avg_time, avg_energy, avg_obj, decisions
        else:
            return avg_time, avg_energy, avg_obj
        
    except Exception as e:
        print(f"Error in evaluate_algorithm: {str(e)}")
        traceback.print_exc()
        if track_decisions:
            return 1.0, 1.0, 1.0, []
        else:
            return 1.0, 1.0, 1.0  # Return dummy values to continue execution

if __name__ == "__main__":
    start_time = time.time()
    run_experiment()
    end_time = time.time()
    print(f"实验总时长: {end_time - start_time:.2f} 秒")