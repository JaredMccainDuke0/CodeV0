import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from models.gnn_base import GATLayer, GNNBase
import traceback

class GNNDRL(GNNBase):
    """
    GNN-DRL: Graph Neural Network with Deep Reinforcement Learning
    """
    
    def __init__(self, env, hidden_dim=64, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lr=0.001, buffer_size=10000, is_target=False):
        super(GNNDRL, self).__init__(env, hidden_dim)
        
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.is_target = is_target
        
        # GNN layers
        self.gnn1 = nn.Linear(self.input_dim, hidden_dim)
        self.gnn2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Q-network heads
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(env.rsus) + 1)  # +1 for local execution
        )
        
        self.resource_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),  # Discretized resource levels (0.2, 0.4, 0.6, 0.8, 1.0)
            nn.Softmax(dim=1)
        )
        
        self.bandwidth_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),  # Discretized bandwidth levels (0.2, 0.4, 0.6, 0.8, 1.0)
            nn.Softmax(dim=1)
        )
        
        # Target network (for stability)
        self.target_network = None
        if not is_target:
            # 简化目标网络初始化
            print("创建目标网络...")
            self.target_network = GNNDRL(self.env, self.hidden_dim, is_target=True)
            # 此处不主动更新，避免递归引用
        
        # Optimization - 只有非目标网络需要优化器和经验回放
        if not is_target:
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
            self.replay_buffer = deque(maxlen=buffer_size)
            self.batch_size = 32
            self.training_step = 0
        else:
            self.optimizer = None
            self.replay_buffer = None
            self.batch_size = 0
            self.training_step = 0
    
    def create_target_network(self):
        """创建目标网络并同步权重"""
        if self.target_network is None and not self.is_target:
            self.target_network = GNNDRL(self.env, self.hidden_dim, is_target=True)
            self.update_target_network()
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass of the GNN-DRL model
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]
        
        Returns:
            q_values: Q-values for offloading options
            resource_values: Resource allocation levels probabilities
            bandwidth_values: Bandwidth allocation levels probabilities
        """
        # First GNN layer
        x = self.gnn1(x)
        x = F.relu(x)
        
        # Message passing along edges
        if edge_index.shape[1] > 0:  # If there are edges
            source, target = edge_index
            for _ in range(2):  # Two rounds of message passing
                # Aggregate messages from neighbors
                messages = torch.zeros_like(x)
                for i in range(edge_index.shape[1]):
                    src, tgt = source[i], target[i]
                    messages[tgt] += x[src]
                
                # Update node features
                x = x + 0.1 * messages  # Simple aggregation
                x = F.relu(x)
        
        # Second GNN layer
        x = self.gnn2(x)
        x = F.relu(x)
        
        # Global pooling for graph-level prediction
        if batch is not None:
            x = self.global_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Decision heads
        q_values = self.q_head(x)
        resource_values = self.resource_head(x)
        bandwidth_values = self.bandwidth_head(x)
        
        return q_values, resource_values, bandwidth_values
    
    def make_decision(self, task):
        """
        Make offloading and resource allocation decision for a task
        
        Args:
            task: DAG task to be executed
        
        Returns:
            Tuple of (offload_decision, comp_resource, bandwidth)
        """
        # Convert task to graph data
        x, edge_index = self._task_to_graph(task)
        
        # Get current state
        state = self.env.get_state(task, task.vehicle_id)
        
        # Epsilon-greedy exploration
        if np.random.rand() <= self.epsilon:
            # Random action
            offload_decision = np.random.randint(0, len(self.env.rsus) + 1)
            resource_level = np.random.randint(0, 5)
            bandwidth_level = np.random.randint(0, 5)
            
            # Convert to actual values
            comp_resource = (resource_level + 1) * 0.2
            bandwidth = (bandwidth_level + 1) * 0.2
            
            if offload_decision > 0:
                rsu = self.env.rsus[offload_decision - 1]
                comp_resource = comp_resource * rsu.max_comp_resource
                bandwidth = bandwidth * rsu.max_bandwidth
        else:
            # Forward pass
            with torch.no_grad():
                q_values, resource_probs, bandwidth_probs = self.forward(x, edge_index)
                
                # Convert to decisions
                offload_decision = torch.argmax(q_values, dim=1).item()
                resource_level = torch.argmax(resource_probs, dim=1).item()
                bandwidth_level = torch.argmax(bandwidth_probs, dim=1).item()
                
                # Convert to actual values
                comp_resource = (resource_level + 1) * 0.2
                bandwidth = (bandwidth_level + 1) * 0.2
                
                if offload_decision > 0:
                    rsu = self.env.rsus[offload_decision - 1]
                    comp_resource = comp_resource * rsu.max_comp_resource
                    bandwidth = bandwidth * rsu.max_bandwidth
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return offload_decision, comp_resource, bandwidth
    
    def update_target_network(self):
        """Update target network weights with current network weights"""
        if self.target_network is not None and not self.is_target:
            try:
                # 获取当前网络的状态字典并过滤掉target_network相关的参数
                state_dict = self.state_dict()
                filtered_state_dict = {}
                
                for k, v in state_dict.items():
                    if not k.startswith('target_network.'):
                        filtered_state_dict[k] = v
                
                self.target_network.load_state_dict(filtered_state_dict)
                print("目标网络权重更新成功")
            except Exception as e:
                print(f"目标网络更新失败: {str(e)}")
                traceback.print_exc()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def perform_train_step(self):
        """Perform one step of training using experience replay"""
        if self.is_target:
            print("目标网络不进行训练")
            return 0.0
            
        if self.replay_buffer is None or len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # 确保目标网络存在
        if self.target_network is None:
            self.create_target_network()
            
        try:
            # Sample mini-batch from replay buffer
            minibatch = random.sample(self.replay_buffer, self.batch_size)
            
            loss = 0.0
            for state, action, reward, next_state, done in minibatch:
                # Convert state/next_state to graph data
                x, edge_index = self._state_to_graph(state)
                
                # 确保device一致性
                device = next(self.parameters()).device
                x = x.to(device)
                edge_index = edge_index.to(device)
                
                # Current Q-values
                q_values, resource_probs, bandwidth_probs = self.forward(x, edge_index)
                target_q_values = q_values.clone().detach()
                
                # 获取next_state的q值（如果next_state不为None）
                if next_state is not None:
                    next_x, next_edge_index = self._state_to_graph(next_state)
                    next_x = next_x.to(device)
                    next_edge_index = next_edge_index.to(device)
                    
                    # Target Q-values
                    with torch.no_grad():
                        next_q_values, _, _ = self.target_network.forward(next_x, next_edge_index)
                        max_next_q = torch.max(next_q_values).item()
                        
                        # Calculate target
                        if done:
                            target = reward
                        else:
                            target = reward + self.gamma * max_next_q
                else:
                    # 如果没有next_state，直接使用reward作为目标
                    target = reward
                
                # 获取action的各个组件
                offload_action, resource_action, bandwidth_action = action
                
                # 更新对应动作的Q值
                target_q_values[0, offload_action] = target
                
                # 计算损失
                q_loss = F.mse_loss(q_values, target_q_values)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                q_loss.backward()
                self.optimizer.step()
                
                loss += q_loss.item()
            
            # 更新目标网络
            self.training_step += 1
            if self.training_step % 10 == 0:  # 更频繁地更新目标网络
                self.update_target_network()
            
            return loss / len(minibatch)
            
        except Exception as e:
            print(f"训练步骤出错: {str(e)}")
            traceback.print_exc()
            return 0.0
    
    def reset(self):
        """Reset the model for a new episode"""
        self.epsilon = 1.0  # Reset exploration rate
    
    def _task_to_graph(self, task):
        """Convert DAG task to graph data for GNN"""
        if not hasattr(task, 'subtasks'):
            # Single task without DAG structure
            node_features = [task.get_feature_vector()]
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            
            return torch.tensor(node_features, dtype=torch.float32), edge_index
        
        # Extract node features
        node_features = []
        for subtask in task.subtasks:
            node_features.append(subtask.get_feature_vector())
        
        # Create edge index from dependencies
        edge_index = [[], []]
        for i, subtask in enumerate(task.subtasks):
            for dep in subtask.dependencies:
                # Add edge from dependency to current subtask
                edge_index[0].append(dep)
                edge_index[1].append(i)
        
        return torch.tensor(node_features, dtype=torch.float32), torch.tensor(edge_index, dtype=torch.long)
    
    def _state_to_graph(self, state):
        """Convert environment state to graph data for GNN"""
        try:
            # 如果state为None或不是字典，使用默认空状态
            if state is None or not isinstance(state, dict):
                # 创建一个默认状态
                default_features = torch.zeros((3, self.input_dim), dtype=torch.float32)  # 至少包含任务、车辆和一个RSU
                edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)  # 简单的连接关系
                return default_features, edge_index
            
            # 提取状态特征
            task_features = state.get('task_features', [0, 0, 0])  # 默认值
            vehicle_resources = state.get('vehicle_resources', [0])  # 默认值
            rsu_resources = state.get('rsu_resources', [[0, 0]] * max(1, len(self.env.rsus)))  # 默认值
            channel_conditions = state.get('channel_conditions', [0] * max(1, len(self.env.rsus)))  # 默认值
            
            # 确保所有数据格式一致
            if isinstance(task_features, torch.Tensor):
                task_features = task_features.cpu().tolist()
            if isinstance(vehicle_resources, torch.Tensor):
                vehicle_resources = vehicle_resources.cpu().tolist()
            
            # 创建节点特征
            # 节点0：任务节点
            # 节点1：车辆节点
            # 节点2+：RSU节点
            node_features = []
            
            # 任务节点特征 - [comp_complexity, data_size, priority]
            node_features.append(task_features)
            
            # 车辆节点特征 - 扩展到3维，与任务特征保持一致维度
            # [available_resource, 0, 0] - 补充两个0作为占位符
            vehicle_feature = vehicle_resources + [0, 0]
            if len(vehicle_feature) > 3:
                vehicle_feature = vehicle_feature[:3]  # 如果超长则截断
            elif len(vehicle_feature) < 3:
                # 确保长度为3
                vehicle_feature = vehicle_feature + [0] * (3 - len(vehicle_feature))
            node_features.append(vehicle_feature)
            
            # RSU节点特征（包含通道条件）- 也扩展到3维
            for i, rsu_res in enumerate(rsu_resources):
                if i < len(channel_conditions):
                    # [comp_resource, bandwidth, channel_condition]
                    rsu_feature = rsu_res + [channel_conditions[i]]
                else:
                    rsu_feature = rsu_res + [0]  # 默认通道条件
                
                # 确保RSU特征也是3维
                if len(rsu_feature) > 3:
                    rsu_feature = rsu_feature[:3]  # 如果超长则截断
                elif len(rsu_feature) < 3:
                    # 确保长度为3
                    rsu_feature = rsu_feature + [0] * (3 - len(rsu_feature))
                
                node_features.append(rsu_feature)
            
            # 创建边索引 - 将任务连接到车辆和所有RSU
            edge_index = [[], []]
            
            # 任务到车辆的边
            edge_index[0].append(0)
            edge_index[1].append(1)
            
            # 任务到RSU的边
            for i in range(len(rsu_resources)):
                edge_index[0].append(0)
                edge_index[1].append(i + 2)
            
            # 车辆到RSU的边
            for i in range(len(rsu_resources)):
                edge_index[0].append(1)
                edge_index[1].append(i + 2)
            
            return torch.tensor(node_features, dtype=torch.float32), torch.tensor(edge_index, dtype=torch.long)
        
        except Exception as e:
            print(f"Error in _state_to_graph: {str(e)}")
            # 返回简单默认图
            default_features = torch.zeros((3, self.input_dim), dtype=torch.float32)  # 至少包含任务、车辆和一个RSU
            edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)  # 简单的连接关系
            return default_features, edge_index