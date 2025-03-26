import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models.gnn_base import GATLayer, GNNBase

class GNNReuseIL(GNNBase):
    """
    GNN-Reuse-IL: Graph Neural Network with Computation Reuse and Imitation Learning
    """
    
    def __init__(self, env, hidden_dim=64, num_heads=4, lr=0.001):
        super(GNNReuseIL, self).__init__(env, hidden_dim)
        
        self.num_heads = num_heads
        
        # GNN layers with attention
        self.gat1 = GATLayer(self.input_dim, hidden_dim, num_heads)
        self.gat2 = GATLayer(hidden_dim * num_heads, hidden_dim, 1)
        
        # Decision heads
        self.offload_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(env.rsus) + 1)  # +1 for local execution
        )
        
        self.resource_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Resource allocation between 0 and 1
        )
        
        self.bandwidth_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Bandwidth allocation between 0 and 1
        )
        
        # Optimization
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Cache for computation reuse
        self.cache = {}
        
        # Load expert model if available
        self.expert_model = None
        try:
            self.expert_model = torch.load('models/expert_model.pth')
            self.expert_model.eval()
        except:
            print("No expert model found. Will train from scratch.")
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass of the GNN-Reuse-IL model
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]
        
        Returns:
            offload_decision: Probability distribution over offloading options
            resource_alloc: Resource allocation prediction
            bandwidth_alloc: Bandwidth allocation prediction
        """
        # 检查边索引是否为空
        if edge_index.shape[1] == 0:
            # 如果没有边，创建自环边
            num_nodes = x.shape[0]
            self_loops = torch.arange(num_nodes, device=x.device)
            edge_index = torch.stack([self_loops, self_loops], dim=0)
        
        # First GAT layer
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        
        # Second GAT layer
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling for graph-level prediction
        if batch is not None:
            x = self.global_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Decision heads
        offload_decision = self.offload_head(x)
        resource_alloc = self.resource_head(x)
        bandwidth_alloc = self.bandwidth_head(x)
        
        return offload_decision, resource_alloc, bandwidth_alloc
    
    def make_decision(self, task):
        """
        Make offloading and resource allocation decision for a task
        
        Args:
            task: DAG task to be executed
        
        Returns:
            Tuple of (offload_decision, comp_resource, bandwidth)
        """
        # First check if task can be reused from cache
        if self._check_reuse_opportunity(task):
            # Return decision from cache
            cached_decision = self._get_cached_decision(task)
            return cached_decision
        
        # Convert task to graph data
        x, edge_index = self._task_to_graph(task)
        
        # Forward pass
        with torch.no_grad():
            offload_logits, resource_pred, bandwidth_pred = self.forward(x, edge_index)
            
            # Convert logits to decisions
            offload_decision = torch.argmax(offload_logits, dim=1).item()  # 0 for local, 1+ for RSU
            comp_resource = resource_pred.item()  # Between 0 and 1
            bandwidth = bandwidth_pred.item()  # Between 0 and 1
            
            # Scale resource and bandwidth values
            if offload_decision > 0:
                rsu = self.env.rsus[offload_decision - 1]
                comp_resource = comp_resource * rsu.max_comp_resource
                bandwidth = bandwidth * rsu.max_bandwidth
        
        # Cache the decision
        self._cache_decision(task, (offload_decision, comp_resource, bandwidth))
        
        return offload_decision, comp_resource, bandwidth
    
    def train_epoch(self, expert_data):
        """
        Train the model for one epoch using expert data
        
        Args:
            expert_data: List of (task, expert_decision) tuples
        
        Returns:
            Average loss for the epoch
        """
        try:
            self.train()
            epoch_loss = 0.0
            
            for task, expert_decision in expert_data:
                try:
                    # Convert task to graph data
                    x, edge_index = self._task_to_graph(task)
                    
                    # 确定设备
                    device = next(self.parameters()).device
                    x = x.to(device)
                    edge_index = edge_index.to(device)
                    
                    # Expert decisions
                    expert_offload, expert_resource, expert_bandwidth = expert_decision
                    
                    # Convert to tensors with error handling
                    try:
                        expert_offload_tensor = torch.tensor([expert_offload], dtype=torch.long, device=device)
                        expert_resource_tensor = torch.tensor([[float(expert_resource)]], dtype=torch.float32, device=device)
                        expert_bandwidth_tensor = torch.tensor([[float(expert_bandwidth)]], dtype=torch.float32, device=device)
                    except Exception as e:
                        print(f"张量转换错误: {str(e)}, 使用默认值")
                        expert_offload_tensor = torch.tensor([0], dtype=torch.long, device=device)
                        expert_resource_tensor = torch.tensor([[0.5]], dtype=torch.float32, device=device)
                        expert_bandwidth_tensor = torch.tensor([[0.5]], dtype=torch.float32, device=device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    offload_logits, resource_pred, bandwidth_pred = self.forward(x, edge_index)
                    
                    # Calculate losses
                    offload_loss = F.cross_entropy(offload_logits, expert_offload_tensor)
                    resource_loss = F.mse_loss(resource_pred, expert_resource_tensor)
                    bandwidth_loss = F.mse_loss(bandwidth_pred, expert_bandwidth_tensor)
                    
                    # Combine losses
                    loss = offload_loss + resource_loss + bandwidth_loss
                    
                    # Backward and optimize
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                except Exception as e:
                    print(f"训练单个样本出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if len(expert_data) > 0:
                return epoch_loss / len(expert_data)
            else:
                return 0.0
                
        except Exception as e:
            print(f"整个训练轮次出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1.0  # 返回一个默认损失值
    
    def reset(self):
        """Reset the model for a new episode"""
        self.cache = {}
    
    def _check_reuse_opportunity(self, task):
        """Check if a task can be reused from cache"""
        if not hasattr(task, 'subtasks'):
            return False
        
        for subtask in task.subtasks:
            subtask_features = subtask.get_feature_vector()
            subtask_key = str(subtask_features)
            
            if subtask_key in self.cache:
                return True
        
        return False
    
    def _get_cached_decision(self, task):
        """Get decision from cache for a task"""
        # This is a simplified implementation - in practice, would combine cached results
        # Get the most frequently accessed cached decision
        if not hasattr(task, 'subtasks'):
            return 0, 0, 0  # Default values
        
        cached_decisions = []
        for subtask in task.subtasks:
            subtask_features = subtask.get_feature_vector()
            subtask_key = str(subtask_features)
            
            if subtask_key in self.cache:
                cached_decisions.append(self.cache[subtask_key])
        
        if not cached_decisions:
            return 0, 0, 0  # Default values
        
        # Return the most common decision
        return max(cached_decisions, key=cached_decisions.count)
    
    def _cache_decision(self, task, decision):
        """Cache decision for a task"""
        if not hasattr(task, 'subtasks'):
            return
        
        for subtask in task.subtasks:
            subtask_features = subtask.get_feature_vector()
            subtask_key = str(subtask_features)
            self.cache[subtask_key] = decision
    
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