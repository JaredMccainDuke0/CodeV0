import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """Graph Attention Layer"""
    
    def __init__(self, in_dim, out_dim, num_heads=1):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        # Linear transformation for node features
        self.W = nn.Linear(in_dim, out_dim * num_heads)
        
        # Attention parameters
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_dim)))
        nn.init.xavier_uniform_(self.a.data)
        
        # Edge feature transformation
        self.W_edge = nn.Linear(1, out_dim)
        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(0.2)
    
    def forward(self, x, edge_index):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            h_prime: Updated node features [num_nodes, out_dim*num_heads]
        """
        # Linear transformation of node features
        Wh = self.W(x)  # [num_nodes, num_heads * out_dim]
        Wh = Wh.view(-1, self.num_heads, self.out_dim)  # [num_nodes, num_heads, out_dim]
        
        # Initialize attention output
        h_prime = torch.zeros_like(Wh)  # [num_nodes, num_heads, out_dim]
        
        num_nodes = x.shape[0]
        
        # Extract source and target nodes
        src, dst = edge_index
        
        # Create edge features (assume all edges have weight 1)
        edge_attr = torch.ones(edge_index.shape[1], 1, device=x.device)
        
        # Transform edge features
        edge_features = self.W_edge(edge_attr)
        
        for head in range(self.num_heads):
            # Compute attention coefficients
            src_features = Wh[src, head]  # [num_edges, out_dim]
            dst_features = Wh[dst, head]  # [num_edges, out_dim]
            
            # Concatenate source and destination features
            edge_features_cat = torch.cat([src_features, dst_features], dim=1)  # [num_edges, 2*out_dim]
            
            # Compute attention scores
            e = self.leakyrelu(torch.matmul(edge_features_cat, self.a.t()))  # [num_edges, 1]
            
            # 对每个节点处理其邻居节点
            for i in range(num_nodes):
                # 获取当前节点的所有出边索引
                edge_indices = (src == i).nonzero(as_tuple=True)[0]
                
                if len(edge_indices) > 0:
                    # 获取邻居节点
                    neighbors = dst[edge_indices]
                    
                    # 获取对应的注意力分数
                    attention_scores = e[edge_indices].squeeze(-1)
                    
                    # 应用softmax获取注意力系数
                    attention_coefs = F.softmax(attention_scores, dim=0)
                    
                    # 计算加权邻居特征
                    weighted_features = torch.zeros_like(Wh[i, head])
                    for j, neighbor_idx in enumerate(neighbors):
                        weighted_features += attention_coefs[j] * Wh[neighbor_idx, head]
                    
                    h_prime[i, head] = weighted_features
                else:
                    # 如果没有邻居，保持原始特征
                    h_prime[i, head] = Wh[i, head]
        
        # Reshape back to [num_nodes, out_dim * num_heads]
        return h_prime.view(-1, self.num_heads * self.out_dim)


class GNNBase(nn.Module):
    """Base class for GNN models"""
    
    def __init__(self, env, hidden_dim):
        super(GNNBase, self).__init__()
        
        self.env = env
        self.hidden_dim = hidden_dim
        
        # Define input dimension based on feature vector size
        # Assuming the feature vector includes:
        # - Task computation complexity
        # - Task data size
        # - Task priority
        self.input_dim = 3
    
    def global_pool(self, x, batch):
        """
        Global pooling operation
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment for nodes [num_nodes]
        
        Returns:
            graph_features: Graph-level features [num_graphs, hidden_dim]
        """
        if batch is None:
            # If no batch information, assume single graph
            return torch.mean(x, dim=0, keepdim=True)
        
        # Mean pooling for each graph in the batch
        num_graphs = batch.max().item() + 1
        graph_features = torch.zeros(num_graphs, x.shape[1])
        
        for i in range(num_graphs):
            mask = (batch == i)
            graph_features[i] = torch.mean(x[mask], dim=0)
        
        return graph_features