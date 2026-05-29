import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.inits import glorot

class LateFusionGAT(nn.Module):
    def __init__(self, num_proteins, protein_embedding_dim, num_heads, gat_hidden_channels, 
                 gat_output_channels, tissue_address_dim, mlp_hidden_channels, 
                 num_tissues, global_tissue_address_tensor, dropout_rate=0.3):
        super().__init__()
        self.num_tissues = num_tissues
        
        # Load static pre-computed tissue address embeddings (no-grad Parameter)
        self.global_tissue_address = nn.Parameter(global_tissue_address_tensor, requires_grad=False)
        self.protein_embedding = nn.Embedding(num_proteins, protein_embedding_dim)
        
        # GAT Layer 1 + skip
        self.gat1 = GATv2Conv(protein_embedding_dim, gat_hidden_channels, heads=num_heads, dropout=dropout_rate)
        self.skip1 = nn.Linear(protein_embedding_dim, gat_hidden_channels * num_heads)
        
        # GAT Layer 2 + skip
        self.gat2 = GATv2Conv(gat_hidden_channels * num_heads, gat_hidden_channels, heads=num_heads, concat=True, dropout=dropout_rate)
        self.skip2 = nn.Linear(gat_hidden_channels * num_heads, gat_hidden_channels * num_heads)
        
        # GAT Layer 3 (Final features, concat=False to average heads)
        self.gat3 = GATv2Conv(gat_hidden_channels * num_heads, gat_output_channels, heads=1, concat=False, dropout=dropout_rate)
        
        # MLP Fusion head with concatenated protein features and tissue address features
        combined_dim = gat_output_channels * 2 + tissue_address_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, mlp_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_channels, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.protein_embedding.weight)
        self.gat1.reset_parameters()
        self.gat2.reset_parameters()
        self.gat3.reset_parameters()
        glorot(self.skip1.weight)
        glorot(self.skip2.weight)
        
        # Initialize MLP weights using Glorot (Xavier) uniform
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                glorot(layer.weight)

    def forward(self, batch, return_attention_weights=False):
        edge_index = batch.edge_index
        x_in = self.protein_embedding(batch.x)
        
        # GAT 1 + Skip
        x = self.gat1(x_in, edge_index)
        x = F.elu(x + self.skip1(x_in))
        
        # GAT 2 + Skip
        x_prev = x
        x = self.gat2(x, edge_index)
        x = F.elu(x + self.skip2(x_prev))
        
        # GAT 3 (Optionally return attention weights)
        if return_attention_weights:
            x, (att_edge_index, att_weights) = self.gat3(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat3(x, edge_index)
        
        # Late Fusion Logic
        u_idx, v_idx = batch.edge_label_index[0], batch.edge_label_index[1]
        u_feat, v_feat = x[u_idx], x[v_idx] 
        
        batch_size = u_feat.size(0)
        
        # Expand dimensions to calculate prediction for each pair across all 219 tissues
        u_feat_exp = u_feat.unsqueeze(1).expand(-1, self.num_tissues, -1)
        v_feat_exp = v_feat.unsqueeze(1).expand(-1, self.num_tissues, -1)
        tissue_feat = self.global_tissue_address.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate features
        combined = torch.cat([u_feat_exp, v_feat_exp, tissue_feat], dim=-1)
        
        # Forward through the MLP
        out = self.mlp(combined).squeeze(-1)
        logits = torch.sigmoid(out)
        
        if return_attention_weights:
            return logits, (att_edge_index, att_weights)
        return logits
