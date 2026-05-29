import os
import torch
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src.config import config
from src.data_pipeline import prepare_pipeline
from src.models import LateFusionGAT

def analyze_attention_hub():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data Pipeline
    pipeline = prepare_pipeline(config, device)
    df = pipeline["df"]
    protein_to_idx = pipeline["protein_to_idx"]
    full_graph_data = pipeline["full_graph_data"]
    global_tissue_address = pipeline["global_tissue_address"]
    
    config.num_proteins = pipeline["num_proteins"]
    config.num_tissues = pipeline["num_tissues"]
    config.tissue_address_dim = global_tissue_address.shape[1]
    
    # 2. Initialize and Load Model
    model = LateFusionGAT(
        num_proteins=config.num_proteins, 
        protein_embedding_dim=config.protein_embedding_dim,
        num_heads=config.num_heads, 
        gat_hidden_channels=config.gat_hidden_channels,
        gat_output_channels=config.gat_output_channels, 
        tissue_address_dim=config.tissue_address_dim,
        mlp_hidden_channels=config.mlp_hidden_channels, 
        num_tissues=config.num_tissues,
        global_tissue_address_tensor=global_tissue_address, 
        dropout_rate=config.dropout_rate
    ).to(device)
    
    model_path = config.model_save_path
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(config.model_save_path), "../best_model.pt")
        
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print("Model loaded successfully!")
    else:
        raise FileNotFoundError(f"Weights file not found at {config.model_save_path} or root directory.")

    # 3. Identify Hub Protein
    hub_protein = df['protein_a'].value_counts().idxmax()
    target_idx = protein_to_idx[hub_protein]
    idx_to_protein = {v: k for k, v in protein_to_idx.items()}
    
    unique_pairs = df[['protein_a', 'protein_b']].drop_duplicates()
    neighbors_from_a = unique_pairs[unique_pairs['protein_a'] == hub_protein]['protein_b']
    neighbors_from_b = unique_pairs[unique_pairs['protein_b'] == hub_protein]['protein_a']
    total_neighbors = pd.concat([neighbors_from_a, neighbors_from_b]).unique()
    print(f"\nHub Protein: {hub_protein}")
    print(f"It has {len(total_neighbors)} unique actual neighbors.")

    # 4. Extract Attention Coefficients from GAT1 Layer
    print("Extracting attention weights from GAT Layer 1...")
    with torch.no_grad():
        x_emb = model.protein_embedding(full_graph_data.x.to(device))
        _, (edge_idx, alpha) = model.gat1(x_emb, full_graph_data.edge_index.to(device), return_attention_weights=True)
        
        alpha = alpha.mean(dim=-1).cpu().numpy()
        edge_idx = edge_idx.cpu().numpy()
        
        # Filter for edges where the hub is the receiver (index 1) to inspect incoming attention
        mask_receiver = (edge_idx[1] == target_idx)
        neighbor_indices = edge_idx[0, mask_receiver] 
        neighbor_weights = alpha[mask_receiver]
        
        sorted_indices = np.argsort(neighbor_weights)[::-1]
        
        print(f"\nTop 10 Neighbors Attended to by Hub {hub_protein}:")
        print("-" * 45)
        print(f"{'Neighbor Protein':<20} | {'Attention Weight':<15}")
        print("-" * 45)
        for i in range(min(10, len(sorted_indices))):
            idx = sorted_indices[i]
            neighbor_id = idx_to_protein[neighbor_indices[idx]]
            weight = neighbor_weights[idx]
            print(f"{neighbor_id:<20} | {weight:.4f}")
        print("-" * 45)

        # 5. Generate and save the Spring Layout Network Plot for Hub Protein
        print(f"\nPlotting attention neighbors for {hub_protein}...")
        mask_source = (edge_idx[0] == target_idx)
        sub_edges = edge_idx[:, mask_source]
        sub_alpha = alpha[mask_source]
        
        sort_idx = np.argsort(sub_alpha)[::-1]
        top_idx = sort_idx[:10]  # Take top 10
        
        G = nx.Graph()
        for i in top_idx:
            u_name = idx_to_protein[sub_edges[0, i]]
            v_name = idx_to_protein[sub_edges[1, i]]
            G.add_edge(u_name, v_name, weight=sub_alpha[i])
            
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=config.seed)
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold')
        
        # Add labels for the weights on the edges
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.title(f"Top 10 Attention Neighbors for Protein {hub_protein}")
        
        output_path = config.data_dir.parent / "outputs" / "attention_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Attention plot successfully saved to {output_path}!")
        plt.close()

if __name__ == "__main__":
    analyze_attention_hub()
