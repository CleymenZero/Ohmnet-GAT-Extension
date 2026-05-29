import os
import json
import torch
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from pronto import Ontology
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import Node2Vec
from sklearn.model_selection import train_test_split

def build_tissue_hierarchy(hierarchy_path):
    """Builds a Directed Acyclic Graph representing the tissue hierarchy."""
    print(f"Building tissue hierarchy from {hierarchy_path}...")
    G_tissue = nx.read_edgelist(str(hierarchy_path), create_using=nx.DiGraph())
    
    tissue_nodes = sorted(list(G_tissue.nodes()))
    tissue_to_idx = {node: i for i, node in enumerate(tissue_nodes)}
    
    edges_tissue = []
    for u, v in G_tissue.edges():
        edges_tissue.append([tissue_to_idx[u], tissue_to_idx[v]])
        
    edges_index_tissue = torch.tensor(edges_tissue, dtype=torch.long).t().contiguous()
    hierarchical_pairs = edges_index_tissue.t()
    
    return G_tissue, tissue_nodes, tissue_to_idx, edges_index_tissue, hierarchical_pairs

def get_tissue_mapping(obo_path, tissue_to_idx):
    """Parses BrendaTissue.obo and returns mapping helper function."""
    print("Building mapping from BrendaTissue.obo...")
    name_to_bto = {}
    with open(obo_path, 'r', encoding='utf-8') as f:
        curr_id = None
        for line in f:
            line = line.strip()
            if line.startswith('id: '):
                curr_id = line[4:]
            elif line.startswith('name: ') and curr_id:
                name_to_bto[line[6:].lower()] = curr_id

    manual_rescue = {
        'culture condition cd8 cell': 'BTO:0004410',
        'b lymphocyte': 'BTO:0000776',
        'b lymphocytes': 'BTO:0000776', 
        't lymphocyte': 'BTO:0000782', 
        't lymphocytes': 'BTO:0000782'  
    }

    def get_tissue_idx(t_name):
        normalized = str(t_name).lower().replace('_', ' ')
        if normalized in manual_rescue:
            bto_id = manual_rescue[normalized]
        else:
            bto_id = name_to_bto.get(normalized)
            
        if not bto_id:
            return None
        
        if bto_id in tissue_to_idx:
            return tissue_to_idx[bto_id]
        if bto_id + '_NODE' in tissue_to_idx:
            return tissue_to_idx[bto_id + '_NODE']
        return None

    return get_tissue_idx

def load_and_preprocess_ppi(ppi_path, tissue_mapping_func):
    """Loads and maps the PPI edgelist to tissue indices."""
    print(f"Loading PPI file from {ppi_path}...")
    df = pd.read_csv(ppi_path, sep='\t', names=['protein_a', 'protein_b', 'tissue'])
    
    df['protein_a'] = df['protein_a'].astype(str)
    df['protein_b'] = df['protein_b'].astype(str)
    
    df['t_idx'] = df['tissue'].apply(tissue_mapping_func)
    df = df.dropna(subset=['t_idx'])
    df['t_idx'] = df['t_idx'].astype(int) 
    
    all_proteins = sorted(pd.concat([df['protein_a'], df['protein_b']]).unique())
    protein_to_idx = {prot: i for i, prot in enumerate(all_proteins)}
    
    return df, protein_to_idx

def propagate_labels_true_path(labels_matrix, G_tissue, tissue_nodes, tissue_to_idx):
    """Applies True Path Rule to propagate labels from child nodes to all ancestor nodes."""
    print("Propagating labels through hierarchy (True Path Rule)...")
    new_labels_matrix = labels_matrix.copy()
    
    # Precalculate ancestor indices
    ancestor_map = {}
    for idx, node in enumerate(tissue_nodes):
        ancestors = nx.descendants(G_tissue, node)
        ancestor_map[idx] = [tissue_to_idx[anc] for anc in ancestors if anc in tissue_to_idx]
        
    # Apply propagation
    for i in range(labels_matrix.shape[0]):
        active_indices = np.where(labels_matrix[i] == 1)[0]
        for tidx in active_indices:
            for anc_idx in ancestor_map[tidx]:
                new_labels_matrix[i, anc_idx] = 1.0
                
    edge_label_all = torch.tensor(new_labels_matrix, dtype=torch.float)
    return new_labels_matrix, edge_label_all

def get_tissue_embeddings(edges_index_tissue, device, cache_path=None):
    """Trains a Node2Vec model to generate tissue representations, or loads from cache if available."""
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached Node2Vec tissue embeddings from {cache_path}...")
        return torch.load(cache_path, map_location=device)
        
    print("Training Node2Vec model for tissue hierarchy embeddings...")
    tissue_n2v_model = Node2Vec(edges_index_tissue, embedding_dim=256, walk_length=10, 
                                context_size=10, walks_per_node=10, num_negative_samples=1, 
                                sparse=True).to(device)
    
    torch.nn.init.xavier_uniform_(tissue_n2v_model.embedding.weight)
    n2v_loader = tissue_n2v_model.loader(batch_size=32, shuffle=True, num_workers=0) 
    n2v_optimizer = torch.optim.SparseAdam(list(tissue_n2v_model.parameters()), lr=0.001)
    
    tissue_n2v_model.train()
    for epoch in range(1, 1001):
        total_loss = 0
        for pos_rw, neg_rw in n2v_loader:
            n2v_optimizer.zero_grad()
            loss = tissue_n2v_model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            n2v_optimizer.step()
            total_loss += loss.item()
        if epoch % 100 == 0:
            print(f"Node2Vec Epoch {epoch}/1000, Loss: {total_loss/len(n2v_loader):.4f}")
            
    global_tissue_address = tissue_n2v_model().detach()
    
    if cache_path:
        print(f"Caching tissue embeddings to {cache_path}...")
        torch.save(global_tissue_address, cache_path)
        
    return global_tissue_address

def _filter_eval_message_edges(batch, train_edge_set):
    """Filters target edges from evaluation message passing to prevent transductive leakage."""
    target_edges = set()
    for i in range(batch.edge_label_index.size(1)):
        u, v = batch.edge_label_index[0, i].item(), batch.edge_label_index[1, i].item()
        target_edges.add((u, v))
        target_edges.add((v, u))
        
    edges_to_remove = target_edges - train_edge_set
    if len(edges_to_remove) == 0:
        return batch
        
    keep_mask = torch.ones(batch.edge_index.size(1), dtype=torch.bool)
    for i in range(batch.edge_index.size(1)):
        e = (batch.edge_index[0, i].item(), batch.edge_index[1, i].item())
        if e in edges_to_remove:
            keep_mask[i] = False
    batch.edge_index = batch.edge_index[:, keep_mask]
    return batch

def prepare_pipeline(config, device):
    """Runs data loading, label propagation, embedding generation, and returns all loaders."""
    G_tissue, tissue_nodes, tissue_to_idx, edges_index_tissue, hierarchical_pairs = build_tissue_hierarchy(config.hierarchy_path)
    
    get_tissue_idx = get_tissue_mapping(config.obo_path, tissue_to_idx)
    
    df, protein_to_idx = load_and_preprocess_ppi(config.ppi_path, get_tissue_idx)
    num_unique_proteins = len(protein_to_idx)
    
    # Aggregating interactions and collapsing to multi-hot matrix
    interaction_groups = df.groupby(['protein_a', 'protein_b'])
    num_tissues = len(tissue_nodes)
    num_interactions = len(interaction_groups)
    
    labels_matrix = np.zeros((num_interactions, num_tissues))
    edge_list_all = []
    
    for i, ((p1, p2), group) in enumerate(interaction_groups):
        u = protein_to_idx[p1]
        v = protein_to_idx[p2]
        edge_list_all.append([u, v])
        
        tissues_present_indices = group['t_idx'].unique()
        for tidx in tissues_present_indices:
            labels_matrix[i, int(tidx)] = 1.0
            
    edge_label_index_all = torch.tensor(edge_list_all, dtype=torch.long).t().contiguous()
    
    # Enforce True Path Rule label propagation
    labels_matrix, edge_label_all = propagate_labels_true_path(
        labels_matrix, G_tissue, tissue_nodes, tissue_to_idx
    )
    
    # Setup data splits (72/8/20 split)
    indices = np.arange(num_interactions)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=config.seed, shuffle=True)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1, random_state=config.seed, shuffle=True)
    
    # Transductive leakage correction
    train_edge_index = edge_label_index_all[:, train_idx]
    undirected_train_edge_index = to_undirected(train_edge_index)
    
    train_edge_set = set()
    for i in range(undirected_train_edge_index.size(1)):
        u, v = undirected_train_edge_index[0, i].item(), undirected_train_edge_index[1, i].item()
        train_edge_set.add((u, v))
        
    full_graph_data = Data(x=torch.arange(num_unique_proteins), edge_index=undirected_train_edge_index)
    
    g = torch.Generator()
    g.manual_seed(config.seed)
    
    # neighbor loader configurations
    train_loader = LinkNeighborLoader(
        full_graph_data,
        num_neighbors=[15, 10], 
        batch_size=config.batch_size,
        edge_label_index=edge_label_index_all[:, train_idx],
        edge_label=edge_label_all[train_idx],
        shuffle=True,
        generator=g
    )
    
    val_loader = LinkNeighborLoader(
        full_graph_data,
        num_neighbors=[15, 10],
        batch_size=config.batch_size,
        edge_label_index=edge_label_index_all[:, val_idx],
        edge_label=edge_label_all[val_idx],
        shuffle=False
    )
    
    test_loader = LinkNeighborLoader(
        full_graph_data,
        num_neighbors=[15, 10],
        batch_size=config.batch_size,
        edge_label_index=edge_label_index_all[:, test_idx],
        edge_label=edge_label_all[test_idx],
        shuffle=False
    )
    
    # Train/load tissue hierarchy embeddings
    cache_path = os.path.join(config.data_dir, "tissue_embeddings.pt")
    global_tissue_address = get_tissue_embeddings(edges_index_tissue, device, cache_path=cache_path)
    
    leaf_node_indices = [i for i, node in enumerate(tissue_nodes) if G_tissue.in_degree(node) == 0 and node != 'Root_NODE']
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_edge_set": train_edge_set,
        "full_graph_data": full_graph_data,
        "global_tissue_address": global_tissue_address,
        "hierarchical_pairs": hierarchical_pairs,
        "num_proteins": num_unique_proteins,
        "num_tissues": num_tissues,
        "leaf_node_indices": leaf_node_indices,
        "protein_to_idx": protein_to_idx,
        "df": df,
        "tissue_nodes": tissue_nodes
    }
