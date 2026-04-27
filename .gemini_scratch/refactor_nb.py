import json
import sys

def split_source(code_string):
    lines = code_string.split('\n')
    return [line + '\n' if i < len(lines)-1 else line for i, line in enumerate(lines)]

def modify_notebook(nb_path):
    print(f"Reading {nb_path}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    cells = nb['cells']
    
    # 1. Functionize Data Preprocessing (PPI Loading)
    ppi_loading_code = """import pandas as pd

def load_and_preprocess_ppi(ppi_file_path, tissue_mapping_func):
    print(f"Loading PPI file from {ppi_file_path}...")
    df = pd.read_csv(ppi_file_path, sep='\\t', names=['protein_a', 'protein_b', 'tissue'])
    
    print("Applying tissue mapping...")
    df['t_idx'] = df['tissue'].apply(tissue_mapping_func)
    df = df.dropna(subset=['t_idx'])
    df['t_idx'] = df['t_idx'].astype(int) 
    
    all_proteins = sorted(pd.concat([df['protein_a'], df['protein_b']]).unique())
    protein_to_idx = {prot: i for i, prot in enumerate(all_proteins)}
    
    num_ppis = len(df)
    num_unique_proteins = len(all_proteins)
    
    print("\\n--- PPI Processing Summary ---")
    print(f"Total PPIs present:      {num_ppis:,}")
    print(f"Total unique proteins:   {num_unique_proteins:,}")
    print("------------------------------\\n")
    
    return df, protein_to_idx

ppi_file = data_dir / 'PPT-Ohmnet_tissues-combined.edgelist'
df, protein_to_idx = load_and_preprocess_ppi(ppi_file, get_tissue_idx)
num_unique_proteins = len(protein_to_idx)
"""
    for i, c in enumerate(cells):
        src = "".join(c['source']) if isinstance(c['source'], list) else c['source']
        if c['cell_type'] == 'code' and "ppi_file = data_dir / 'PPT-Ohmnet_tissues-combined.edgelist'" in src and "load_and_preprocess_ppi" not in src:
            cells[i]['source'] = split_source(ppi_loading_code)
            print("Replaced PPI Loading cell.")
            break

    # 2. Refactor Tissue Hierarchy
    tissue_hierarchy_code = """import networkx as nx

def build_tissue_hierarchy(hierarchy_path):
    G_tissue = nx.read_edgelist(str(hierarchy_path), create_using=nx.DiGraph())
    print(f"The tissue graph has {G_tissue.number_of_nodes()} nodes.")
    
    tissue_nodes = sorted(list(G_tissue.nodes()))
    tissue_to_idx = {node: i for i, node in enumerate(tissue_nodes)}
    
    edges_tissue = []
    for u, v in G_tissue.edges():
        edges_tissue.append([tissue_to_idx[u], tissue_to_idx[v]])
        
    edges_index_tissue = torch.tensor(edges_tissue, dtype=torch.long).t().contiguous()
    hierarchical_pairs = edges_index_tissue.t()
    
    print(f"Tissue edge index shape processed: {edges_index_tissue.shape} ")
    return G_tissue, tissue_nodes, tissue_to_idx, edges_index_tissue, hierarchical_pairs

tissue_hierarchy_path = data_dir / 'tissue.hierarchy'
G_tissue, tissue_nodes, tissue_to_idx, edges_index_tissue, hierarchical_pairs = build_tissue_hierarchy(tissue_hierarchy_path)
"""
    for i, c in enumerate(cells):
        src = "".join(c['source']) if isinstance(c['source'], list) else c['source']
        if c['cell_type'] == 'code' and "G_tissue = nx.read_edgelist" in src and "build_tissue_hierarchy" not in src:
            cells[i]['source'] = split_source(tissue_hierarchy_code)
            print("Replaced Tissue Hierarchy cell.")
            break

    # 3. Label Propagation Refactoring
    label_prop_code = """def propagate_labels_true_path(labels_matrix, G_tissue, tissue_nodes, tissue_to_idx, leaf_node_indices):
    print("Propagating labels through hierarchy...")
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
    
    tissues_with_labels_orig = (labels_matrix.sum(axis=0) > 0).sum()
    tissues_with_labels_prop = (new_labels_matrix.sum(axis=0) > 0).sum()
    
    print(f"--- Global Density (All 219 Nodes) ---")
    print(f"Original label density: {labels_matrix.mean():.4%}")
    print(f"Propagated label density: {new_labels_matrix.mean():.4%}")
    print(f"\\n--- Tissue Coverage ---")
    print(f"Tissues represented in raw PPI: {tissues_with_labels_orig} / {len(tissue_nodes)}")
    print(f"Tissues active after Propagation: {tissues_with_labels_prop} / {len(tissue_nodes)}")
    
    leaf_density_prop = new_labels_matrix[:, leaf_node_indices].mean()
    print(f"\\nPropagated leaf node density: {leaf_density_prop:.4%}")
    print("\\nLabel propagation complete. Re-splitting data is required to update loaders.")
    
    return new_labels_matrix, edge_label_all

new_labels_matrix, edge_label_all = propagate_labels_true_path(
    labels_matrix, G_tissue, tissue_nodes, tissue_to_idx, leaf_node_indices
)
"""
    for i, c in enumerate(cells):
        src = "".join(c['source']) if isinstance(c['source'], list) else c['source']
        if c['cell_type'] == 'code' and "new_labels_matrix = labels_matrix.copy()" in src and "propagate_labels_true_path" not in src:
            cells[i]['source'] = split_source(label_prop_code)
            print("Replaced Label Propagation cell.")
            break

    # 4. Extract and Move Ground Truth Consistency Check
    gt_check_idx = -1
    for i, c in enumerate(cells):
        src = "".join(c['source']) if isinstance(c['source'], list) else c['source']
        if c['cell_type'] == 'code' and "CRITICAL CHECK: Is the ground truth hierarchical?" in src:
            gt_check_idx = i
            break
            
    if gt_check_idx != -1:
        gt_cell = cells.pop(gt_check_idx)
        print("Found GT Check cell. Popped it from the bottom.")
        insert_idx = -1
        for i, c in enumerate(cells):
            src = "".join(c['source']) if isinstance(c['source'], list) else c['source']
            if c['cell_type'] == 'code' and "propagate_labels_true_path" in src:
                insert_idx = i
                break
        if insert_idx != -1:
            md_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": split_source("### Hierarchical Consistency Check (Pre-Propagation)\\nLet's verify why label propagation is necessary by checking the ground truth labels.")
            }
            cells.insert(insert_idx, gt_cell)
            cells.insert(insert_idx, md_cell)
            print("Inserted GT Check cell before label propagation.")

    # 5. Optimize Training Loop & GPU clear
    train_loop_code = """import gc

# Clear Memory before large model allocation
torch.cuda.empty_cache()
gc.collect()

# ADAPTIVE TARGET TRAINING LOOP (Tracked with Weights & Biases)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
num_epochs = config.num_epochs
patience = config.patience
best_val_loss = float('inf')
epochs_no_improve = 0
current_lambda = config.initial_lambda

history = {
    'train': [], 'val': [], 
    'train_bce': [], 'train_penalty': [],
    'val_bce': [], 'val_penalty': [],
    'val_violation_rate': [],
    'lambda_history': []
}

if 'hierarchical_pairs' not in globals():
    hierarchical_pairs = edges_index_tissue.t()
hier_pairs_dev = hierarchical_pairs.to(device)

print(f"Starting Adaptive Training (Target: <{config.violation_threshold:.0%}, Initial Lambda: {current_lambda})...")

for epoch in range(num_epochs):
    model.train()
    total_train_loss, total_train_bce, total_train_pen = 0, 0, 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        preds = model(batch)
        
        loss, bce, penalty = custom_loss(preds, batch.edge_label, hier_pairs_dev, lambda_penalty=current_lambda)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()
        
        total_train_loss += loss.item()
        total_train_bce += bce.item()
        total_train_pen += penalty.item()
        
    model.eval()
    total_val_loss, total_val_bce, total_val_pen = 0, 0, 0
    val_violations, val_checks = 0, 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = _filter_eval_message_edges(batch, _train_edge_set)
            batch = batch.to(device)
            preds = model(batch)
            
            loss, bce, penalty = custom_loss(preds, batch.edge_label, hier_pairs_dev, lambda_penalty=current_lambda)
            total_val_loss += loss.item()
            total_val_bce += bce.item()
            total_val_pen += penalty.item()
            
            # Integrated Violation Check
            if (epoch + 1) % 5 == 0 or epoch == 0:
                child_probs = preds[:, hier_pairs_dev[:, 0]]
                parent_probs = preds[:, hier_pairs_dev[:, 1]]
                val_violations += (child_probs > parent_probs).sum().item()
                val_checks += (preds.size(0) * hier_pairs_dev.size(0))
    
    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    history['train'].append(avg_train_loss)
    history['val'].append(avg_val_loss)
    history['train_bce'].append(total_train_bce / len(train_loader))
    history['val_bce'].append(total_val_bce / len(val_loader))
    history['train_penalty'].append(total_train_pen / len(train_loader))
    history['val_penalty'].append(total_val_pen / len(val_loader))
    
    v_rate = None
    if val_checks > 0:
        v_rate = val_violations / val_checks
        history['val_violation_rate'].append(v_rate)
        
        if v_rate > config.violation_threshold:
            current_lambda = min(current_lambda * config.lambda_multiplier, config.max_lambda)
            print(f"--- Violation Threshold Exceeded ({v_rate:.2%})! New Lambda: {current_lambda:.2f} ---")
        else:
            print(f"--- Hierarchy Stable ({v_rate:.2%}). Lambda Fixed. ---")
            
        print(f"Epoch {epoch+1:03d} | BCE: {total_train_bce/len(train_loader):.4f} | Pen: {total_train_pen/len(train_loader):.6f}")
        print(f"Losses | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

    wandb_metrics = {
        "epoch": epoch + 1,
        "train/loss": avg_train_loss,
        "val/loss": avg_val_loss,
        "train/bce": total_train_bce / len(train_loader),
        "val/bce": total_val_bce / len(val_loader),
        "train/penalty": total_train_pen / len(train_loader),
        "val/penalty": total_val_pen / len(val_loader),
        "params/lambda": current_lambda,
        "params/patience_counter": epochs_no_improve
    }
    if v_rate is not None:
        wandb_metrics["val/hierarchy_violation_rate"] = v_rate
    
    wandb.log(wandb_metrics)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pt')
        wandb.save('best_model.pt') 
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}!")
        break
"""
    for i, c in enumerate(cells):
        src = "".join(c['source']) if isinstance(c['source'], list) else c['source']
        if c['cell_type'] == 'code' and "ADAPTIVE TARGET TRAINING LOOP" in src and "gc.collect()" not in src:
            cells[i]['source'] = split_source(train_loop_code)
            print("Replaced Training Loop cell.")
            break

    # 6. Consolidate Hub Protein cells
    hub_cell_idx_1 = -1
    hub_cell_idx_2 = -1
    hub_cell_idx_3 = -1
    
    for i, c in enumerate(cells):
        src = "".join(c['source']) if isinstance(c['source'], list) else c['source']
        if c['cell_type'] == 'code':
            if "Identify the hub protein (the one with the most interactions in column 'protein_a') - EXTRA" in src:
                hub_cell_idx_1 = i
            elif "Identify the hub protein - EXTRA" in src and "Top 10 Attention Weights for Protein" in src:
                hub_cell_idx_2 = i
            elif "Identify the hub protein - EXTRA" in src and "Total Unique Neighbors found in Train Graph" in src:
                hub_cell_idx_3 = i

    consolidated_hub_code = """# EXTRA: Hub Protein Analysis & Attention Visualization
def analyze_hub_protein(model, df, protein_to_idx, full_graph_data, device):
    hub_protein = df['protein_a'].value_counts().idxmax()
    target_idx = protein_to_idx[hub_protein]
    idx_to_protein = {v: k for k, v in protein_to_idx.items()}
    
    # 1. Neighbor Statistics
    unique_pairs = df[['protein_a', 'protein_b']].drop_duplicates()
    neighbors_from_a = unique_pairs[unique_pairs['protein_a'] == hub_protein]['protein_b']
    neighbors_from_b = unique_pairs[unique_pairs['protein_b'] == hub_protein]['protein_a']
    total_neighbors = pd.concat([neighbors_from_a, neighbors_from_b]).unique()
    print(f"Hub Protein {hub_protein} has {len(total_neighbors)} unique actual neighbors.")

    # 2. Attention Analysis
    model.eval()
    with torch.no_grad():
        x_emb = model.protein_embedding(full_graph_data.x.to(device))
        _, (edge_idx, alpha) = model.gat1(x_emb, full_graph_data.edge_index.to(device), return_attention_weights=True)
        
        alpha = alpha.mean(dim=-1).cpu().numpy()
        edge_idx = edge_idx.cpu().numpy()
        
        # We look at edges where the hub is the receiver (index 1) since we want to see who it attends to
        mask = (edge_idx[1] == target_idx)
        neighbor_indices = edge_idx[0, mask] 
        neighbor_weights = alpha[mask]
        
        sorted_sort_idx = np.argsort(neighbor_weights)[::-1]
        
        print(f"\\nTop 10 Neighbors Attended to by {hub_protein}")
        print("-" * 45)
        print(f"{'Neighbor Protein':<20} | {'Attention Weight':<15}")
        print("-" * 45)
        
        for i in range(min(10, len(sorted_sort_idx))):
            idx = sorted_sort_idx[i]
            neighbor_id = idx_to_protein[neighbor_indices[idx]]
            weight = neighbor_weights[idx]
            print(f"{neighbor_id:<20} | {weight:.4f}")
            
    return hub_protein
    
hub_protein_name = analyze_hub_protein(model, df, protein_to_idx, full_graph_data, device)
simple_plot_attn(hub_protein_name)
"""
    to_delete = sorted([i for i in [hub_cell_idx_1, hub_cell_idx_2, hub_cell_idx_3] if i != -1], reverse=True)
    if to_delete:
        insert_pos = to_delete[-1]
        for idx in to_delete:
            cells.pop(idx)
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": split_source(consolidated_hub_code)
        }
        cells.insert(insert_pos, new_cell)
        print(f"Consolidated Hub cells at indices {to_delete} into 1 cell.")

    nb['cells'] = cells
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook modifications saved.")

if __name__ == '__main__':
    modify_notebook('c:/Users/steve/Documents/ML Workspace/Ohmnet GAT Extension/COMP6841 - Capstone Project - SR.ipynb')
