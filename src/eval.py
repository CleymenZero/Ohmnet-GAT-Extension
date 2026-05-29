import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from src.config import config
from src.data_pipeline import prepare_pipeline, _filter_eval_message_edges
from src.models import LateFusionGAT
from src.loss import check_hierarchical_violations

def optimize_threshold(y_true, y_score):
    """Searches for the optimal classification threshold using validation data to avoid leakage."""
    best_thresh = 0.5
    best_f1 = 0
    for t in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(y_true, (y_score >= t).astype(int), average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh

def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data Pipeline
    pipeline = prepare_pipeline(config, device)
    
    val_loader = pipeline["val_loader"]
    test_loader = pipeline["test_loader"]
    train_edge_set = pipeline["train_edge_set"]
    global_tissue_address = pipeline["global_tissue_address"]
    hierarchical_pairs = pipeline["hierarchical_pairs"]
    leaf_node_indices = pipeline["leaf_node_indices"]
    tissue_nodes = pipeline["tissue_nodes"]
    
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
        # Fallback to look in root directory if not found in models/
        model_path = os.path.join(os.path.dirname(config.model_save_path), "../best_model.pt")
        
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print("Model loaded successfully!")
    else:
        raise FileNotFoundError(f"Weights file not found at {config.model_save_path} or root directory.")

    # 3. Optimize Threshold on Validation Set
    print(f"Collecting validation predictions...")
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = model(batch)
            val_preds.append(preds.cpu())
            val_labels.append(batch.edge_label.cpu())
    val_preds = torch.cat(val_preds, dim=0).numpy()
    val_labels = torch.cat(val_labels, dim=0).numpy()
    
    best_threshold = optimize_threshold(val_labels[:, leaf_node_indices], val_preds[:, leaf_node_indices])
    print(f"Optimal Threshold found on validation set: {best_threshold:.2f}")

    # 4. Predict on Test Set
    print("Collecting test predictions...")
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch)
            test_preds.append(preds.cpu())
            test_labels.append(batch.edge_label.cpu())
    test_preds = torch.cat(test_preds, dim=0).numpy()
    test_labels = torch.cat(test_labels, dim=0).numpy()

    # 5. Compute Metrics for Leaf Tissues
    leaf_auroc, leaf_auprc, leaf_f1_scores = [], [], []
    tissue_weights = []
    valid_tissues = 0
    
    for j in leaf_node_indices:
        y_true = test_labels[:, j]
        y_pred = test_preds[:, j]
        
        if y_true.sum() > 0 and (1.0 - y_true).sum() > 0:
            leaf_auroc.append(roc_auc_score(y_true, y_pred))
            leaf_auprc.append(average_precision_score(y_true, y_pred))
            leaf_f1_scores.append(f1_score(y_true, (y_pred >= best_threshold).astype(int), zero_division=0))
            tissue_weights.append(y_true.sum())
            valid_tissues += 1

    tissue_weights = np.array(tissue_weights)
    tissue_weights_norm = tissue_weights / tissue_weights.sum()

    print(f"\n--- Test Set Evaluation Results ({valid_tissues} leaf tissues) ---")
    print(f"Macro-AUROC: {np.mean(leaf_auroc):.4f}")
    print(f"Macro-AUPRC: {np.mean(leaf_auprc):.4f}")
    print(f"Macro-F1    : {np.mean(leaf_f1_scores):.4f} (at threshold {best_threshold:.2f})")
    print(f"\n--- Prevalence-Weighted Results ---")
    print(f"Weighted-AUROC: {np.average(leaf_auroc, weights=tissue_weights_norm):.4f}")
    print(f"Weighted-AUPRC: {np.average(leaf_auprc, weights=tissue_weights_norm):.4f}")
    print(f"Weighted-F1   : {np.average(leaf_f1_scores, weights=tissue_weights_norm):.4f}")

    # 6. Global Confusion Matrix
    print("\nGenerating Confusion Matrix for Leaf Tissues...")
    flat_true = test_labels[:, leaf_node_indices].flatten()
    flat_pred = (test_preds[:, leaf_node_indices] >= best_threshold).astype(int).flatten()
    
    cm = confusion_matrix(flat_true, flat_pred)
    tn, fp, fn, tp = cm.ravel()
    
    micro_f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"True Positives:  {tp:,}")
    print(f"False Positives: {fp:,} (Type I Error)")
    print(f"False Negatives: {fn:,} (Type II Error)")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"Micro-F1:        {micro_f1:.4f}")

    # Export confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Inactive', 'Active'])
    disp.plot(cmap='Blues', values_format='d', ax=ax)
    plt.title(f'Global Confusion Matrix (Threshold: {best_threshold:.2f})')
    plt.grid(False)
    
    output_path = config.data_dir.parent / "outputs" / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to {output_path}!")
    plt.close()

    # 7. Check Hierarchical Violations on Test Set
    rate, count, total = check_hierarchical_violations(model, test_loader, hierarchical_pairs.to(device), device)
    print(f"\n--- Hierarchical Consistency (Test Set) ---")
    print(f"Violation Rate: {rate:.2%}")
    print(f"Violations:     {count:,} / {total:,}")

if __name__ == "__main__":
    evaluate_model()
