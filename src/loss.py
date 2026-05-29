import torch
import torch.nn.functional as F

def custom_loss(predictions, targets, hierarchy_edges, lambda_penalty=200.0, margin=0.01):
    """Calculates custom loss combining binary cross-entropy with a hierarchical consistency penalty.
    
    Penalizes instances where a child tissue's predicted activation is greater than its parent's activation.
    """
    bce = F.binary_cross_entropy(predictions, targets)
    
    child_idx, parent_idx = hierarchy_edges[:, 0], hierarchy_edges[:, 1]
    
    child_probs = predictions[:, child_idx]
    parent_probs = predictions[:, parent_idx]
    
    # We penalize if child_prob + margin > parent_prob
    # This forces the parent to be slightly higher than the child
    diff = (child_probs - parent_probs) + margin
    penalty = torch.mean(torch.clamp(diff, min=0)**2)
    
    return bce + lambda_penalty * penalty, bce, penalty

def check_hierarchical_violations(model, data_loader, hierarchical_pairs, device):
    """Evaluates hierarchical violation rate across predictions from a given DataLoader."""
    model.eval()
    total_violations = 0
    total_checks = 0
    
    child_indices = hierarchical_pairs[:, 0]
    parent_indices = hierarchical_pairs[:, 1]
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            preds = model(batch) 
            
            child_probs = preds[:, child_indices]
            parent_probs = preds[:, parent_indices]
            
            # Count cases where child exceeds parent
            violations = (child_probs > parent_probs).sum().item()
            total_violations += violations
            total_checks += (preds.size(0) * hierarchical_pairs.size(0))
            
    if total_checks == 0:
        return 0.0, 0, 0
    return total_violations / total_checks, total_violations, total_checks
