import os
import gc
import torch
import shutil
from src.config import config, set_seed
from src.data_pipeline import prepare_pipeline, _filter_eval_message_edges
from src.models import LateFusionGAT
from src.loss import custom_loss

def train_model(use_wandb=False):
    # Enforce seed for training reproducibility
    set_seed(config.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data Pipeline
    pipeline = prepare_pipeline(config, device)
    
    train_loader = pipeline["train_loader"]
    val_loader = pipeline["val_loader"]
    train_edge_set = pipeline["train_edge_set"]
    global_tissue_address = pipeline["global_tissue_address"]
    hierarchical_pairs = pipeline["hierarchical_pairs"]
    num_proteins = pipeline["num_proteins"]
    num_tissues = pipeline["num_tissues"]
    
    # Update config dynamic attributes
    config.num_proteins = num_proteins
    config.num_tissues = num_tissues
    config.tissue_address_dim = global_tissue_address.shape[1]
    
    print("\nInitializing model LateFusionGAT...")
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

    # 2. WandB Setup
    if use_wandb:
        try:
            import wandb
            from dotenv import load_dotenv
            load_dotenv()
            wandb.login()
            wandb.init(
                entity="steven-rav-concordia-university",
                project="Ohmnet GAT extension",
                config=config.__dict__
            )
            print("WandB successfully initialized!")
        except Exception as e:
            print(f"Warning: Failed to initialize WandB. Proceeding without it. Error: {e}")
            use_wandb = False
            
    # Clear CUDA memory before allocation
    torch.cuda.empty_cache()
    gc.collect()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    patience = config.patience
    best_val_loss = float('inf')
    epochs_no_improve = 0
    current_lambda = config.initial_lambda
    
    hier_pairs_dev = hierarchical_pairs.to(device)
    print(f"Starting Adaptive Training (Initial Lambda: {current_lambda})...")
    
    for epoch in range(config.num_epochs):
        # --- TRAINING PHASE ---
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
            
        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss, total_val_bce, total_val_pen = 0, 0, 0
        val_violations, val_checks = 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = _filter_eval_message_edges(batch, train_edge_set)
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
        avg_train_bce = total_train_bce / len(train_loader)
        avg_val_bce = total_val_bce / len(val_loader)
        avg_train_pen = total_train_pen / len(train_loader)
        avg_val_pen = total_val_pen / len(val_loader)
        
        v_rate = None
        if val_checks > 0:
            v_rate = val_violations / val_checks
            if v_rate > config.violation_threshold:
                current_lambda = min(current_lambda * config.lambda_multiplier, config.max_lambda)
                print(f"Violation threshold exceeded ({v_rate:.2%})! New Lambda: {current_lambda:.2f}")
            else:
                print(f"Hierarchy stable ({v_rate:.2%}). Lambda fixed.")
                
            print(f"Epoch {epoch+1:03d} | Train BCE: {avg_train_bce:.4f} | Train Pen: {avg_train_pen:.6f}")
            print(f"Losses | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        # Log metrics
        if use_wandb:
            import wandb
            metrics = {
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "train/bce": avg_train_bce,
                "val/bce": avg_val_bce,
                "train/penalty": avg_train_pen,
                "val/penalty": avg_val_pen,
                "params/lambda": current_lambda,
                "params/patience_counter": epochs_no_improve
            }
            if v_rate is not None:
                metrics["val/hierarchy_violation_rate"] = v_rate
            wandb.log(metrics)

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), config.model_save_path)
            print(f"New top checkpoint saved to {config.model_save_path}!")
            if use_wandb and wandb.run is not None:
                shutil.copy(config.model_save_path, os.path.join(wandb.run.dir, 'best_model.pt')) 
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            break
            
    # Extract final attention weights
    print("\nExtracting final attention coefficients...")
    model.eval()
    test_loader = pipeline["test_loader"]
    with torch.no_grad():
        all_att_edges = []
        all_att_weights = []
        for batch in test_loader:
            batch = batch.to(device)
            _, (att_edge_index, att_weights) = model(batch, return_attention_weights=True)
            all_att_edges.append(att_edge_index.cpu())
            all_att_weights.append(att_weights.cpu())
        
        attention_data = {
            "edge_indices": all_att_edges,
            "attention_weights": all_att_weights
        }
        torch.save(attention_data, config.attention_save_path)
        print(f"Attention coefficients saved to {config.attention_save_path}!")
        if use_wandb and wandb.run is not None:
            shutil.copy(config.attention_save_path, os.path.join(wandb.run.dir, 'final_attention_coefficients.pt'))
            
    if use_wandb:
        import wandb
        wandb.finish()

if __name__ == "__main__":
    train_model(use_wandb=False)
