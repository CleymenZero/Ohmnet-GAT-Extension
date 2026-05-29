import argparse
import sys
from src.train import train_model
from src.eval import evaluate_model
from src.analyze_attention import analyze_attention_hub

def main():
    parser = argparse.ArgumentParser(description="Ohmnet GAT Extension - Modular Pipeline Runner")
    parser.add_argument("--train", action="store_true", help="Run the model training pipeline")
    parser.add_argument("--eval", action="store_true", help="Run the test-set evaluation pipeline")
    parser.add_argument("--attention", action="store_true", help="Run the attention hub analysis")
    parser.add_argument("--use-wandb", action="store_true", help="Enable WandB logging during training")
    
    args = parser.parse_args()
    
    # If no flags are provided, print help
    if not (args.train or args.eval or args.attention):
        parser.print_help()
        sys.exit(0)
        
    if args.train:
        print("==============================")
        print("  LAUNCHING TRAINING PIPELINE ")
        print("==============================")
        train_model(use_wandb=args.use_wandb)
        
    if args.eval:
        print("==============================")
        print("LAUNCHING EVALUATION PIPELINE ")
        print("==============================")
        evaluate_model()
        
    if args.attention:
        print("==============================")
        print("LAUNCHING ATTENTION ANALYSIS  ")
        print("==============================")
        analyze_attention_hub()

if __name__ == "__main__":
    main()
