# Late Fusion Graph Attention Network (GAT) for Tissue-Specific Protein Function Prediction

This project explores the application of a **Late Fusion Graph Attention Network (GAT)** to predict tissue-specific functions in complex protein-protein interaction (PPI) networks. Built as an advanced extension of the **OhmNet** study, this architecture replaces unsupervised random-walk features with supervised multi-head attention to enhance both predictive accuracy and biological interpretability.

## 🚀 Key Features

- **Late Fusion GAT Architecture**: Implements the `LateFusionGAT` model, integrating multi-head attention (16 heads) with a multi-layer perceptron (MLP) for feature fusion and classification.
- **Hierarchical Ontological Context**: Leverages `node2vec` embeddings of the **BRENDA Tissue Ontology (BTO)** to provide a global anatomical context for predictions.
- **Efficient Data Handling**: Uses `LinkNeighborLoader` from PyTorch Geometric for scalable training on large-scale PPI graphs with batch-based neighborhood sampling.
- **Hierarchical Constrained Loss**: Features a custom loss function designed to preserve anatomical consistency across parent and child nodes in the tissue hierarchy.
- **SOTA Performance**: Optimized with specific hyperparameter tuning, maintaining an optimal dropout rate of 0.3 to maximize generalizability across 107 leaf tissues.

## 🏗️ Project Structure

- `COMP6841 - Capstone Project - SR.ipynb`: The primary research notebook featuring data preprocessing, model implementation, and final evaluation.
- `data/`: Contains raw PPI datasets, tissue hierarchy files (`tissue.hierarchy`), and BTO documentation (`BrendaTissue.obo`).
- `best_model.pt`: Checkpoint containing weights for the top-performing model (calculated using validation loss early stopping).
- `node2vec_model.pth`: Pre-trained embeddings for the tissue hierarchy.

## 🔬 Methodology

1. **Hierarchy Embedding**: The Brenda Tissue Ontology is modeled as a directed graph, and `Node2Vec` is applied to capture structural relationships between anatomical sites.
2. **Graph Sampling**: `LinkNeighborLoader` samples spatial neighborhoods, facilitating memory-efficient training on representative subgraphs.
3. **Late Fusion Mechanism**: Models protein pair interactions using the GAT's contextual encoding, subsequently fusing these features with the hierarchical tissue "address" before final prediction.
4. **Optimization**: Training utilizes **Early Stopping** (patience=50) and a learning rate of 0.001 with the Adam optimizer over up to 1000 epochs.

## 📊 Results

The model significantly exceeds industry benchmarks and the initial project goal of 0.756 AUROC.

- **Test Leaf AUROC**: `0.9759`
- **Macro-AUPRC**: `0.8974`
- **Macro-F1 Score**: `0.8440`

*Note: These results reflect the 0.3 dropout configuration, which showed consistent superiority over higher regularization rates (0.4/0.5).*

## 🧠 Model Interpretability (Attention Analysis)

The attention mechanism identifies functional clusters by prioritizing neighbors with consistent biological roles.

**Hub Analysis: Protein 1956**
A study on this highly connected protein revealed that the GAT captures functional relevance through weighted attention:
- **Top Neighbor**: Protein `3226` (Weight: `0.2933`)
- **Functional Alignment**: Higher weights were observed for biological neighbors sharing the same nervous system tissue tags, confirming the model's ability to filter noise and focus on critical functional pathways.

## 🛠️ Requirements

- Python 3.12+
- PyTorch & PyTorch Geometric
- NetworkX
- Pandas, NumPy, Scikit-learn
- Matplotlib

## 🏃 Usage

1. **Populate Data**: Ensure all PPI and BTO files are located in the `data/` directory.
2. **Execute Workflow**: Open `COMP6841 - Capstone Project - SR.ipynb` and run cells sequentially to:
   - Generate tissue embeddings.
   - Train the `LateFusionGAT` model.
   - Visualize attention weights for network hub proteins.
3. **Evaluation**: Use the provided `best_model.pt` to replicate the leaf-tissue metrics analysis.
