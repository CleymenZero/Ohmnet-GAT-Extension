# Late Fusion Graph Attention Network (GAT) for Tissue-Specific Function Prediction

## Overview
This project targets the prediction of multicellular functions by mapping protein-protein interaction (PPI) networks across human anatomical hierarchies. It represents a **deep supervised extension** of the **OhmNet** study (Zitnik and Leskovec, 2017), replacing unsupervised random-walk features with a modern **Graph Attention Network (GAT)** architecture.

The model is designed to leverage both local interaction network structure and global anatomical context (BRENDA Tissue Ontology) to predict tissue-specific interactions with high biological fidelity.

## Key Features
- **Hierarchical GNN Architecture:** Uses a `LateFusionGAT` model combining multi-head attention (16 heads) with a multi-layer perceptron (MLP).
- **Ontological Context:** Integrates `node2vec` embeddings of the **BRENDA Tissue Ontology (BTO)** directly into the feature fusion step.
- **Hierarchical Constrained Loss:** Implements a custom loss function that penalizes logical violations, ensuring the model respects anatomical logic (e.g., $P(\text{Heart}) \ge P(\text{Left Ventricle})$).
- **Benchmark Performance:** Specifically optimized to exceed the **0.756 ROCAUC target** on 107 leaf-level tissues.

## Project Structure
```text
├── COMPR6841 - Capstone Project - SR.ipynb  # Main experiment notebook
├── best_model.pt                             # Best performing model weights
├── node2vec_model.pth                         # Learned tissue hierarchy embeddings
├── checkpoints/                              # Periodic model saves
└── data/
    ├── tissue.hierarchy                      # The anatomical tree structure
    ├── BrendaTissue.obo                      # Official BTO documentation
    └── PPT-Ohmnet_tissues-combined.edgelist # Multi-tissue PPI dataset
```

## Methodology
1. **Tissue Embedding:** The tissue hierarchy is processed as a directed graph. `Node2Vec` is used to learn recursive structural features for each anatomical node.
2. **Network Sampling:** Uses `LinkNeighborLoader` from PyTorch Geometric for spatial neighborhood sampling, allowing the model to process massive PPI networks in efficient batches (1024).
3. **Late Fusion:** For any protein pair $(A, B)$, the GAT contextualizes the protein features, which are then fused with the "address" of the target tissue hierarchy's root node.
4. **Supervised Training:** The model is trained over 1000 epochs with **Early Stopping** (patience=50) to optimize for both accuracy and hierarchical consistency.

## Results
The model demonstrates **high-precision ranking capabilities**:
- **Specific Tissues:** Achieved ~88% Precision for localized interactions (e.g., Cardiac-specific pairs).
- **Ubiquitous Proteins:** Demonstrated "Graceful Degradation," maintaining >75% precision even for widespread housekeeping proteins by assigning "soft" lower-confidence probabilities to related secondary tissues.
- **Benchmark:** Successfully exceeded the performance threshold for leaf tissue reconstruction tasks.

## Model Interpretability
Unlike previous unsupervised approaches (OhmNet), this architecture provides a biologically interpretable mechanism via **Graph Attention Weights**. 

A validation study on a major network hub (**Protein 4914**, with 786 neighbors) revealed the following:
- **Functional Specificity:** The top 5 attention neighbors (e.g., Neighbors 401, 57611, 51062) all share the **`nervous_system`** tissue tag with the hub.
- **Attention Selection:** The model assigned ~36% of its total attention to just the top two neighbors, successfully filtering out ~780 noisy connections.
- **Role Differentiation:**
    - **Specialist Neighbors:** (e.g., 57611, 51062) helped the model pinpoint niche tissues.
    - **Generalist Neighbors:** (e.g., 10156, 27352) helped reinforce broader tissue groups like `blood` and `hematopoietic`.
- **Conclusion:** This confirms the GAT mechanism is successfully learning the **spatial co-occurrence** of proteins, "inheriting" functional information from biologically consistent neighbors.

## Requirements
- Python 3.x
- PyTorch / PyTorch Geometric
- NetworkX
- Pandas / NumPy
- Matplotlib
- Pronto (for OBO parsing)

## Usage
1. Ensure the `data/` directory contains the necessary edge lists and ontology files.
2. Run the `Node2Vec` training cell to generate `node2vec_model.pth`.
3. Execute the `LateFusionGAT` training loop within the Jupyter Notebook.
4. Evaluate the model using the final metrics cell to compare against the leaf-tissue benchmark.

---
*Created as part of the COMP6841 Capstone Project.*
