# Drug-Target Interaction Prediction using GIN and CNN

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Results](#results)
- [Key Technologies and Libraries](#key-technologies-and-libraries)
- [Files in this Repository](#files-in-this-repository)
- [How to Run (Usage)](#how-to-run-usage)
- [Future Work](#future-work)
- [Contact](#contact)

---

## Project Overview
This project presents a deep learning-based system designed to predict drug–target interactions by estimating the binding affinity (K<sub>d</sub>) between drug molecules and proteins. We integrate a **Graph Isomorphism Network (GIN)** for molecular graph encoding and a **Convolutional Neural Network (CNN)** for protein sequence embedding. The model aims to accelerate early-stage drug discovery by identifying promising drug candidates more efficiently.

---

## Problem Statement
A major bottleneck in drug development is identifying how strongly a drug molecule binds to a target protein. The **dissociation constant (K<sub>d</sub>)** quantifies this binding; lower values indicate stronger interactions. Accurately predicting K<sub>d</sub> values allows researchers to eliminate weak binders early, reducing experimental costs and speeding up drug screening.

---

## Methodology

### 🔹 1. Graph Isomorphism Network (GIN)
- Converts drug SMILES into molecular graphs using RDKit.
- GIN is used to extract structural features due to its superior expressiveness over GCN/GAT.

### 🔹 2. Convolutional Neural Network (CNN)
- Encodes the protein sequences into meaningful embeddings.
- Captures local patterns and motifs in amino acid sequences.

### 🔹 3. Feature Fusion & Prediction
- Drug and protein embeddings are concatenated.
- Fully connected layers are used to predict binding affinity.

---

## Dataset
We used the **DAVIS dataset**, a benchmark dataset for drug-target binding prediction containing:
- Drug-protein pairs
- Binding affinities (K<sub>d</sub>) as regression targets

---

## Results

| Metric              | Value     |
|---------------------|-----------|
| Training Accuracy   | 88.8%     |
| Test Accuracy       | 87.8%     |
| RMSE (Test)         | *Add Here* |
| MAE (Test)          | *Add Here* |
| Pearson Correlation | *Add Here* |

> ⚠️ *Note: Accuracy is only meaningful if K<sub>d</sub> is binned into classes. Please include regression metrics if you're treating K<sub>d</sub> as continuous.*

---

## Key Technologies and Libraries

- **Python 3.x**
- **PyTorch**, **torch_geometric** – GIN implementation
- **RDKit** – SMILES parsing and molecule graph generation
- **PyTDC** – Dataset access and bioinformatics tools
- **NumPy**, **Pandas** – Data processing
- **Matplotlib**, **Seaborn** – Visualization
- **Scikit-learn** – Metrics and preprocessing

---

## Files in this Repository

- `DrugTargetInteraction.ipynb` – Main notebook with full training pipeline
- `document_17.pdf` – Report explaining architecture, results, and dataset
- `requirements.txt` – Python dependencies *(recommended to add if missing)*
- `data/` – Contains dataset files (optional)
- `models/` – Trained model weights (optional)

---

## How to Run (Usage)

### 1. Clone the Repository
```bash
git clone <https://github.com/xPREMy/drugtargetinteraction>
cd drug-target-interaction-prediction
