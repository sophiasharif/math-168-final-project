# MATH-168 Final Project: AML Replication and MEGA-GNN

This repository contains reproducible experiments for anti-money laundering (AML) transaction classification on the IBM AMLworld synthetic datasets, including notebook-based baselines and a code implementation of MEGA-GNN.

## Repository Links

- Code repository: [https://github.com/sophiasharif/math-168-final-project](https://github.com/sophiasharif/math-168-final-project)
- Primary AML data source (IBM AMLworld datasets): [https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)

## Problem Setup and Notation

We model financial transactions as a directed multigraph.

- **Graph:** G = (V, E), where V is the set of accounts and E is the multiset of directed transaction edges.
- **Edge:** e_k = (u_k, v_k, t_k, z_k, y_k) for transaction k:
  - u_k in V: source account
  - v_k in V: destination account
  - t_k in R: timestamp
  - z_k: raw edge feature vector (amount, currency, payment format, etc.)
  - y_k in {0,1}: laundering label (1 = illicit, 0 = licit)
- **Edge index order:** transactions are sorted by timestamp for temporal training and feature extraction.
- **Splits:** E_train, E_val, E_test with a 60/20/20 temporal split.
- **Node features:** x_i for node i, usually simple structural placeholders or engineered statistics.
- **Edge embedding:** h_e_k^(0) is the encoded edge feature before message passing.
- **Node embedding:** h_i^(l) is node embedding at layer l.
- **Prediction target:** edge-level probability p_hat_k = P(y_k = 1 | G, e_k).
- **Decision threshold:** tau, with predicted class y_hat_k = 1[p_hat_k >= tau].
- **Primary metric:** minority-class F1 score, F1 = 2PR / (P + R), where P is precision and R is recall on class y = 1.

### Motif Features (when enabled)

For motif-enhanced experiments, each edge may include motif support features m_k = [m_k,1, ..., m_k,d_m], computed in a streaming, past-only manner from prior graph history (no future leakage).

## Repository Organization

Top-level files and folders:

- `paper_replication.ipynb`: paper-style GFP+XGBoost replication attempt on HI-Small.
- `gnn_baseline_replication.ipynb`: GNN baseline replication (PNA-like and GIN+EU-like).
- `mega_gnn_replication.ipynb`: compact notebook implementation of MEGA-style training.
- `model.ipynb`: motif-GNN style experimental notebook.
- `MEGA-GNN/`: full script-based MEGA-GNN codebase (training, inference, data loading, model definitions).
- `data/`: local AML files used by notebooks (CSV files).
- `paper.pdf`: AMLworld dataset/benchmark paper.
- `2412.00241v2.pdf`: MEGA-GNN paper.

Important files inside `MEGA-GNN/`:

- `main.py`: main entrypoint for train/inference.
- `util.py`: CLI args and utilities.
- `models.py`: model architectures and message passing modules.
- `data_loading.py`: dataset ingestion and split logic.
- `training.py`, `training_eth.py`: training loops.
- `inference.py`: inference pipeline.
- `data_config.json`: filesystem paths for datasets/checkpoints.

## Reproducibility Instructions

### 1) Data Preparation

1. Download AML files from Kaggle (link above), especially the HI/LI transaction CSVs.
2. Place files in `data/` for notebook workflows, or format data for `MEGA-GNN/` script workflows as required by `format_kaggle_files.py`.
3. For Colab in `MEGA-GNN`, set paths in `MEGA-GNN/data_config.json` to your Drive folders.

### 2) Notebook Replication (quick path)

Run these notebooks in order (or independently):

1. `paper_replication.ipynb`
2. `gnn_baseline_replication.ipynb`
3. `model.ipynb`
4. `mega_gnn_replication.ipynb`

Each notebook writes JSON metrics to an `artifacts/` folder under the project root.

### Colab Note: Training `MEGA-GNN/` via Notebook

If you are using Google Colab, the easiest way to train the script-based `MEGA-GNN/` pipeline is the root notebook `MEGA_GNN.ipynb`. That notebook is intended to mount Drive, set `MEGA-GNN/data_config.json` paths, install dependencies, and launch training commands from the `MEGA-GNN/` folder in one Colab workflow. In short: use `MEGA_GNN.ipynb` as the Colab runner for the `MEGA-GNN/` codebase.

### 3) Script-Based MEGA-GNN Replication

From `MEGA-GNN/`:

1. Create environment and install dependencies (see `MEGA-GNN/README.md`).
2. Confirm `data_config.json` paths.
3. Run training, for example:

```bash
python main.py --data Small_HI --model pna --emlps --reverse_mp --ego --flatten_edges --edge_agg_type pna --n_epochs 80 --save_model --task edge_class
```

To run the motif-aware edge encoder path (if motif columns are present in input edge features), include:

```bash
--motif_encoder --motif_dim 8
```

where `motif_dim` is the number of motif features appended at the end of each edge feature vector.

## Exact Replication Notes

- Temporal order is preserved in both splitting and streaming feature extraction.
- Minority-class weighting is used because class imbalance is extreme.
- Results may vary slightly across hardware/runtime unless random seeds, software versions, and preprocessing are exactly matched.
- If reproducing paper table values, use the same dataset variant (HI/LI, small/medium/large), model family, and threshold protocol.

## Expected Outputs

- Training logs in terminal / logger outputs.
- JSON metrics in `artifacts/` (notebook runs) and model outputs/checkpoints from script runs.
- F1/precision/recall/PR-AUC values for minority class.

