<div align="center">

# TPTTPN: Tri-Path Spatio-Temporal Network with Tensor-Train Probabilistic Head for Traffic Flow Prediction 🚦📈

Lei Yang, Zecan Yang, Laurence T. Yang, Honglu Zhao, Song Zhang

A PyTorch framework for probabilistic spatio-temporal series forecasting on traffic data.

<br/>

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76b900?logo=nvidia)

</div>

---

## ✨ Features

- **🗺️ Rich Feature Embeddings**: Automatically incorporates embeddings for time-of-day, day-of-week, and adaptive spatial features.
- **Tri-Path Hybrid Architecture**: Fuses features from three parallel paths:
  1. Spatio-Temporal Transformers with cross-attention.
  2. Multi-Scale Convolutions for local context.
  3. Triple Graph Convolutions (Static Forward/Backward + Dynamic Graph).
- **🎯 Probabilistic Forecasting**: Outputs uncertainty estimations by modeling parameters for **Gaussian** or **Student's t** distributions.
- **🧩 Decoupled TT-Decomposed Head**: Utilizes a `Tensor-Train` decomposed output head (`OutputHead`) for efficient prediction generation.

## 📦 Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ylcmy/TPTTPN.git
    cd TPTTPN
    ```

2.  **Create and activate a Conda environment:**

    ```bash
    conda create -n tpttpn python=3.9
    conda activate tpttpn
    ```

3.  **Install dependencies using pip:**
    ```bash
    pip install -r requirements.txt
    ```
    _Minimum Requirements: Python 3.9+, PyTorch 1.12+ (GPU recommended)._

## 📂 Data Preparation

Download the required datasets (e.g., PeMS) from a source like [this repository](https://github.com/laiguokun/multivariate-time-series-data).

Place the dataset files under the `./data/` directory, following this structure:

```
./data/
├── PEMS04/
│   ├── PEMS04.npz
│   └── PEMS04.csv
└── PEMS08/
    ├── PEMS08.npz
    └── PEMS08.csv
```

Default configurations for common datasets (`PEMS03`, `PEMS04`, `PEMS07`, `PEMS08`) are already included in `config/dataset_config.py`.

## 🚀 Quick Start

### 🚆 Training

To train the `TPTTPN` model on the `PEMS08` dataset:

```bash
python run.py --mode train --model TPTTPN --dataset PEMS08
```

Training logs, model checkpoints, and result visualizations will be saved to the `experiments/` directory.

### 🧪 Testing

To test a trained model, the script can automatically find the latest experiment for a given dataset:

```bash
python run.py --mode test --model TPTTPN --dataset PEMS08
```

Alternatively, you can specify an experiment directory to load a specific checkpoint:

```bash
python run.py --mode test --model TPTTPN --dataset PEMS08 \
  --exp_dir experiments/PEMS08/20250601_120000
```

## ⚙️ Configuration & Parameters

Key arguments can be passed via the command line.

- **Execution Control**: `--mode [train|test]`, `--model TPTTPN`, `--dataset [PEMS03|PEMS04|PEMS07|PEMS08]`
- **Time Series Shape**: `--lag 12`, `--horizon 12`
- **Probabilistic Head**:
  - `--probabilistic [True|False]`
  - `--distribution [gaussian|student_t]`
  - `--nll_weight 0.5`, `--huber_weight 0.5`
- **Output Head**: `--tensor_decomp [none|tt]`, `--tt_ranks [1,20,1]`
- **Learning Rate**: `--lr`, `--lr_scheduler [reduce_on_plateau|multi_step]`, `--milestones`

> **Note**: For dataset-specific hyperparameters (e.g., `embed_dim`, `batch_size`), it is recommended to modify them directly in `config/dataset_config.py`.

## 📁 Project Structure

```
TPTTPN/
├── models/         # Model definitions (TPTTPN, Attention, OutputHead, etc.)
├── config/         # Default dataset configurations
├── lib/            # Data loading, normalization, and graph utilities
├── utils/          # Trainer, metrics, logger, and visualization tools
├── data/           # Root for datasets (user-provided)
├── experiments/    # Output directory for logs, checkpoints, and results
└── run.py          # Main script for training and testing
```

## 🧪 Example Command

A common command to replicate experiments:

```bash
python run.py --mode train --model TPTTPN --dataset PEMS08 \
  --probabilistic True --distribution student_t --tensor_decomp tt
```

```bash
python run.py --mode train --model TPTTPN --dataset PEMS08 \
  --probabilistic True --distribution student_t --tensor_decomp tt
```
