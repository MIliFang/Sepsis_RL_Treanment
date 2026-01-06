# Sepsis RL Treatment Policy

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This repository implements a **multi-task reinforcement learning framework** for optimal vasopressor administration in septic ICU patients, using MIMIC-III data. The system combines clinical constraints, behavior cloning, and conservative Q-learning to learn safe and effective treatment policies.

## 🔍 Key Features

- **Clinically constrained action space**: Only valid drug combinations (NE, AVP, EPI, PHE, DOPA) based on real-world clinical guidelines
- **Hybrid training approach**: Behavior cloning + TD learning + CQL (Conservative Q-Learning) regularization
- **Multi-task learning**: Jointly optimizes treatment policy and 90-day mortality prediction
- **Comprehensive evaluation**: 
  - Off-policy evaluation (OPE) using Weighted Importance Sampling (WIS)
  - Clinical consistency analysis (action matching with physicians)
  - Subgroup analysis for tachycardic patients (HR ≥ 100)
- **Interpretability**: Action distribution analysis, attention visualization, and ablation studies
- **Robust validation**: Train/validation/test splits with patient-level separation

## 📦 Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Required packages:**
- Python 3.8+
- PyTorch 1.10+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- SciPy

## 🚀 Quick Start

### 1. Data Preparation
1. Obtain MIMIC-III dataset access from [PhysioNet](https://physionet.org/content/mimiciii/1.4/)
2. Preprocess the data to create `mimictabl.csv` with the required columns (see `data/README.md` for schema)
3. Place your `mimictabl.csv` file in the `./dataset/` directory

### 2. Run Main Training
Execute the end-to-end training pipeline:

```bash
python main.py
```

### 3. Run Ablation Study
Compare different model configurations:

```bash
python ablation_study.py
```

## 📂 Output Structure

All results are automatically saved under `./outputs/`:

```
outputs/
├── data/           # Training history, experiment results (JSON/CSV)
├── models/         # Trained model checkpoints (.pth files)
├── figs/           # Generated visualizations (WIS analysis, dopamine usage plots)
└── logs/           # Detailed training logs with timestamp
```

## 📊 Sample Results

| Model | Clinical Accuracy | WIS Score | JSD | Avg Q-Value |
|-------|------------------|-----------|-----|-------------|
| MLP (Baseline) | 68.2% | 12.4 | 0.42 | 1.85 |
| + MTL | 71.5% | 13.1 | 0.38 | 1.72 |
| + CQL | 70.8% | 13.6 | 0.32 | **1.21** |
| Attention + CQL | **74.3%** | **14.2** | **0.28** | 1.35 |

- **Clinical Accuracy**: % of actions matching physician decisions
- **WIS Score**: Estimated policy return via Off-Policy Evaluation
- **JSD**: Jensen-Shannon Divergence (lower = more similar to physician policy)
- **Avg Q-Value**: Indicates conservatism (lower = less overestimation)

> Full results with confidence intervals available in `outputs/data/`.

## 🏗️ Project Structure

```
sepsis-rl-treatment/
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── config.py                   # All hyperparameters and configuration
├── main.py                     # Main training pipeline
├── ablation_study.py           # Ablation experiments
├── data/                       # Data documentation
│   └── README.md
├── dataset/                    # Place your data here (gitignored)
│   └── README.md
├── src/                        # Core source code
│   ├── __init__.py
│   ├── data_processor.py       # Data preprocessing and action discretization
│   ├── rl_models.py            # RL models (Dueling DQN, Attention CQL)
│   ├── physician_policy.py     # Physician policy simulator for OPE
│   ├── OPE.py                  # Off-Policy Evaluation implementation
│   ├── setup.py                # Logging and random seed utilities
│   └── arrhythmia_analysis.py  # Tachycardia subgroup analysis
├── outputs/                    # Generated outputs (gitignored)
│   ├── data/
│   ├── models/
│   ├── figs/
│   └── logs/
└── notebooks/                  # Optional Jupyter notebooks
    └── explore_data.ipynb
```

## 🔬 Key Components

### Action Space Construction
- Extracts high-frequency, clinically plausible drug combinations from real data
- Applies clinical constraints (e.g., DOPA prohibited with high-dose NE)
- Creates a reduced action space of 59 valid combinations (from 114 rule-based)

### Model Architecture
- **MTL Dueling Q-Learning**: Standard dueling DQN with mortality prediction head
- **MTL Attention CQL**: Token-based attention encoder with CQL regularization
- **Conservative Q-Learning**: Penalizes overestimation on out-of-distribution actions

### Evaluation Metrics
- **Off-Policy Evaluation**: Weighted Importance Sampling (WIS) with clipping
- **Clinical Consistency**: Action matching accuracy and distribution similarity (JSD)
- **Safety Monitoring**: Average Q-value as overestimation indicator

## 📝 Configuration

All hyperparameters are centralized in `config.py`:

- **Clinical parameters**: Drug dosage ranges, action constraints
- **RL parameters**: Gamma, learning rate, batch size, exploration strategy
- **CQL parameters**: Alpha coefficient, diversity penalty
- **MTL parameters**: Auxiliary loss weight, dynamic adjustment
- **Training parameters**: Epochs, gradient clipping, oversampling ratio

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MIMIC-III Critical Care Database (Johnson et al., Scientific Data 2016)
- Conservative Q-Learning (Kumar et al., NeurIPS 2020)
- Weighted Importance Sampling (Precup et al., ICML 2000)

## 📬 Citation

If you use this code in your research, please cite:

```bibtex
@misc{sepsisrl2026,
  author = {MiliFang},
  title = {Sepsis RL Treatment Policy},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/MiliFang/sepsis-rl-treatment}}
}
```
