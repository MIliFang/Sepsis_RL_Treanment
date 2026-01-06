# Sepsis RL Treatment Policy

This repository implements a **multi-task reinforcement learning framework** for optimal vasopressor administration in septic ICU patients, using MIMIC-III data.

## 🔍 Key Features

- **Clinically constrained action space**: Only valid drug combinations (NE, AVP, EPI, PHE, DOPA)
- **Hybrid training**: Behavior cloning + TD learning + CQL regularization
- **Off-policy evaluation (OPE)**: Weighted Importance Sampling (WIS)
- **Interpretability**: Action distribution analysis, tachycardia subgroup study
- **Ablation studies**: MTL, CQL, and Attention modules

## 📦 Requirements

```bash
pip install -r requirements.txt
