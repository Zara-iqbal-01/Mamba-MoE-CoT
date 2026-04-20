# Mamba-MoE-CoT
Novel architecture integrating Mamba SSM, Mixture-of-Experts, and Chain-of-Thought distillation for text classification

# MambaMoECoT: Hybrid State Space Model for Text Classification

A novel architecture integrating Mamba SSM, Mixture-of-Experts (MoE), and 
Chain-of-Thought (CoT) distillation for efficient sequence modeling — combining 
three complementary techniques in a single unified classifier.


## Overview

Transformers dominate NLP, but alternatives like **Mamba** (selective state space models) 
offer linear-time inference and stronger long-context handling. This project asks: can Mamba, 
**Mixture-of-Experts (MoE)**, and **Chain-of-Thought (CoT) distillation** coexist in a 
single unified architecture?

We built `MambaMoEClassifier` — a custom model integrating all three — and evaluated it 
on SST-2 binary sentiment classification.

## Architecture

- **Mamba blocks** (×4): Selective SSM layers using `mamba-ssm`, state dim=32, conv window=4
- **MoE layer**: 8 experts, top-k=2 routing, 2-layer MLP per expert, auxiliary load-balancing loss
- **CoT head**: Auxiliary classification head trained on pseudo-labels based on sequence complexity
- **Loss**: Cross-entropy (main) + CoT loss (λ=0.2) + MoE load-balancing loss (λ=0.05)

## Results (SST-2, 5 epochs)

| Epoch | Val Loss | Accuracy | F1 Score |
|-------|----------|----------|----------|
| 3     | 0.5445   | 73.28%   | 0.7325   |
| **4** | **0.5388** | **73.74%** | **0.7366** |
| 5     | 0.5404   | 72.48%   | 0.7246   |

Peak performance at **epoch 4: 73.74% accuracy, 0.7366 F1**.

## Key Findings

- MoE expert utilization was volatile across epochs (4.4%–31.5%), with best performance 
  coinciding with *lowest* uniform utilization — suggesting the model benefits from 
  expert specialization over uniform routing for binary classification
- Validation loss plateaued after epoch 3, indicating early stopping as a future direction
- Architecture is viable at small scale (~500K params, 67K samples) despite reference 
  papers operating at 25M–790M params

## Tech Stack

`PyTorch` · `mamba-ssm` · `HuggingFace Transformers` · `Datasets` · `AdamW` · `AMP (fp16)` · `Google Colab T4`

## Links

- 📓 [Colab Notebook](https://colab.research.google.com/drive/1a6557oXASDlFRWZn8O6MEGbBto60jKf7?usp=sharing)
- 📄 [Full Report](./ZaraEricMatthewJacob_Final_Project.pdf)
- 🎥 [Video Walkthrough](https://youtu.be/NruJO_AlAe4)

## Team

Columbia University · COMS W4995 Deep Learning  
**Zara Iqbal** · Matthew Reynolds · Jacob Boyar · Eric Yi 
