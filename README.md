# 🏗️ Probabilistic Regression Models for Structural Engineering

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Predicting concrete compressive strength with uncertainty-aware machine learning models for safer, risk-informed decision making in engineering.

---

## 📌 Overview

This repository implements **two powerful probabilistic regression models** to estimate concrete compressive strength — not just as a single number, but with principled uncertainty bounds. This is critical in high-stakes fields like structural engineering, where understanding risk is as important as making accurate predictions.

---

## 🔍 Models Implemented

### 1. 🧠 Deep Ensemble with Aleatoric Uncertainty

A state-of-the-art deep learning ensemble approach that models both:

- **Aleatoric Uncertainty**: Noise inherent in the data.
- **Epistemic Uncertainty**: Uncertainty due to limited data/model ignorance.

#### ✅ Key Features
- 📊 **Disentangled Uncertainty** — Separates data noise from model uncertainty.
- 🧪 **Robust Predictions** — Ensemble averaging improves stability and accuracy.
- 🚀 **Scalable & Modern** — Built on PyTorch and suited for real-world deployment.

#### 📈 Results
Generates predictions with combined uncertainty bounds (±2σ), useful for structural safety evaluations.

> ![Sample Output](docs/deep_ensemble_uncertainty_plot.png)  
> *Mean prediction with 95% confidence interval*

#### 🛠️ How to Use

```bash
# Install required packages
pip install torch numpy pandas scikit-learn matplotlib tensorboard
