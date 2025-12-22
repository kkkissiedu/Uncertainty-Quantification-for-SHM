# Uncertainty Quantification for SHM — Concrete Strength Prediction

Two UQ methods for predicting concrete compressive strength: a deep ensemble with aleatoric uncertainty decomposition, and a Gaussian process surrogate. Both serve as probabilistic alternatives to physical destructive testing in structural health monitoring.

## Results

| Metric        | Deep Ensemble | GP       |
|---------------|---------------|----------|
| R²            | —             | 0.8749   |
| RMSE (MPa)    | —             | 5.68     |
| PICP 95% (%)  | —             | —        |
| MPIW (MPa)    | —             | —        |
| Params        | ~5 × 8.5 K    | kernel   |

*Trained on UCI Concrete Compressive Strength (1030 samples). Evaluated on 20% held-out test set.*

*Fill ensemble columns after running `evaluate.py`.*

## Architecture

**Deep Ensemble:** Five `AleatoricNet` members (8→64→64→[μ, log σ²]), each trained with Gaussian NLL loss. Epistemic uncertainty from member disagreement, aleatoric uncertainty from the learned variance head.

**Gaussian Process:** RBF + WhiteKernel with optimised hyperparameters via marginal likelihood maximisation. Provides a closed-form posterior predictive distribution.

## Setup

```bash
conda activate cuda_pt
pip install -r requirements.txt
```

## Usage

**Train deep ensemble:**
```bash
python train_ensemble.py
```

**Train GP:**
```bash
python train_gp.py
```

**Evaluate (prints PICP, MPIW, R², RMSE table):**
```bash
python evaluate.py
```

**Visualise:**
```bash
python visualise.py
```

## Dataset

```
Concrete_Data.xls   (UCI Concrete Compressive Strength, 1030 samples)
    Cement, BlastFurnaceSlag, FlyAsh, Water, Superplasticizer,
    CoarseAggregate, FineAggregate, Age  →  Strength (MPa)
```

Source: [UCI ML Repository — Concrete Compressive Strength](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength)

## Citation

```
@misc{yeh1998,
  title  = {Concrete Compressive Strength},
  author = {Yeh, I-Cheng},
  year   = {1998},
  note   = {UCI Machine Learning Repository}
}
```
