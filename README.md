# Echo State Networks (ESN) – MNIST & ECG5000

This repository contains a set of **Echo State Network (ESN)** experiments for:
- **Image classification (MNIST)** — ESN vs a linear-regression baseline
- **Time-series classification (ECG5000)** — ESN with multiple reservoir weight initializations, confusion matrices, and basic stats (ANOVA/Tukey)

It’s written in **pure NumPy + scikit-learn**, with Matplotlib/Seaborn for plots.

---

## ✨ What’s inside

### MNIST (handwritten digits, 10 classes)
- Loads `mnist_784` from OpenML  
- Normalizes inputs and one-hot encodes labels  
- Trains:
  - An **ESN classifier** (reservoir + linear readout)
  - A **linear regression baseline** on raw pixels  
- Evaluates on validation and test sets, prints **accuracy**, **confusion matrix**, and **classification report**

**Example results (from a sample run — will vary):**
- ESN **Val** ≈ `0.921`, **Test** ≈ `0.920`
- Linear regression **Val** ≈ `0.851`, **Test** ≈ `0.854`

### ECG5000 (univariate ECG time series, 5 classes)
- Loads `ECG5000` (OpenML `data_id=44793`)
- **Balances** the dataset by downsampling to the smallest class count
- Splits into train/test
- Trains ESNs under **9 reservoir weight initializations**:
  - `sparsity`, `identity`, `diag`, `tridiag`, `upper`, `lower`, `leading_diagonal`, `diag_plus_leading`, `trailing_diagonal`
- Produces:
  - **Heatmaps** of reservoir weight matrices (`connectivity_matrices.png`)
  - **Confusion matrices** per method
  - A **results table** (`esn_results.csv`)
  - (Illustrative) **ANOVA + Tukey** post-hoc test (see note below)

**Example test accuracies (from a sample run — will vary):**
| Method | Acc. |
|---|---|
| sparsity | 0.9796 |
| trailing_diagonal | 0.9748 |
| diag | 0.9700 |
| leading_diagonal | 0.9688 |
| diag_plus_leading | 0.9519 |
| identity | 0.8990 |
| tridiag | 0.8498 |
| upper | 0.7428 |
| lower | 0.6575 |

> ⚠️ **Stats note:** The included ANOVA/Tukey snippet is illustrative only. In the shown code each method is evaluated **once**, so ANOVA warns about degenerate data. For valid significance tests, run **multiple seeds/repetitions per method** and aggregate.

---

## 🔧 Environment

- Python **3.10+**
- Install dependencies:
  ```bash
  pip install numpy pandas scikit-learn matplotlib seaborn statsmodels
