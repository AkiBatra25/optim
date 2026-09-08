# 📈 Uplift Modeling & Causal Optimization: Deep DragonNet vs. Two-Model T-Learner

An empirical machine learning and causal inference study benchmarking deep counterfactual representation learning (**Optimized DragonNet with Targeted Regularization**) against a traditional meta-learner baseline (**Two-Model T-Learner with Random Forests**) for Individual Treatment Effect (ITE) estimation on large-scale ad-tech data.

---

## 🚀 Overview

Standard supervised learning predicts user conversion probability $\hat{Y} = P(Y=1|X)$. In marketing, recommendation systems, and clinical trials, this leads to suboptimal intervention because it targets users who would have converted regardless of treatment (*"Sure Things"*).

**Uplift Modeling (Causal Machine Learning)** predicts the causal effect of an action:
$$\tau(X) = \mathbb{E}[Y(1) - Y(0) \mid X]$$

By ranking users according to predicted uplift $\hat{\tau}(X)$, campaigns can selectively target **"Persuadables"** while avoiding wasted ad spend on **"Sure Things"** and preventing adverse reactions from **"Sleeping Dogs"**.

---

## 📊 Dataset & Preprocessing

The benchmark is conducted on a **1,000,000-row stratified sample** derived from the 13.9M-row **Criteo AI Uplift Dataset**:
- **Features (`f0` – `f11`)**: 12 continuous, anonymized customer engagement features.
- **Treatment Indicator (`treatment`)**: Binary flag ($T \in \{0, 1\}$) indicating whether the ad was served.
- **Outcome Variable (`conversion`)**: Binary conversion outcome ($Y \in \{0, 1\}$).
- **Split Strategy**: 70% Train (700k), 15% Validation (150k), 15% Test (150k), stratified across the joint distribution of $(T, Y)$ to preserve class and treatment balance.
- **Scaling**: Robust feature standardization via `StandardScaler` fitted strictly on training partitions.

---

## 🏗️ Model Architectures & Methodology

### 1. Optimized DragonNet (Deep Causal Architecture)

DragonNet is a deep multi-head architecture specifically designed to overcome confounding bias in observational and semi-synthetic causal inference:

- **Shared Representation Trunk**: 3-stage dense residual blocks with GELU activations, Batch Normalization, L2 regularization, and Dropout to learn a unified latent space $\Phi(X)$.
- **Hypothesis Heads**:
  - **$\hat{Y}(0)$ Head**: Multi-layer feedforward network predicting potential outcome under control ($T=0$).
  - **$\hat{Y}(1)$ Head**: Multi-layer feedforward network predicting potential outcome under treatment ($T=1$).
  - **Propensity Head $\hat{t}(X)$**: Sigmoid network predicting treatment assignment probability $e(X) = P(T=1 \mid X)$.
- **Targeted Regularization (TARReg)**:
  - Integrates ideas from Targeted Maximum Likelihood Estimation (TMLE).
  - Uses a custom trainable scalar $\varepsilon$ layer and the clever covariate:
    $$h(X, T) = \frac{T}{\hat{e}(X)} - \frac{1 - T}{1 - \hat{e}(X)}$$
  - Perturbs outcome predictions $y_{\text{pert}} = \hat{y} + \varepsilon \cdot h(X, T)$ to directly minimize asymptotic bias in treatment effect estimation.
- **Total Loss Function**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{factual}}(Y, \hat{Y}) + \mathcal{L}_{\text{propensity}}(T, \hat{t}) + \alpha \cdot \mathcal{L}_{\text{TARReg}}(Y, y_{\text{pert}})$$

```text
               Input Features X (12 dim)
                          │
                          ▼
            [ Shared Representation Trunk Φ(X) ]
             (Dense + BatchNorm + GELU + ResBlock)
             ┌────────────┼────────────┐
             │            │            │
             ▼            ▼            ▼
      [ Y(0) Head ]  [ Y(1) Head ] [ Propensity Head ]
       Output: y0     Output: y1     Output: t_pred
             │            │            │
             └────────────┬────────────┘
                          ▼
                 [ Epsilon Layer ε ]
                          │
                          ▼
                [ TARReg Loss Layer ]
```

---

### 2. Two-Model T-Learner (Baseline Meta-Learner)

A decoupled two-model approach where separate regression trees model factual response in each treatment regime:
$$\hat{\mu}_1(X) = \text{RandomForestRegressor}(X_{T=1}, Y_{T=1})$$
$$\hat{\mu}_0(X) = \text{RandomForestRegressor}(X_{T=0}, Y_{T=0})$$
$$\hat{\tau}_{\text{T-Learner}}(X) = \hat{\mu}_1(X) - \hat{\mu}_0(X)$$

---

## 📈 Evaluation Metrics

Because counterfactual outcomes are never simultaneously observable for an individual (*The Fundamental Problem of Causal Inference*), standard supervised metrics (MSE, Accuracy, ROC-AUC) are inapplicable. Models are evaluated using:

1. **AUUC (Area Under the Uplift Curve)**:
   Measures cumulative incremental conversions as the population is targeted in descending order of predicted uplift:
   $$\text{Gain}(k) = \sum_{i=1}^k Y_i \cdot T_i - \sum_{i=1}^k Y_i \cdot (1 - T_i)$$
2. **Qini Curve & Qini Coefficient**:
   Adjusts cumulative uplift for treatment allocation ratios across the population deciles:
   $$\text{Qini}(k) = n_{t,y=1}(k) - n_{c,y=1}(k) \cdot \frac{N_t(k)}{N_c(k)}$$

---

## 🏆 Results & Comparison

### Quantitative Benchmark (150,000 Test Set)

| Model Architecture | AUUC Score | Qini Coefficient | $\Delta$ vs Baseline |
| :--- | :---: | :---: | :---: |
| **Two-Model T-Learner (Random Forest)** | 226.77 | 253.86 | *Baseline* |
| **Optimized DragonNet (TARReg)** | **301.67** | **338.24** | **+33.0% AUUC / +33.2% Qini** |

---

### Visual Diagnostic Curves

#### 1. Baseline T-Learner Results
<img width="1490" height="615" alt="T-Learner Uplift and Qini Curves" src="https://github.com/user-attachments/assets/228a37fc-254a-451d-a20d-df71d22b28eb" />

*T-Learner demonstrates positive uplift identification but exhibits higher variance due to uncoupled estimation across partitions.*

---

#### 2. Optimized DragonNet Results
<img width="1484" height="611" alt="DragonNet Uplift and Qini Curves" src="https://github.com/user-attachments/assets/0c9c18d9-c08c-4fb4-b4bf-f918bacf8f9c" />

*DragonNet exhibits smooth, monotonic cumulative gain with sharp early-decile concentration of high-uplift responders.*

---

#### 3. Uplift Curve Comparison
<img width="870" height="591" alt="Uplift Curve Comparison" src="https://github.com/user-attachments/assets/37c24712-117a-4c7b-8ba6-31adceed3905" />

*DragonNet (AUUC = 301.67) substantially outperforms T-Learner (AUUC = 226.77) across all target proportions.*

---

#### 4. Qini Curve Comparison
<img width="866" height="581" alt="Qini Curve Comparison" src="https://github.com/user-attachments/assets/0c7f5900-eb07-471b-8eb4-0da674c5e307" />

*DragonNet maintains a strictly dominant Qini curve (Qini = 338.24 vs 253.86), demonstrating superior ranking fidelity under treatment imbalance.*

---

## 🔑 Key Findings & Takeaways

1. **Shared Representation Prevents Regularization Bias**: Independent models (T-Learner) fit different feature subspaces for treated and control groups, distorting counterfactual difference estimates. DragonNet forces both potential outcomes to be mapped from the same shared representation.
2. **Targeted Regularization (TARReg) Stabilizes Extreme Weights**: Propensity score estimation embedded directly into the loss function helps penalize regions with poor overlap, stabilizing counterfactual predictions.
3. **Business & ROI Impact**: By selecting customers using DragonNet's top-ranked deciles, marketing campaigns capture **~33% more incremental conversions** for the same targeting budget compared to traditional methods.

---

## 📁 Repository Structure

```text
├── Preprocessing_EDA.ipynb       # 13M -> 1M stratified sampling, data hygiene & feature correlation
├── Hard_Easy_problem.ipynb       # DragonNet architecture, TARReg loss, T-Learner baseline & plots
├── requirements.txt              # Environment dependencies
└── README.md                     # Research overview and documentation
```

---

## ⚡ Quickstart & Reproduction

### Prerequisites
Install the required packages:
```bash
pip install -r requirements.txt
```

### Execution Steps
1. **Data Preprocessing & EDA**:
   Open and execute [`Preprocessing_EDA.ipynb`](file:///c:/IIITB/Academics/UPLIFT-MODELING-OPTIMIZATION/UPLIFT-MODELING-OPTIMIZATION-main/Preprocessing_EDA.ipynb) to inspect the stratified sampling methodology and exploratory feature distributions.
2. **Model Training & Evaluation**:
   Open and run [`Hard_Easy_problem.ipynb`](file:///c:/IIITB/Academics/UPLIFT-MODELING-OPTIMIZATION/UPLIFT-MODELING-OPTIMIZATION-main/Hard_Easy_problem.ipynb) to train both DragonNet and T-Learner models, compute AUUC/Qini metrics, and render comparison plots.
