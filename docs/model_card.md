# Model Card: Credit Risk Predictive Ensembles and Neural Architectures

## Model Details
- **Developer**: Anar Gadirli (ASOIU / Research Project)
- **Model Type**: Supervised binary probability of default (PD) estimation
- **Task**: Predict financial delinquency (`SeriousDlqin2yrs` = 1) over a 24-month horizon.
- **Input Features**: 10 normalized and winsorized financial ratios and historical delinquency indicators.
- **Output**: Calibrated default probability $P(Y=1|X) \in [0, 1]$.

## Evaluated Architectures
1. **XGBoost Classifier (`XGBClassifier`)**:
   - Objective: `binary:logistic`
   - Estimators: 200 trees, max depth: 5, learning rate: 0.05
   - Regularization: L1/L2 shrinkage, subsample: 0.8, colsample: 0.8
   - Compatibility: Direct compatibility with TreeSHAP and KernelSHAP.
2. **LightGBM Classifier (`LGBMClassifier`)**:
   - Objective: `binary`
   - Estimators: 200 trees, max depth: 5, num leaves: 31, learning rate: 0.05
   - Compatibility: TreeSHAP (native C++ tree traversal) and KernelSHAP.
3. **Multilayer Perceptron (`PyTorch MLP`)**:
   - Architecture: Input (10) -> Linear(64) -> ReLU -> Dropout(0.2) -> Linear(32) -> ReLU -> Linear(1) -> Sigmoid
   - Optimizer: Adam (lr=0.001, weight decay=1e-4), BCEWithLogitsLoss
   - Compatibility: KernelSHAP, FastSHAP (amortized surrogate), LIME.
4. **Logistic Regression (`LogisticRegression`)**:
   - Intrinsic linear model providing an analytical baseline ($\beta_i \cdot x_i$).

## Evaluation Metrics
- Discrimination: ROC-AUC, Precision-Recall AUC (PR-AUC), F1-Score, Balanced Accuracy.
- Calibration: Brier Score, Expected Calibration Error (ECE).
- Decision Threshold: Selected via validation set optimizing F1-score and Youden's J statistic.
