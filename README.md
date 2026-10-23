# Transaction Risk Scoring

Fraud detection on **590,540 IEEE-CIS transactions**, comparing XGBoost, LightGBM, and a normal-only PyTorch autoencoder with chronological evaluation, component SHAP explanations, and local MLflow tracking.

The complete workflow, results, and charts are in **[transaction-risk-scoring.ipynb](transaction-risk-scoring.ipynb)**.

## Results

The ensemble was selected by validation average precision. Its blend weight and alert threshold were frozen before test scoring.

| Chronological test model | ROC-AUC | Average precision | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Reference XGBoost | 0.8793 | 0.4559 | 0.5341 | 0.4122 | 0.4653 |
| Reference LightGBM | 0.8762 | 0.4546 | 0.5845 | 0.3888 | 0.4670 |
| Temporal/entity LightGBM | 0.8771 | 0.5233 | 0.6167 | 0.4779 | 0.5385 |
| Temporal/entity XGBoost | 0.8993 | 0.5382 | 0.6466 | 0.4668 | 0.5422 |
| **25% LightGBM + 75% XGBoost** | **0.9017** | **0.5517** | **0.6826** | **0.4604** | **0.5499** |

The selected ensemble caught **1,871 fraud cases** with **870 false alarms**, while missing **2,193 fraud cases**. Accuracy was **97.41%**; always predicting legitimate would already achieve 96.56%, so accuracy alone is misleading.

The optional autoencoder scored **0.7568 ROC-AUC / 0.1147 average precision**. Its numeric representation differs from the trees. Labels identify legitimate training rows and select its validation threshold, so development is not fully label-free.

## Notebook flow

1. Setup
2. Chronological data split
3. Features
4. Train and compare on validation
5. Evaluate the frozen selection
6. Explain the LightGBM component
7. Score a small batch
8. Optional autoencoder comparison
9. Local experiment tracking
10. Evaluation context and limitations

The main workflow reproduces the final four-model comparison and three fixed blends. It does not rerun every earlier experiment. Historical results and the separate random-split diagnostic remain documented in the notebook and its embedded results metadata.

## Run

Download `train_transaction.csv` and `train_identity.csv` from [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data), subject to its access terms, and place them in `data/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab transaction-risk-scoring.ipynb
```

Leave `RUN_TRAINING = False` to view the embedded results. Set it to `True` to reproduce the main workflow. Set `RUN_AUTOENCODER = True` as well to include the optional comparison. Run cells in order from the project root. Fresh models and tracking artifacts are written under `output/notebook_run/`; existing frozen studies are protected from overwriting.

The completed model runs used Python 3.14.6 on CPU. Embedded results come from those runs. The reorganized notebook has passed syntax and structure checks but has not been executed end-to-end in a Jupyter kernel.

## Evaluation and sources

The split keeps timestamp ties together: **354,324 training**, **118,108 validation**, and **118,108 test** rows. Feature mappings and preprocessing fit on training only. Early stopping, model selection, blend selection, and alert thresholds use validation.

Reference settings adapt [this project snapshot](https://github.com/neshatsh/Financial-Fraud-Detection-System/tree/cc44401d9640072a740dd353a93713743903f31c) to CPU and chronological validation. Temporal/group feature ideas follow [the competition winning team's technical explanation](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/). Proxy groups are constructed features, not verified identities.

This is an iterative study on a previously reported test period, not a new independent test. A separate random-split diagnostic gave the adapted reference XGBoost 0.9444 ROC-AUC / 0.6675 average precision and the richer LightGBM 0.9744 / 0.8901. Those diagnostic scores do not replace the chronological results or estimate future-period performance.

Scores are not calibrated probabilities, F1 thresholds do not represent verified business costs, and linked transactions may remain dependent across time splits. SHAP describes a model component, not the complete blended probability. No live serving, production monitoring, or monetary savings are claimed.

See [LICENSE](LICENSE).
