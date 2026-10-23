# Transaction Risk Scoring

Fraud detection on **590,540 IEEE-CIS transactions**, comparing XGBoost, LightGBM, and a PyTorch autoencoder with chronological evaluation, SHAP explanations, and local MLflow tracking.

The latest validation-selected model achieved **0.5517 average precision**, **0.9017 ROC-AUC**, **68.3% precision**, and **46.0% recall** on the existing chronological test period.

The complete implementation, results, and charts are in **[transaction-risk-scoring.ipynb](transaction-risk-scoring.ipynb)**. No separate source modules or report files are required.

## Initial results

| Model | ROC-AUC | Average precision | Precision | Recall |
|---|---:|---:|---:|---:|
| LightGBM | **0.8854** | **0.4679** | 0.4988 | 0.4254 |
| XGBoost | 0.8812 | 0.4645 | **0.5072** | 0.4183 |
| Autoencoder | 0.7568 | 0.1147 | 0.1850 | 0.3063 |
| Dummy baseline | 0.5000 | 0.0344 | 0.0344 | 1.0000 |

The dataset is split into 354,324 training, 118,108 validation, and 118,108 test transactions in chronological order, keeping timestamp ties together. Feature mappings and preprocessing fit on training only. Model selection, early stopping, and alert thresholds use validation. LightGBM was selected by validation average precision; thresholds maximize validation F1.

Average precision is the specific precision-recall summary used here. Held-out fraud prevalence is 3.44%. The autoencoder learns from known legitimate transactions; labels select that subset and its validation threshold, so development is not fully label-free.

The autoencoder was added after the supervised test results were reported. Its fixed recipe and validation-selected threshold are documented in the notebook. Its result reuses the existing holdout, rather than providing a new independent test.

## Validation-selected improvements

A fixed four-candidate follow-up tested native categorical handling, more boosting rounds, class weights, and training-only entity/amount summaries. The selected model increased validation average precision from **0.5560** to **0.6166**.

| Metric | Initial LightGBM | Selected follow-up |
|---|---:|---:|
| Average precision | 0.4679 | **0.4937** |
| ROC-AUC | 0.8854 | 0.8742 |
| Precision | 0.4988 | 0.6228 |
| Recall | 0.4254 | 0.4124 |

Average precision, precision, and F1 improved; ROC-AUC and recall decreased. This is a measured tradeoff rather than a gain on every metric.

All candidate selection and thresholds used validation. The follow-up was evaluated once on the existing, previously reported test period; it is **not a new independent test**. Original results are retained above. The notebook includes candidate results, the selected configuration, scoring code, and a comparison chart.

## Controlled reference comparison

A new study compared the reference project's 500-tree settings under our chronological protocol, then tested temporal/entity features inspired by [the competition winning-team explanation](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/). Frequency maps, missingness decisions, and group summaries fit on training only. This adapts the reference settings; it is not an exact reproduction of the original random-split notebook.

Four model fits and three fixed blends were declared before training. **blend_0.25** was selected by validation average precision (0.6596); its threshold was frozen before test scoring.

| Model | ROC-AUC | Average precision | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| reference_xgboost_500 | 0.8793 | 0.4559 | 0.5341 | 0.4122 | 0.4653 |
| reference_lightgbm_500 | 0.8762 | 0.4546 | 0.5845 | 0.3888 | 0.4670 |
| temporal_entity_lightgbm | 0.8771 | 0.5233 | 0.6167 | 0.4779 | 0.5385 |
| temporal_entity_xgboost | 0.8993 | 0.5382 | 0.6466 | 0.4668 | 0.5422 |
| blend_0.25 | 0.9017 | 0.5517 | 0.6826 | 0.4604 | 0.5499 |

The selected model caught **1,871 fraud cases** with **870 false alarms**, achieving **97.41% accuracy**. Accuracy alone remains misleading because 96.56% of test transactions are legitimate.

This is an iterative evaluation on the existing, previously reported test period, not a new independent test. The table includes all predefined models rather than choosing a winner on test results. Reported scores from a different split should not be treated as an equal-protocol benchmark.

### Random-split diagnostic

The adapted reference XGBoost recipe scored **0.9444 ROC-AUC / 0.6675 average precision** under a clean random 60/20/20 split, close to the reference project's reported **0.9483 / 0.6755**. Under our chronological split, the adapted recipe scored **0.8793 / 0.4559**. The recipe and training count are held fixed in this split comparison; feature mappings and early stopping use only the allowed training/validation partitions.

The temporal/entity LightGBM scored **0.9744 / 0.8901** on the random diagnostic. These are diagnostic scores, not future-period performance. Proxy-group novelty was 68.6% in the chronological test period versus 27.8% in the random test split. Proxy groups are constructed features, not verified identities; random splitting can share those groups across time.

The chronological winner was frozen before this diagnostic and was not changed by it. Random scores are shown separately rather than replacing the primary evaluation. Full paired results, adaptations, and source attribution are in Section 7 of the notebook.

## Run

1. Download the labeled transaction and identity tables from [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data), subject to its access terms.
2. Place `train_transaction.csv` and `train_identity.csv` in `data/`.
3. Install `requirements.txt` in your Python environment and open the notebook from the project root.
4. Read the embedded completed-run outputs, or set `RUN_TRAINING = True` and run in order to reproduce the baseline. Set `RUN_IMPROVEMENTS = True` in Section 6 for the first follow-up, or `RUN_CONTROLLED = True` in Section 7 for the controlled reference and temporal/entity study. `RUN_RANDOM_DIAGNOSTIC = True` optionally runs the paired split diagnostic after the controlled policy is frozen. Fresh artifacts are written under `output/notebook_run/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab transaction-risk-scoring.ipynb
```

The completed model runs used Python 3.14.6 on CPU. The consolidated notebook has passed syntax and structure checks; it has not been executed end-to-end in a Jupyter kernel. Embedded results and charts come from the completed model runs.

## Methods

- Training-only card frequencies, amount features, relative-time phase, missing-value handling, and categorical encoding.
- Class-weighted XGBoost and LightGBM with validation early stopping.
- Normal-only numeric autoencoder with an inner reconstruction holdout and a validation-selected anomaly threshold.
- SHAP analysis on a representative validation sample.
- Saved preprocessing and models for raw-record inference; local MLflow experiment logging.

## Limitations

This is an offline study. Scores are not calibrated probabilities, F1 thresholds do not represent verified business costs, and linked transactions may remain dependent across time splits. Tree and autoencoder input representations differ. No live serving, production monitoring, or monetary savings are claimed.

See [LICENSE](LICENSE).
