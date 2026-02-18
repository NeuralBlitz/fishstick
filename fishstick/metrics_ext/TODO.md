# TODO: Evaluation Metrics Extension for Fishstick - COMPLETED

## Phase 1: Classification Metrics ✅
- [x] 1.1 Create classification_metrics.py
  - [x] Macro F1 Score
  - [x] Micro F1 Score  
  - [x] Weighted F1 Score
  - [x] Balanced Accuracy
  - [x] Cohen's Kappa
  - [x] Matthew's Correlation Coefficient
  - [x] Specificity/Sensitivity
  - [x] Per-class metrics

## Phase 2: Regression Metrics ✅
- [x] 2.1 Create regression_metrics.py
  - [x] Mean Absolute Error (MAE)
  - [x] Mean Squared Error (MSE)
  - [x] Root Mean Squared Error (RMSE)
  - [x] Huber Loss (as metric)
  - [x] R² Score (coefficient of determination)
  - [x] Adjusted R²
  - [x] MAPE (Mean Absolute Percentage Error)
  - [x] SMAPE (Symmetric MAPE)
  - [x] Quantile Loss

## Phase 3: Ranking Metrics ✅
- [x] 3.1 Create ranking_metrics.py
  - [x] NDCG@k (Normalized Discounted Cumulative Gain)
  - [x] MAP (Mean Average Precision)
  - [x] MRR (Mean Reciprocal Rank)
  - [x] Hit Rate@k
  - [x] Precision@k / Recall@k
  - [x] Average Precision
  - [x] DCG

## Phase 4: Detection Metrics ✅
- [x] 4.1 Create detection_metrics.py
  - [x] mAP (mean Average Precision)
  - [x] IoU (Intersection over Union)
  - [x] Precision-Recall curve
  - [x] AP per class
  - [x] F1-score at different thresholds
  - [x] Confusion matrix
  - [x] TP/FP/FN counting

## Phase 5: Custom Domain Metrics ✅
- [x] 5.1 Create custom_metrics.py
  - [x] Time series metrics (MAE, MASE)
  - [x] NLP metrics (BLEU, ROUGE simplified)
  - [x] Generative model metrics (FID, IS)
  - [x] Statistical parity difference
  - [x] Equal opportunity difference
  - [x] Composite metrics

## Phase 6: Utilities and Exports ✅
- [x] 6.1 Create __init__.py with proper exports
- [x] 6.2 Create base classes for metric computation
- [x] 6.3 Add comprehensive docstrings and examples
