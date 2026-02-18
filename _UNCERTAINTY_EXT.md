# TODO List: Uncertainty Quantification Extensions for fishstick

## Directory: /home/runner/workspace/fishstick/uncertainty_ext/

### Phase 1: Core Infrastructure
- [x] 1. Create directory structure and __init__.py
- [x] 2. Implement conformal_prediction.py - Conformal prediction implementations
- [x] 3. Implement temperature_scaling.py - Temperature scaling for calibration

### Phase 2: Ensemble & Disagreement
- [x] 4. Implement ensemble_disagreement.py - Ensemble disagreement measures
- [x] 5. Implement epistemic_aleatoric.py - Epistemic vs aleatoric uncertainty decomposition

### Phase 3: Advanced Methods
- [x] 6. Implement uncertainty_losses.py - Uncertainty-aware training losses
- [x] 7. Implement batch_bald.py - BALD for deep ensembles
- [x] 8. Implement uncertainty_metrics.py - Comprehensive metrics

### Phase 4: Integration
- [x] 9. Create full __init__.py with all exports
- [x] 10. Add documentation and examples

## Module Details:

### 1. conformal_prediction.py
- Adaptive conformal prediction
- Split conformal prediction
- Full conformal prediction
- Jackknife+ 
- CV+ conformal prediction

### 2. temperature_scaling.py
- Temperature scaling for neural networks
- Vector temperature scaling
- Class-wise temperature scaling
- Platt scaling
- Beta calibration

### 3. ensemble_disagreement.py
- Variance-based disagreement
- Entropy-based disagreement
- Cosine disagreement
- pairwise KL divergence
- disagreement visualization utilities

### 4. epistemic_aleatoric.py
- Mutual information decomposition
- Information gain estimation
- Ensemble-based epistemic/aleatoric separation
- Gradient-based uncertainty (MCMC-like)
- Dropout-based separation

### 5. uncertainty_losses.py
- Focal loss with uncertainty
- Label smoothing with uncertainty
- Evidential regression loss
- Uncertainty-aware contrastive loss
- Mixup with uncertainty weighting

### 6. batch_bald.py
- Batch BALD implementation
- BALD with abstention
- Expected entropy search
- Batch dropdown selection

### 7. uncertainty_metrics.py
- ECE (Expected Calibration Error)
- NLL (Negative Log-Likelihood)
- Brier Score
- AUROC for OOD detection
- Confidence accuracy correlation
