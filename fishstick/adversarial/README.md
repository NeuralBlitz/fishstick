# Adversarial

Adversarial attacks, defenses, and certified robustness.

## Overview

Comprehensive toolkit for adversarial machine learning including attack methods, defense mechanisms, and certified robustness verification.

## Attacks

```python
from fishstick.adversarial import FGSM, PGD, CW, DeepFool, AutoAttack

# Fast Gradient Sign Method
attack = FGSM(epsilon=0.3)
adversarial_x = attack(model, x, y)

# Projected Gradient Descent
attack = PGD(epsilon=0.3, alpha=0.01, iterations=40)
adversarial_x = attack(model, x, y)

# Carlini-Wagner
attack = CW(c=1.0, iterations=1000)
adversarial_x = attack(model, x, y)

# AutoAttack (ensemble)
attack = AutoAttack(epsilon=0.3)
adversarial_x = attack(model, x, y)
```

## Defenses

```python
from fishstick.adversarial import (
    AdversarialTraining,
    TRADES,
    RandomizedSmoothing,
    InputTransformation,
)

# Adversarial training
defense = AdversarialTraining(attack=PGD(), epochs=100)
defense.fit(model, train_data)

# TRADES
defense = TRADES(
    beta=6.0,
    optimizer="adam",
    epochs=100
)
defense.fit(model, train_data)

# Randomized smoothing
defense = RandomizedSmoothing(sigma=0.25, n_classes=10)
certified = defense.certify(x)
```

## Certified Robustness

```python
from fishstick.adversarial import (
    RandomizedSmoothingCertifier,
    IBP,
    CROWN,
    CROWN_IBP,
)

# Interval Bound Propagation
certifier = IBP(epsilon=0.3)
lower, upper = certifier.verify(model, x)

# CROWN linear bound
certifier = CROWN(epsilon=0.3)
lower, upper = certifier.verify(model, x)

# Randomized smoothing certification
certifier = RandomizedSmoothingCertifier(sigma=0.25, n0=100, n=1000)
radius = certifier.certify(model, x)
```
