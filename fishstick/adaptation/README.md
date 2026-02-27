# Adaptation

Domain adaptation and transfer learning utilities.

## Overview

Provides tools for adapting models to new domains with limited labeled data.

## Usage

```python
from fishstick.adaptation import DomainAdaptation

# Domain adaptation
adapter = DomainAdaptation(
    source_model=source_model,
    method="domain_adversarial"
)

adapted_model = adapter.fit(source_data, target_data)
predictions = adapted_model(target_test_data)
```
