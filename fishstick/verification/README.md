# Verification Module

## Overview

Formal verification for ML models through dependently-typed specifications, safety property checking, and proof certificate generation. Integrates with proof assistants (Lean 4 / Coq) for machine-checked proofs.

## Purpose and Scope

- Dependent type specifications for neural networks
- Safety property verification (robustness, fairness)
- Lipschitz constant estimation
- SMT-based verification
- Proof certificate generation

## Key Classes and Functions

### TypeSpec

Type specification for input/output constraints.

```python
from fishstick.verification import TypeSpec

input_spec = TypeSpec(
    name="ImageInput",
    constraints=["normalized", "shape=(3,224,224)"],
    dependencies=[]
)

output_spec = TypeSpec(
    name="ClassificationOutput",
    constraints=["probabilities", "sum_to_one"],
    dependencies=["input_spec"]
)
```

### SafetyProperty

Formal safety property definition.

```python
from fishstick.verification import SafetyProperty

robustness_prop = SafetyProperty(
    name="adversarial_robustness",
    property_type="robustness",
    bound=0.01,
    condition="l2_norm"
)

fairness_prop = SafetyProperty(
    name="demographic_parity",
    property_type="fairness",
    bound=0.1
)
```

### DependentlyTypedLearner

Neural network with dependent type specifications and verification.

```python
from fishstick.verification import DependentlyTypedLearner, TypeSpec
import torch.nn as nn

# Create verified model
model = DependentlyTypedLearner(
    layers=[nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)],
    input_spec=TypeSpec("Input", constraints=["normalized"]),
    output_spec=TypeSpec("Output", constraints=["logits"]),
    safety_properties=[robustness_prop]
)

# Verify robustness
certificate = model.verify_robustness(x, epsilon=0.01)
print(f"Verified: {certificate.is_verified}")
print(f"Certified radius: {certificate.details['certified_radius']}")

# Verify fairness
fairness_cert = model.verify_fairness(x, sensitive_attr, epsilon=0.1)

# Get Lipschitz constant
L = model.lipschitz_constant
```

### VerificationPipeline

Automated verification pipeline for multiple properties.

```python
from fishstick.verification import VerificationPipeline, SafetyProperty

pipeline = VerificationPipeline(
    model=verified_model,
    properties=[
        SafetyProperty("robustness", "robustness", bound=0.01),
        SafetyProperty("fairness", "fairness", bound=0.1)
    ]
)

# Run all verifications
results = pipeline.verify_all(x, epsilon=0.01, sensitive_attr=attr)

# Generate report
report = pipeline.generate_report()
print(report)
```

### LeanInterface

Interface to Lean 4 proof assistant for formal verification.

```python
from fishstick.verification import LeanInterface

lean = LeanInterface(lean_path="lean")

# Generate Lean theorems
theorem = lean.generate_robustness_theorem(
    model_name="MyModel",
    epsilon=0.01,
    lipschitz=2.5
)

fairness_theorem = lean.generate_fairness_theorem("MyModel", epsilon=0.1)

# Export to file
lean.export_to_file("proofs.lean")
```

### SMTVerifier

SMT-based verification using constraint solving.

```python
from fishstick.verification import SMTVerifier

verifier = SMTVerifier()

# Verify robustness
is_verified, counterexample = verifier.verify_robustness_smt(
    model, x, epsilon=0.01
)

if not is_verified:
    print(f"Counterexample found: {counterexample}")

# Verify output bounds
is_safe = verifier.verify_output_bounds(
    model,
    input_bounds=(0.0, 1.0),
    output_bounds=(-10.0, 10.0)
)
```

## Mathematical Background

### Lipschitz Certification

For robustness certification, use Lipschitz bound:

```
L ≤ ∏ᵢ ||Wᵢ||₂ (product of spectral norms)

Certified radius: r = margin / (2L)
```

### Dependent Types

Type signatures encode specifications:

```
CertifiedMLP : (ε : ℝ) → (K : ℝ) →
    Σ'(θ : Params),
        (∀ x, ||∇f_θ(x)|| ≤ K) ∧
        (∀ a b, |P[f_θ(x)|A=a] - P[f_θ(x)|A=b]| ≤ ε)
```

### Robustness Verification

Local robustness: ∀ x', ||x'-x|| ≤ ε ⟹ f(x') = f(x)

### Fairness Verification

Demographic parity: |P[Ŷ=1|A=0] - P[Ŷ=1|A=1]| ≤ ε

## Supported Properties

| Property | Method | Guarantees |
|----------|--------|------------|
| Robustness | Lipschitz bound | Certified radius |
| Fairness | Statistical test | Demographic parity |
| Output bounds | SMT sampling | Range constraints |

## Dependencies

- `torch` - Neural network operations
- `hashlib` - Proof hash computation
- `json` - Certificate serialization
- `dataclasses` - Data structures
- `abc` - Abstract base classes

## Usage Examples

### Full Verification Pipeline

```python
from fishstick.verification import (
    DependentlyTypedLearner,
    VerificationPipeline,
    SafetyProperty,
    LeanInterface
)
import torch.nn as nn

# Create model with specs
model = DependentlyTypedLearner(
    layers=[nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10)],
    input_spec=TypeSpec("input", constraints=["normalized"]),
    output_spec=TypeSpec("output", constraints=["logits"]),
    safety_properties=[
        SafetyProperty("robustness", "robustness", bound=0.05),
        SafetyProperty("fairness", "fairness", bound=0.1)
    ]
)

# Create pipeline
pipeline = VerificationPipeline(model)

# Verify
results = pipeline.verify_all(x, epsilon=0.05, sensitive_attr=attr)

# Generate proof certificates
lean = LeanInterface()
lean.generate_robustness_theorem("MyModel", 0.05, model.lipschitz_constant)
lean.export_to_file("verification_proofs.lean")
```

### Robustness Certification

```python
from fishstick.verification import DependentlyTypedLearner

model = DependentlyTypedLearner(layers, input_spec, output_spec)

# Certify robustness
cert = model.verify_robustness(x, epsilon=0.01)

if cert.is_verified:
    print(f"Certified up to ε={cert.details['certified_radius']}")
else:
    print(f"Failed at ε={epsilon}")
```
