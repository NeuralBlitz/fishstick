# Neurosymbolic

Differentiable logic, Logic Tensor Networks, and symbolic reasoning.

## Installation

```bash
pip install fishstick[neurosymbolic]
```

## Overview

The `neurosymbolic` module provides tools for combining neural networks with symbolic reasoning including differentiable logic and Logic Tensor Networks.

## Usage

```python
from fishstick.neurosymbolic import SoftAnd, SoftOr, SoftNot, LogicTensorNetwork

# Soft logic gates
and_gate = SoftAnd()
or_gate = SoftOr()
not_gate = SoftNot()

output = and_gate(tensor1, tensor2)

# Logic Tensor Networks
ltn = LogicTensorNetwork(
    semantics=ltn.fuzzy,
    initializers=ltn.glorot_uniform
)
ltn.add_variable("x", shape=(10,))
ltn.add_formula("forall x: P(x) -> Q(x)")
```

## Classes

| Class | Description |
|-------|-------------|
| `SoftAnd` | Differentiable AND gate |
| `SoftOr` | Differentiable OR gate |
| `SoftNot` | Differentiable NOT gate |
| `LogicTensorNetwork` | Logic Tensor Network |
| `DifferentiableSAT` | Differentiable SAT solver |

## Examples

See `examples/neurosymbolic/` for complete examples.
