# Logic

Propositional logic and SAT solving.

## Installation

```bash
pip install fishstick[logic]
```

## Overview

The `logic` module provides propositional logic representations and SAT solving capabilities.

## Usage

```python
from fishstick.logic import Atom, And, Or, Not, Implies

# Create formulas
p, q = Atom("p"), Atom("q")
formula = And(p, Or(q, Not(p)))

# Evaluate
env = {"p": True, "q": False}
result = formula.evaluate(env)

# SAT checking
from fishstick.logic import sat_check
model = sat_check(formula)
```

## Classes

| Class | Description |
|-------|-------------|
| `Atom` | Propositional atom |
| `Formula` | Base formula class |
| `And`, `Or`, `Not` | Logical connectives |
| `Implies` | Implication |

## Functions

| Function | Description |
|----------|-------------|
| `sat_check` | Check satisfiability |
| `davis_putnam` | Davis-Putnam algorithm |
| `truth_table` | Generate truth table |
| `logical_entailment` | Check entailment |
| `logical_equivalence` | Check equivalence |

## Examples

See `examples/logic/` for complete examples.
