# Game Theory

Game theory, mechanism design, and multi-agent learning.

## Installation

```bash
pip install fishstick[gametheory]
```

## Overview

The `gametheory` module provides tools for game theory and multi-agent systems including normal form games, solution concepts, and auctions.

## Usage

```python
from fishstick.gametheory import NormalFormGame, NashEquilibriumSolver, ShapleyValue

# Create game
payoffs = [[3, 0], [1, 2]]
game = NormalFormGame(payoffs)

# Find Nash equilibrium
solver = NashEquilibriumSolver()
equilibrium = solver.find_nash(game)

# Compute Shapley value
shapley = ShapleyValue(game)
values = shapley.compute()
```

## Games

| Class | Description |
|-------|-------------|
| `NormalFormGame` | Normal form game |
| `ZeroSumGame` | Zero-sum game |
| `CooperativeGame` | Cooperative game |

## Solution Concepts

| Class | Description |
|-------|-------------|
| `NashEquilibriumSolver` | Nash equilibrium |
| `ShapleyValue` | Shapley value |
| `Core` | Core of a game |
| `Nucleolus` | Nucleolus |

## Mechanism Design

| Class | Description |
|-------|-------------|
| `VCGMechanism` | Vickrey-Clarke-Groves |
| `VickreyAuction` | Vickrey auction |

## Examples

See `examples/gametheory/` for complete examples.
