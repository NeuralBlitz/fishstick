# categorical - Categorical Structures Module

## Overview

The `categorical` module provides implementations of category theory concepts for compositional learning systems. It models neural network operations and data flow using categorical abstractions.

## Purpose and Scope

This module enables:
- Categorical composition of neural network layers
- Monoidal category structures for parallel composition
- Lens-based bidirectional transformations (forward/backward passes)
- Natural transformations for training dynamics

## Key Classes and Functions

### Categories (`category.py`)

#### `Object`
Object in a category with shape and constraints.

```python
from fishstick.categorical import Object

obj = Object(
    name="input",
    shape=(784,),
    constraints={"dtype": "float32"}
)
```

#### `Category`
Abstract category with objects and morphisms.

**Abstract Methods:**
- `objects()`: Return list of objects
- `morphisms(source, target)`: Return morphisms between objects
- `compose(f, g)`: Compose two morphisms g ∘ f
- `identity(obj)`: Return identity morphism

#### `MonoidalCategory`
Monoidal Category (C, ⊗, I) with tensor product structure.

```python
from fishstick.categorical import MonoidalCategory, Object

cat = MonoidalCategory(name="NeuralNet")

# Add objects
input_obj = Object(name="input", shape=(784,))
hidden_obj = Object(name="hidden", shape=(256,))
cat.add_object(input_obj)
cat.add_object(hidden_obj)

# Tensor product
combined = cat.tensor_product(input_obj, hidden_obj)
# Object with shape (784, 256)
```

#### `TracedMonoidalCategory`
Traced Symmetric Monoidal Category with trace operator for feedback loops.

```python
from fishstick.categorical import TracedMonoidalCategory

cat = TracedMonoidalCategory()

# Trace creates feedback loop
traced_fn = cat.trace(fn, dim=0)
```

#### `Functor`
Functor F: C → D between categories.

```python
from fishstick.categorical import Functor

functor = Functor(
    on_objects=lambda x: transform(x),
    on_morphisms=lambda f: transform_morphism(f)
)

# Apply functor
result = functor(input_object)

# Compose functors
composed = functor1.compose(functor2)
```

#### `NaturalTransformation`
Natural transformation α: F ⇒ G between functors.

```python
from fishstick.categorical import NaturalTransformation

transform = NaturalTransformation(
    components={"obj1": alpha1, "obj2": alpha2},
    source_functor=F,
    target_functor=G
)

# Verify naturality condition
is_natural = transform.verify_naturality("obj1", "obj2", morphism, test_input)
```

#### `DaggerCategory`
Dagger compact closed category for reversibility (backpropagation).

```python
from fishstick.categorical import DaggerCategory

cat = DaggerCategory()

# Get adjoint (backward pass)
adjoint = cat.dagger(forward_fn)

# Dual object
dual_obj = cat.dual(obj)

# Cap and cup morphisms
cap = cat.cap(obj)  # I → A* ⊗ A
cup = cat.cup(obj)  # A ⊗ A* → I
```

### Lenses (`lens.py`)

#### `Lens`
Bidirectional transformation with get and put.

```python
from fishstick.categorical import Lens

# get: forward pass
# put: backward update
lens = Lens(
    get=lambda s: s["features"],
    put=lambda s, a: {**s, "features": a}
)

# Forward
value = lens(state)

# Compose lenses
composed = lens1.compose(lens2)
```

#### `NeuralLens`
Neural network as a lens (forward + parameter update).

```python
from fishstick.categorical import NeuralLens
import torch.nn as nn

layer = nn.Linear(784, 256)
lens = NeuralLens(layer, lr=0.01)

# Forward pass
output = lens.get(input)

# Backward with update
lens.put(grad_output)
```

#### `BidirectionalLearner`
Full learning process as bidirectional dataflow.

```python
from fishstick.categorical import BidirectionalLearner, NeuralLens

learner = BidirectionalLearner([layer1, layer2, layer3], lr=0.01)

# Forward pass
output = learner(x)

# Backward pass through all lenses
learner.backward(loss)
```

#### `Optic`
Generalized lens with bidirectional state.

```python
from fishstick.categorical import Optic

optic = Optic(
    resolve=lambda s: (s["data"], s["context"]),
    core=lambda m, a: update(m, a),
    reindex=lambda m: reconstruct(m)
)
```

## Mathematical Background

### Category Theory
- Objects and morphisms form a category
- Composition is associative: h ∘ (g ∘ f) = (h ∘ g) ∘ f
- Identity: id ∘ f = f = f ∘ id

### Monoidal Categories
- Tensor product ⊗ for parallel composition
- Unit object I
- Associator and unitor natural isomorphisms

### Lenses
- Model bidirectional transformations
- get: S → A (read)
- put: S × A → S (update)
- Composition follows chain rule

## Dependencies

- `torch`: PyTorch tensors and modules
- `dataclasses`: For data structures
- `typing`: Type hints

## Usage Examples

### Composable Neural Pipeline

```python
from fishstick.categorical import MonoidalCategory, Object, Functor
import torch.nn as nn

# Define pipeline as category
cat = MonoidalCategory("Pipeline")

# Add layer objects
layers = [
    Object(name="input", shape=(784,)),
    Object(name="hidden1", shape=(256,)),
    Object(name="hidden2", shape=(128,)),
    Object(name="output", shape=(10,))
]

for layer in layers:
    cat.add_object(layer)

# Define morphisms (transformations)
cat.add_morphism("input", "hidden1", nn.Linear(784, 256))
cat.add_morphism("hidden1", "hidden2", nn.Linear(256, 128))
cat.add_morphism("hidden2", "output", nn.Linear(128, 10))

# Compose
pipeline = cat.compose(
    cat.compose(
        cat.morphisms(layers[0], layers[1])[0],
        cat.morphisms(layers[1], layers[2])[0]
    ),
    cat.morphisms(layers[2], layers[3])[0]
)
```

### Lens-Based Training

```python
from fishstick.categorical import BidirectionalLearner, NeuralLens
import torch.nn as nn

# Create layers
layers = [
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
]

# Create bidirectional learner
learner = BidirectionalLearner(layers, lr=0.001)

# Training loop
for x, y in dataloader:
    # Forward
    output = learner(x)
    
    # Compute loss
    loss = criterion(output, y)
    
    # Backward with parameter update
    learner.backward(loss)
```
