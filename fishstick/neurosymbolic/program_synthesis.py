"""
Program Synthesis Primitives for Neuro-Symbolic Systems.

Implements:
- Program AST representations
- Neural program interpreter
- Program skeleton spaces
- Differentiable program search
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple, Callable, Any, Union
from enum import Enum, auto
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class ProgramOp(Enum):
    """Operations available in synthesized programs."""

    INPUT = auto()
    CONSTANT = auto()
    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    POWER = auto()
    LOG = auto()
    EXP = auto()
    SIN = auto()
    COS = auto()
    IF = auto()
    LESS = auto()
    GREATER = auto()
    EQUAL = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    APPEND = auto()
    MAP = auto()
    FOLD = auto()
    FILTER = auto()
    GET = auto()
    LEN = auto()
    INDEX = auto()


@dataclass
class ProgramNode:
    """Node in a program AST.

    Attributes:
        operation: The operation at this node
        args: Arguments (constants or child indices)
        children: Child nodes
        output_type: Output type of this node
    """

    operation: ProgramOp
    args: List[Any] = field(default_factory=list)
    children: List[int] = field(default_factory=list)
    output_type: str = "float"

    def __str__(self) -> str:
        """String representation."""
        if self.operation == ProgramOp.INPUT:
            return f"input({self.args[0]})"
        elif self.operation == ProgramOp.CONSTANT:
            return f"const({self.args[0]})"
        elif self.operation == ProgramOp.IF:
            return "if"
        elif len(self.children) == 0:
            return self.operation.name.lower()
        elif len(self.children) == 1:
            return f"({self.operation.name.lower()} child_{self.children[0]})"
        elif len(self.children) == 2:
            return f"({self.operation.name.lower()} child_{self.children[0]} child_{self.children[1]})"
        return f"({self.operation.name.lower()} ...)"


@dataclass
class Program:
    """A synthesized program.

    Attributes:
        nodes: List of nodes in the program
        inputs: Expected input types
        output_type: Expected output type
        fitness: Fitness score (for search)
    """

    nodes: List[ProgramNode] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    output_type: str = "float"
    fitness: float = 0.0

    def __str__(self) -> str:
        """String representation of program."""
        lines = [f"Program(inputs={self.inputs}, output={self.output_type}):"]
        for i, node in enumerate(self.nodes):
            lines.append(f"  {i}: {node}")
        return "\n".join(lines)

    def execute(self, input_values: List[Tensor]) -> Any:
        """Execute the program.

        Args:
            input_values: List of input tensors

        Returns:
            Program output
        """
        stack: List[Any] = []

        for node in self.nodes:
            if node.operation == ProgramOp.INPUT:
                idx = node.args[0]
                stack.append(input_values[idx] if idx < len(input_values) else None)

            elif node.operation == ProgramOp.CONSTANT:
                stack.append(torch.tensor(node.args[0]))

            elif node.operation == ProgramOp.ADD:
                b = stack.pop()
                a = stack.pop()
                stack.append(a + b)

            elif node.operation == ProgramOp.SUBTRACT:
                b = stack.pop()
                a = stack.pop()
                stack.append(a - b)

            elif node.operation == ProgramOp.MULTIPLY:
                b = stack.pop()
                a = stack.pop()
                stack.append(a * b)

            elif node.operation == ProgramOp.DIVIDE:
                b = stack.pop()
                a = stack.pop()
                stack.append(a / (b + 1e-10))

            elif node.operation == ProgramOp.POWER:
                b = stack.pop()
                a = stack.pop()
                stack.append(a**b)

            elif node.operation == ProgramOp.LOG:
                a = stack.pop()
                stack.append(torch.log(torch.clamp(a, min=1e-10)))

            elif node.operation == ProgramOp.EXP:
                a = stack.pop()
                stack.append(torch.exp(a))

            elif node.operation == ProgramOp.SIN:
                a = stack.pop()
                stack.append(torch.sin(a))

            elif node.operation == ProgramOp.COS:
                a = stack.pop()
                stack.append(torch.cos(a))

            elif node.operation == ProgramOp.LESS:
                b = stack.pop()
                a = stack.pop()
                stack.append((a < b).float())

            elif node.operation == ProgramOp.GREATER:
                b = stack.pop()
                a = stack.pop()
                stack.append((a > b).float())

            elif node.operation == ProgramOp.EQUAL:
                b = stack.pop()
                a = stack.pop()
                stack.append((a == b).float())

            elif node.operation == ProgramOp.AND:
                b = stack.pop()
                a = stack.pop()
                stack.append(torch.min(a, b))

            elif node.operation == ProgramOp.OR:
                b = stack.pop()
                a = stack.pop()
                stack.append(torch.max(a, b))

            elif node.operation == ProgramOp.NOT:
                a = stack.pop()
                stack.append(1.0 - a)

            elif node.operation == ProgramOp.IF:
                else_val = stack.pop()
                if_val = stack.pop()
                cond = stack.pop()
                stack.append(torch.where(cond > 0.5, if_val, else_val))

            elif node.operation == ProgramOp.GET:
                idx = node.args[0]
                stack.append(stack[-1][:, idx])

            elif node.operation == ProgramOp.LEN:
                stack.append(torch.tensor(stack[-1].shape[-1]).float())

        return stack[-1] if stack else None


class ProgramSpace:
    """Space of possible programs for search.

    Defines the grammar and constraints for program synthesis.
    """

    def __init__(
        self,
        max_length: int = 20,
        max_depth: int = 5,
        available_ops: Optional[List[ProgramOp]] = None,
    ):
        """Initialize program space.

        Args:
            max_length: Maximum number of nodes
            max_depth: Maximum depth of AST
            available_ops: List of available operations
        """
        self.max_length = max_length
        self.max_depth = max_depth

        if available_ops is None:
            self.available_ops = [
                ProgramOp.INPUT,
                ProgramOp.CONSTANT,
                ProgramOp.ADD,
                ProgramOp.SUBTRACT,
                ProgramOp.MULTIPLY,
                ProgramOp.DIVIDE,
                ProgramOp.POWER,
                ProgramOp.LOG,
                ProgramOp.SIN,
                ProgramOp.COS,
                ProgramOp.LESS,
                ProgramOp.GREATER,
                ProgramOp.EQUAL,
                ProgramOp.AND,
                ProgramOp.OR,
                ProgramOp.NOT,
                ProgramOp.IF,
            ]
        else:
            self.available_ops = available_ops

        self.unary_ops = {
            ProgramOp.NOT,
            ProgramOp.LOG,
            ProgramOp.EXP,
            ProgramOp.SIN,
            ProgramOp.COS,
            ProgramOp.LEN,
        }
        self.binary_ops = {
            ProgramOp.ADD,
            ProgramOp.SUBTRACT,
            ProgramOp.MULTIPLY,
            ProgramOp.DIVIDE,
            ProgramOp.POWER,
            ProgramOp.LESS,
            ProgramOp.GREATER,
            ProgramOp.EQUAL,
            ProgramOp.AND,
            ProgramOp.OR,
        }
        self.ternary_ops = {ProgramOp.IF}

    def random_program(self, n_inputs: int = 1) -> Program:
        """Generate a random program from the space.

        Args:
            n_inputs: Number of input variables

        Returns:
            Random program
        """
        length = np.random.randint(3, self.max_length + 1)
        nodes = []

        for i in range(length):
            op = np.random.choice(self.available_ops)

            node = ProgramNode(operation=op)

            if op == ProgramOp.INPUT:
                node.args = [np.random.randint(0, n_inputs)]
                node.output_type = "float"

            elif op == ProgramOp.CONSTANT:
                node.args = [np.random.uniform(-5, 5)]
                node.output_type = "float"

            elif op in self.unary_ops:
                node.output_type = "float"

            elif op in self.binary_ops:
                node.output_type = "float"

            elif op in self.ternary_ops:
                node.output_type = "float"

            if op == ProgramOp.GET:
                node.args = [0]

            nodes.append(node)

        return Program(nodes=nodes, inputs=[f"x{i}" for i in range(n_inputs)])


class NeuralProgramInterpreter(nn.Module):
    """Neural network that interprets programs.

    Learns to execute programs from examples.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ):
        """Initialize neural program interpreter.

        Args:
            vocab_size: Size of operation vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension
            n_layers: Number of RNN layers
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.operation_embedding = nn.Embedding(vocab_size, embed_dim)
        self.argument_embedding = nn.Embedding(100, embed_dim)

        self.encoder = nn.LSTM(
            embed_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
        )

        self.state_encoder = nn.Linear(hidden_dim * 2, hidden_dim)

        self.decoder = nn.LSTM(
            embed_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
        )

        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        program_ids: Tensor,
        program_args: Tensor,
        hidden_state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Execute program step.

        Args:
            program_ids: Operation IDs [batch, seq_len]
            program_args: Operation arguments [batch, seq_len]
            hidden_state: Optional LSTM hidden state

        Returns:
            Output logits and new hidden state
        """
        op_embeds = self.operation_embedding(program_ids)
        arg_embeds = self.argument_embedding(program_args)

        embeds = op_embeds + arg_embeds

        if hidden_state is None:
            encoded, hidden_state = self.encoder(embeds)
        else:
            encoded, hidden_state = self.encoder(embeds, hidden_state)

        state = self.state_encoder(encoded)
        output = self.output_proj(state)

        return output, hidden_state


class DifferentiableProgramSearch(nn.Module):
    """Differentiable program search using neural networks.

    Uses continuous relaxation for discrete program space.
    """

    def __init__(
        self,
        n_ops: int,
        max_length: int = 20,
        embed_dim: int = 64,
    ):
        """Initialize differentiable program search.

        Args:
            n_ops: Number of available operations
            max_length: Maximum program length
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.n_ops = n_ops
        self.max_length = max_length
        self.embed_dim = embed_dim

        self.operation_logits = nn.Parameter(torch.randn(max_length, n_ops))

        self.step_embedding = nn.Embedding(max_length, embed_dim)

        self.program_encoder = nn.LSTM(
            embed_dim,
            embed_dim,
            batch_first=True,
        )

    def forward(
        self,
        n_programs: int = 1,
        temperature: float = 1.0,
    ) -> Tensor:
        """Sample programs.

        Args:
            n_programs: Number of programs to sample
            temperature: Sampling temperature

        Returns:
            Program operation sequences [n_programs, max_length]
        """
        step_ids = torch.arange(self.max_length, device=self.operation_logits.device)
        step_embeds = self.step_embedding(step_ids)

        program_embeds = step_embeds.unsqueeze(0).expand(n_programs, -1, -1)

        encoded, _ = self.program_encoder(program_embeds)

        logits = self.operation_logits.unsqueeze(0).expand(n_programs, -1, -1)
        logits = logits + encoded

        ops = F.gumbel_softmax(logits.view(n_programs, -1), tau=temperature, hard=True)
        ops = ops.view(n_programs, self.max_length, self.n_ops)

        return ops.argmax(dim=-1)

    def get_best_program(self) -> Tensor:
        """Get the highest probability program.

        Returns:
            Operation sequence
        """
        return self.operation_logits.argmax(dim=-1)

    def program_loss(
        self,
        program_ids: Tensor,
        rewards: Tensor,
    ) -> Tensor:
        """Compute loss for program search.

        Args:
            program_ids: Sampled program IDs [batch, length]
            rewards: Reward for each program [batch]

        Returns:
            Loss value
        """
        logits = self.operation_logits.unsqueeze(0).expand(program_ids.size(0), -1, -1)

        log_probs = F.log_softmax(logits, dim=-1)

        target = program_ids.unsqueeze(-1).expand(-1, -1, self.n_ops)
        selected_log_probs = log_probs.gather(2, target).squeeze(-1)

        loss = -(selected_log_probs.mean(dim=-1) * rewards).mean()

        return loss


class ProgramSketch:
    """Program sketch with holes to fill.

    Defines partial program structure for guided synthesis.
    """

    def __init__(
        self,
        sketch: List[Optional[ProgramOp]],
    ):
        """Initialize program sketch.

        Args:
            sketch: List of operations (None = hole to fill)
        """
        self.sketch = sketch
        self.n_holes = sum(1 for op in sketch if op is None)
        self.filled: List[ProgramOp] = []

    def fill(self, operations: List[ProgramOp]) -> Program:
        """Fill holes with operations.

        Args:
            operations: Operations to fill

        Returns:
            Complete program
        """
        if len(operations) != self.n_holes:
            raise ValueError(
                f"Expected {self.n_holes} operations, got {len(operations)}"
            )

        nodes = []
        op_idx = 0

        for op in self.sketch:
            if op is None:
                nodes.append(ProgramNode(operation=operations[op_idx]))
                op_idx += 1
            else:
                nodes.append(ProgramNode(operation=op))

        return Program(nodes=nodes)


class ProgramEvaluator:
    """Evaluates programs on examples.

    Computes fitness for program search.
    """

    def __init__(
        self,
        program_space: ProgramSpace,
    ):
        """Initialize program evaluator.

        Args:
            program_space: Program search space
        """
        self.program_space = program_space

    def evaluate(
        self,
        program: Program,
        inputs: List[Tensor],
        targets: Tensor,
        loss_fn: Optional[Callable] = None,
    ) -> float:
        """Evaluate program on examples.

        Args:
            program: Program to evaluate
            inputs: Input examples
            targets: Target outputs
            loss_fn: Loss function (default: MSE)

        Returns:
            Fitness score
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        try:
            outputs = program.execute(inputs)

            if outputs is None:
                return -float("inf")

            loss = loss_fn(outputs, targets)

            fitness = -loss.item()

            if not torch.isfinite(torch.tensor(fitness)):
                return -float("inf")

            return fitness

        except Exception:
            return -float("inf")

    def batch_evaluate(
        self,
        programs: List[Program],
        inputs: List[Tensor],
        targets: Tensor,
    ) -> List[float]:
        """Evaluate multiple programs.

        Args:
            programs: Programs to evaluate
            inputs: Input examples
            targets: Target outputs

        Returns:
            List of fitness scores
        """
        return [self.evaluate(p, inputs, targets) for p in programs]


class ProgramSynthesis:
    """Complete program synthesis system.

    Combines search, evaluation, and optimization.
    """

    def __init__(
        self,
        n_inputs: int,
        max_length: int = 20,
        population_size: int = 100,
        n_generations: int = 50,
    ):
        """Initialize program synthesis.

        Args:
            n_inputs: Number of input variables
            max_length: Maximum program length
            population_size: Population size for genetic search
            n_generations: Number of generations
        """
        self.n_inputs = n_inputs
        self.max_length = max_length
        self.population_size = population_size
        self.n_generations = n_generations

        self.program_space = ProgramSpace(max_length=max_length)
        self.evaluator = ProgramEvaluator(self.program_space)

    def synthesize(
        self,
        inputs: List[Tensor],
        targets: Tensor,
        method: str = "random",
    ) -> Program:
        """Synthesize program from examples.

        Args:
            inputs: Input examples
            targets: Target outputs
            method: Search method ("random", "evolutionary")

        Returns:
            Best program found
        """
        if method == "random":
            return self._random_search(inputs, targets)
        elif method == "evolutionary":
            return self._evolutionary_search(inputs, targets)
        raise ValueError(f"Unknown method: {method}")

    def _random_search(
        self,
        inputs: List[Tensor],
        targets: Tensor,
    ) -> Program:
        """Random search for programs.

        Args:
            inputs: Input examples
            targets: Target outputs

        Returns:
            Best program found
        """
        best_program = None
        best_fitness = -float("inf")

        for _ in range(1000):
            program = self.program_space.random_program(self.n_inputs)
            fitness = self.evaluator.evaluate(program, inputs, targets)

            if fitness > best_fitness:
                best_fitness = fitness
                best_program = program

            if fitness > -0.01:
                break

        return best_program

    def _evolutionary_search(
        self,
        inputs: List[Tensor],
        targets: Tensor,
    ) -> Program:
        """Evolutionary search for programs.

        Args:
            inputs: Input examples
            targets: Target outputs

        Returns:
            Best program found
        """
        population = [
            self.program_space.random_program(self.n_inputs)
            for _ in range(self.population_size)
        ]

        best_program = None
        best_fitness = -float("inf")

        for gen in range(self.n_generations):
            fitnesses = self.evaluator.batch_evaluate(population, inputs, targets)

            sorted_programs = sorted(
                zip(population, fitnesses),
                key=lambda x: x[1],
                reverse=True,
            )

            if sorted_programs[0][1] > best_fitness:
                best_fitness = sorted_programs[0][1]
                best_program = sorted_programs[0][0]

            if best_fitness > -0.01:
                break

            parents = [p for p, _ in sorted_programs[: self.population_size // 2]]

            new_population = parents[:]

            while len(new_population) < self.population_size:
                parent = np.random.choice(parents)
                child = self._mutate(parent)
                new_population.append(child)

            population = new_population

        return best_program

    def _mutate(self, program: Program) -> Program:
        """Mutate a program.

        Args:
            program: Program to mutate

        Returns:
            Mutated program
        """
        new_nodes = [
            ProgramNode(
                operation=node.operation,
                args=node.args.copy(),
                children=node.children.copy(),
            )
            for node in program.nodes
        ]

        mutation_type = np.random.choice(["change_op", "add_node", "remove_node"])

        if mutation_type == "change_op" and new_nodes:
            idx = np.random.randint(0, len(new_nodes))
            op = np.random.choice(self.program_space.available_ops)
            new_nodes[idx].operation = op

        elif mutation_type == "add_node" and len(new_nodes) < self.max_length:
            op = np.random.choice(self.program_space.available_ops)
            new_nodes.append(ProgramNode(operation=op))

        elif mutation_type == "remove_node" and len(new_nodes) > 1:
            idx = np.random.randint(0, len(new_nodes))
            new_nodes.pop(idx)

        return Program(nodes=new_nodes, inputs=program.inputs)
