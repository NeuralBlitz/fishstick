"""
Symbolic Expression Trees for Neuro-Symbolic Computation.

Implements:
- Expression tree data structures
- Differentiable expression evaluation
- Expression simplification and normalization
- Symbolic differentiation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Union, Tuple
from enum import Enum, auto
import torch
from torch import Tensor
import numpy as np


class OpType(Enum):
    """Enumeration of operation types in expression trees."""

    CONSTANT = auto()
    VARIABLE = auto()
    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    POWER = auto()
    LOG = auto()
    EXP = auto()
    SIN = auto()
    COS = auto()
    TANH = auto()
    RELU = auto()
    SIGMOID = auto()
    MATMUL = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    GT = auto()
    LT = auto()
    EQ = auto()
    IF = auto()


@dataclass
class ExpressionNode:
    """Node in a symbolic expression tree.

    Attributes:
        op_type: The operation type of this node
        value: Pre-computed constant value (for CONSTANT nodes)
        name: Variable name (for VARIABLE nodes)
        children: Child nodes
        differentiable: Whether this node participates in gradient computation
    """

    op_type: OpType
    value: Optional[float] = None
    name: Optional[str] = None
    children: List[ExpressionNode] = field(default_factory=list)
    differentiable: bool = True

    def __post_init__(self):
        """Validate node after initialization."""
        if self.op_type == OpType.CONSTANT and self.value is None:
            raise ValueError("CONSTANT nodes must have a value")
        if self.op_type == OpType.VARIABLE and self.name is None:
            raise ValueError("VARIABLE nodes must have a name")

    def to_string(self) -> str:
        """Convert expression to string representation."""
        if self.op_type == OpType.CONSTANT:
            return str(self.value)
        elif self.op_type == OpType.VARIABLE:
            return self.name
        elif len(self.children) == 1:
            if self.op_type == OpType.NOT:
                return f"¬{self.children[0].to_string()}"
            elif self.op_type == OpType.LOG:
                return f"log({self.children[0].to_string()})"
            elif self.op_type == OpType.EXP:
                return f"exp({self.children[0].to_string()})"
            elif self.op_type == OpType.SIN:
                return f"sin({self.children[0].to_string()})"
            elif self.op_type == OpType.COS:
                return f"cos({self.children[0].to_string()})"
            elif self.op_type == OpType.TANH:
                return f"tanh({self.children[0].to_string()})"
            elif self.op_type == OpType.RELU:
                return f"relu({self.children[0].to_string()})"
            elif self.op_type == OpType.SIGMOID:
                return f"sigmoid({self.children[0].to_string()})"
        elif len(self.children) == 2:
            left = self.children[0].to_string()
            right = self.children[1].to_string()
            if self.op_type == OpType.ADD:
                return f"({left} + {right})"
            elif self.op_type == OpType.SUBTRACT:
                return f"({left} - {right})"
            elif self.op_type == OpType.MULTIPLY:
                return f"({left} * {right})"
            elif self.op_type == OpType.DIVIDE:
                return f"({left} / {right})"
            elif self.op_type == OpType.POWER:
                return f"({left} ^ {right})"
            elif self.op_type == OpType.AND:
                return f"({left} ∧ {right})"
            elif self.op_type == OpType.OR:
                return f"({left} ∨ {right})"
            elif self.op_type == OpType.GT:
                return f"({left} > {right})"
            elif self.op_type == OpType.LT:
                return f"({left} < {right})"
            elif self.op_type == OpType.EQ:
                return f"({left} == {right})"
            elif self.op_type == OpType.MATMUL:
                return f"matmul({left}, {right})"
        elif len(self.children) == 3 and self.op_type == OpType.IF:
            cond = self.children[0].to_string()
            then_branch = self.children[1].to_string()
            else_branch = self.children[2].to_string()
            return f"if {cond} then {then_branch} else {else_branch}"
        return f"<op:{self.op_type.name}>"

    def count_nodes(self) -> int:
        """Count total nodes in subtree."""
        return 1 + sum(child.count_nodes() for child in self.children)

    def count_leaves(self) -> int:
        """Count leaf nodes in subtree."""
        if not self.children:
            return 1
        return sum(child.count_leaves() for child in self.children)

    def get_variables(self) -> set:
        """Get all variable names in the expression."""
        if self.op_type == OpType.VARIABLE:
            return {self.name}
        variables = set()
        for child in self.children:
            variables.update(child.get_variables())
        return variables

    def substitute(
        self, var_name: str, value: Union[float, ExpressionNode]
    ) -> ExpressionNode:
        """Substitute a variable with a value or another expression."""
        if self.op_type == OpType.VARIABLE and self.name == var_name:
            if isinstance(value, (int, float)):
                return ExpressionNode(op_type=OpType.CONSTANT, value=float(value))
            return value
        return ExpressionNode(
            op_type=self.op_type,
            value=self.value,
            name=self.name,
            children=[child.substitute(var_name, value) for child in self.children],
            differentiable=self.differentiable,
        )

    def evaluate(self, variables: Dict[str, float]) -> float:
        """Evaluate expression with given variable values."""
        if self.op_type == OpType.CONSTANT:
            return self.value
        elif self.op_type == OpType.VARIABLE:
            return variables.get(self.name, 0.0)
        elif len(self.children) == 1:
            child_val = self.children[0].evaluate(variables)
            if self.op_type == OpType.NOT:
                return not child_val
            elif self.op_type == OpType.LOG:
                return np.log(max(child_val, 1e-10))
            elif self.op_type == OpType.EXP:
                return np.exp(child_val)
            elif self.op_type == OpType.SIN:
                return np.sin(child_val)
            elif self.op_type == OpType.COS:
                return np.cos(child_val)
            elif self.op_type == OpType.TANH:
                return np.tanh(child_val)
            elif self.op_type == OpType.RELU:
                return max(0, child_val)
            elif self.op_type == OpType.SIGMOID:
                return 1 / (1 + np.exp(-child_val))
        elif len(self.children) == 2:
            left = self.children[0].evaluate(variables)
            right = self.children[1].evaluate(variables)
            if self.op_type == OpType.ADD:
                return left + right
            elif self.op_type == OpType.SUBTRACT:
                return left - right
            elif self.op_type == OpType.MULTIPLY:
                return left * right
            elif self.op_type == OpType.DIVIDE:
                return left / max(right, 1e-10)
            elif self.op_type == OpType.POWER:
                return left**right
            elif self.op_type == OpType.AND:
                return left and right
            elif self.op_type == OpType.OR:
                return left or right
            elif self.op_type == OpType.GT:
                return left > right
            elif self.op_type == OpType.LT:
                return left < right
            elif self.op_type == OpType.EQ:
                return left == right
        elif len(self.children) == 3 and self.op_type == OpType.IF:
            cond = self.children[0].evaluate(variables)
            if cond:
                return self.children[1].evaluate(variables)
            return self.children[2].evaluate(variables)
        raise ValueError(f"Cannot evaluate operation: {self.op_type}")


class DifferentiableExpression:
    """Wrapper for differentiable expression evaluation using PyTorch.

    This class converts symbolic expressions into differentiable computations
    that can be used in neural network training.
    """

    def __init__(self, expression: ExpressionNode):
        """Initialize with an expression tree.

        Args:
            expression: The root node of the expression tree
        """
        self.expression = expression
        self._build_compute_graph()

    def _build_compute_graph(self) -> None:
        """Build an ordered list of operations for computation."""
        self.operations: List[Tuple[OpType, List[Any]]] = []
        self._compile(self.expression)

    def _compile(self, node: ExpressionNode) -> int:
        """Compile expression to operations list, return node index."""
        indices = []
        for child in node.children:
            indices.append(self._compile(child))

        self.operations.append((node.op_type, indices))
        return len(self.operations) - 1

    def forward(
        self,
        variables: Dict[str, Tensor],
        constants: Optional[Dict[str, float]] = None,
    ) -> Tensor:
        """Evaluate expression with tensor inputs.

        Args:
            variables: Dictionary mapping variable names to tensors
            constants: Optional dictionary of constant values

        Returns:
            Computed tensor result
        """
        if constants is None:
            constants = {}

        values: Dict[int, Tensor] = {}

        for op_type, child_indices in self.operations:
            if op_type == OpType.CONSTANT:
                values[len(values)] = torch.tensor(
                    constants.get("value", 0.0),
                    device=next(iter(variables.values())).device,
                )
            elif op_type == OpType.VARIABLE:
                pass
            elif len(child_indices) == 1:
                child_val = values[child_indices[0]]
                if op_type == OpType.LOG:
                    values[len(values)] = torch.log(torch.clamp(child_val, min=1e-10))
                elif op_type == OpType.EXP:
                    values[len(values)] = torch.exp(child_val)
                elif op_type == OpType.SIN:
                    values[len(values)] = torch.sin(child_val)
                elif op_type == OpType.COS:
                    values[len(values)] = torch.cos(child_val)
                elif op_type == OpType.TANH:
                    values[len(values)] = torch.tanh(child_val)
                elif op_type == OpType.RELU:
                    values[len(values)] = torch.nn.functional.relu(child_val)
                elif op_type == OpType.SIGMOID:
                    values[len(values)] = torch.sigmoid(child_val)
            elif len(child_indices) == 2:
                left = values[child_indices[0]]
                right = values[child_indices[1]]
                if op_type == OpType.ADD:
                    values[len(values)] = left + right
                elif op_type == OpType.SUBTRACT:
                    values[len(values)] = left - right
                elif op_type == OpType.MULTIPLY:
                    values[len(values)] = left * right
                elif op_type == OpType.DIVIDE:
                    values[len(values)] = left / torch.clamp(right, min=1e-10)
                elif op_type == OpType.POWER:
                    values[len(values)] = torch.pow(left, right)
                elif op_type == OpType.MATMUL:
                    values[len(values)] = torch.matmul(left, right)

        return values[len(values) - 1]

    def differentiate(self, var_name: str) -> ExpressionNode:
        """Compute symbolic derivative with respect to a variable.

        Args:
            var_name: Name of the variable to differentiate with respect to

        Returns:
            Expression tree representing the derivative
        """
        return self._differentiate(self.expression, var_name)

    def _differentiate(self, node: ExpressionNode, var_name: str) -> ExpressionNode:
        """Recursively compute derivative."""
        if node.op_type == OpType.CONSTANT:
            return ExpressionNode(op_type=OpType.CONSTANT, value=0.0)

        if node.op_type == OpType.VARIABLE:
            if node.name == var_name:
                return ExpressionNode(op_type=OpType.CONSTANT, value=1.0)
            return ExpressionNode(op_type=OpType.CONSTANT, value=0.0)

        if node.op_type == OpType.ADD:
            left_deriv = self._differentiate(node.children[0], var_name)
            right_deriv = self._differentiate(node.children[1], var_name)
            return ExpressionNode(
                op_type=OpType.ADD, children=[left_deriv, right_deriv]
            )

        if node.op_type == OpType.SUBTRACT:
            left_deriv = self._differentiate(node.children[0], var_name)
            right_deriv = self._differentiate(node.children[1], var_name)
            return ExpressionNode(
                op_type=OpType.SUBTRACT, children=[left_deriv, right_deriv]
            )

        if node.op_type == OpType.MULTIPLY:
            u = node.children[0]
            v = node.children[1]
            u_deriv = self._differentiate(u, var_name)
            v_deriv = self._differentiate(v, var_name)
            return ExpressionNode(
                op_type=OpType.ADD,
                children=[
                    ExpressionNode(op_type=OpType.MULTIPLY, children=[u_deriv, v]),
                    ExpressionNode(op_type=OpType.MULTIPLY, children=[u, v_deriv]),
                ],
            )

        if node.op_type == OpType.DIVIDE:
            u = node.children[0]
            v = node.children[1]
            u_deriv = self._differentiate(u, var_name)
            v_deriv = self._differentiate(v, var_name)
            numerator = ExpressionNode(
                op_type=OpType.SUBTRACT,
                children=[
                    ExpressionNode(op_type=OpType.MULTIPLY, children=[u_deriv, v]),
                    ExpressionNode(op_type=OpType.MULTIPLY, children=[u, v_deriv]),
                ],
            )
            denominator = ExpressionNode(
                op_type=OpType.POWER,
                children=[v, ExpressionNode(op_type=OpType.CONSTANT, value=2.0)],
            )
            return ExpressionNode(
                op_type=OpType.DIVIDE, children=[numerator, denominator]
            )

        if node.op_type == OpType.POWER:
            base = node.children[0]
            exp = node.children[1]
            base_deriv = self._differentiate(base, var_name)
            if exp.op_type == OpType.CONSTANT:
                return ExpressionNode(
                    op_type=OpType.MULTIPLY,
                    children=[
                        ExpressionNode(op_type=OpType.CONSTANT, value=exp.value),
                        ExpressionNode(
                            op_type=OpType.MULTIPLY,
                            children=[
                                base_deriv,
                                ExpressionNode(
                                    op_type=OpType.POWER,
                                    children=[
                                        base,
                                        ExpressionNode(
                                            op_type=OpType.CONSTANT, value=exp.value - 1
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                )

        if node.op_type == OpType.EXP:
            child = node.children[0]
            child_deriv = self._differentiate(child, var_name)
            return ExpressionNode(
                op_type=OpType.MULTIPLY,
                children=[
                    child_deriv,
                    ExpressionNode(op_type=OpType.EXP, children=[child]),
                ],
            )

        if node.op_type == OpType.LOG:
            child = node.children[0]
            child_deriv = self._differentiate(child, var_name)
            return ExpressionNode(
                op_type=OpType.DIVIDE,
                children=[
                    child_deriv,
                    child,
                ],
            )

        return ExpressionNode(op_type=OpType.CONSTANT, value=0.0)


def simplify_expression(node: ExpressionNode) -> ExpressionNode:
    """Simplify an expression using algebraic rules.

    Args:
        node: Expression to simplify

    Returns:
        Simplified expression
    """
    if not node.children:
        return node

    new_children = [simplify_expression(child) for child in node.children]

    if node.op_type == OpType.ADD:
        const_sum = sum(c.value for c in new_children if c.op_type == OpType.CONSTANT)
        non_const = [c for c in new_children if c.op_type != OpType.CONSTANT]
        if const_sum == 0 and not non_const:
            return ExpressionNode(op_type=OpType.CONSTANT, value=0.0)
        if const_sum != 0:
            non_const.append(ExpressionNode(op_type=OpType.CONSTANT, value=const_sum))
        if len(non_const) == 1:
            return non_const[0]
        return ExpressionNode(op_type=OpType.ADD, children=non_const)

    if node.op_type == OpType.MULTIPLY:
        const_prod = 1.0
        non_const = []
        for c in new_children:
            if c.op_type == OpType.CONSTANT:
                const_prod *= c.value
            else:
                non_const.append(c)
        if const_prod == 0:
            return ExpressionNode(op_type=OpType.CONSTANT, value=0.0)
        if const_prod == 1 and not non_const:
            return ExpressionNode(op_type=OpType.CONSTANT, value=1.0)
        if const_prod != 1:
            non_const.append(ExpressionNode(op_type=OpType.CONSTANT, value=const_prod))
        if len(non_const) == 1:
            return non_const[0]
        return ExpressionNode(op_type=OpType.MULTIPLY, children=non_const)

    return ExpressionNode(
        op_type=node.op_type,
        value=node.value,
        name=node.name,
        children=new_children,
        differentiable=node.differentiable,
    )


def expression_to_function(
    node: ExpressionNode,
) -> Callable[[Dict[str, Tensor]], Tensor]:
    """Convert expression to a differentiable function.

    Args:
        node: Expression tree

    Returns:
        Differentiable function
    """
    diff_expr = DifferentiableExpression(node)

    def func(variables: Dict[str, Tensor]) -> Tensor:
        return diff_expr.forward(variables)

    return func


def parse_expression(s: str) -> ExpressionNode:
    """Parse a string representation into an expression tree.

    Args:
        s: Expression string

    Returns:
        Expression tree
    """
    s = s.replace(" ", "")

    def parse_add_sub(pos: int) -> Tuple[ExpressionNode, int]:
        left, pos = parse_mul_div(pos)
        while pos < len(s) and s[pos] in "+-":
            op = s[pos]
            right, pos = parse_mul_div(pos + 1)
            if op == "+":
                left = ExpressionNode(op_type=OpType.ADD, children=[left, right])
            else:
                left = ExpressionNode(op_type=OpType.SUBTRACT, children=[left, right])
        return left, pos

    def parse_mul_div(pos: int) -> Tuple[ExpressionNode, int]:
        left, pos = parse_unary(pos)
        while pos < len(s) and s[pos] in "*/":
            op = s[pos]
            right, pos = parse_unary(pos + 1)
            if op == "*":
                left = ExpressionNode(op_type=OpType.MULTIPLY, children=[left, right])
            else:
                left = ExpressionNode(op_type=OpType.DIVIDE, children=[left, right])
        return left, pos

    def parse_unary(pos: int) -> Tuple[ExpressionNode, int]:
        if pos < len(s) and s[pos] == "-":
            right, pos = parse_unary(pos + 1)
            return ExpressionNode(
                op_type=OpType.SUBTRACT,
                children=[ExpressionNode(op_type=OpType.CONSTANT, value=0.0), right],
            ), pos
        return parse_primary(pos)

    def parse_primary(pos: int) -> Tuple[ExpressionNode, int]:
        if s[pos] == "(":
            expr, pos = parse_add_sub(pos + 1)
            if pos < len(s) and s[pos] == ")":
                return expr, pos + 1
        if s[pos].isalpha():
            name = s[pos]
            pos += 1
            while pos < len(s) and s[pos].isalnum():
                name += s[pos]
                pos += 1
            if name == "sin":
                arg, pos = parse_primary(pos)
                return ExpressionNode(op_type=OpType.SIN, children=[arg]), pos
            if name == "cos":
                arg, pos = parse_primary(pos)
                return ExpressionNode(op_type=OpType.COS, children=[arg]), pos
            if name == "log":
                arg, pos = parse_primary(pos)
                return ExpressionNode(op_type=OpType.LOG, children=[arg]), pos
            if name == "exp":
                arg, pos = parse_primary(pos)
                return ExpressionNode(op_type=OpType.EXP, children=[arg]), pos
            if name == "tan" or name == "tanh":
                arg, pos = parse_primary(pos)
                return ExpressionNode(op_type=OpType.TANH, children=[arg]), pos
            return ExpressionNode(op_type=OpType.VARIABLE, name=name), pos
        num = ""
        while pos < len(s) and (s[pos].isdigit() or s[pos] == "."):
            num += s[pos]
            pos += 1
        if num:
            return ExpressionNode(op_type=OpType.CONSTANT, value=float(num)), pos
        raise ValueError(f"Cannot parse at position {pos}")

    expr, end_pos = parse_add_sub(0)
    return expr
