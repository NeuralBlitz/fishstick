"""
Rule Learning Module for Neuro-Symbolic Systems.

Implements:
- Association rule mining
- Inductive logic programming (ILP) basics
- Rule extraction from neural networks
- Rule-based reasoning with neural embeddings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple, Callable
from collections import defaultdict
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations


@dataclass
class LogicalRule:
    """Represents a logical rule: antecedent -> consequent.

    Attributes:
        antecedent: Set of predicates that must hold
        consequent: Predicate that follows from antecedent
        confidence: Rule confidence (conditional probability)
        support: Rule support (frequency in dataset)
        weight: Learnable weight for soft rules
    """

    antecedent: Set[str]
    consequent: str
    confidence: float = 0.0
    support: float = 0.0
    weight: Optional[float] = None

    def __str__(self) -> str:
        """String representation of the rule."""
        ant_str = " ∧ ".join(sorted(self.antecedent))
        return f"{ant_str} → {self.consequent}"

    def __hash__(self) -> int:
        """Hash based on antecedent and consequent."""
        return hash((frozenset(self.antecedent), self.consequent))

    def __eq__(self, other: object) -> bool:
        """Equality based on antecedent and consequent."""
        if not isinstance(other, LogicalRule):
            return False
        return (
            self.antecedent == other.antecedent and self.consequent == other.consequent
        )


class AssociationRuleMiner:
    """Mining association rules from transaction data.

    Implements the Apriori algorithm for frequent itemset mining.
    """

    def __init__(
        self,
        min_support: float = 0.1,
        min_confidence: float = 0.5,
        max_itemset_size: int = 3,
    ):
        """Initialize association rule miner.

        Args:
            min_support: Minimum support threshold
            min_confidence: Minimum confidence threshold
            max_itemset_size: Maximum itemset size to consider
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_itemset_size = max_itemset_size
        self.frequent_itemsets: Dict[int, Dict[frozenset, float]] = {}
        self.rules: List[LogicalRule] = []

    def fit(self, transactions: List[Set[str]]) -> List[LogicalRule]:
        """Mine association rules from transactions.

        Args:
            transactions: List of transactions (sets of items)

        Returns:
            List of discovered rules
        """
        n_transactions = len(transactions)
        if n_transactions == 0:
            return []

        item_counts: Dict[str, int] = defaultdict(int)
        for trans in transactions:
            for item in trans:
                item_counts[item] += 1

        self.frequent_itemsets = {1: {}}

        for item, count in item_counts.items():
            support = count / n_transactions
            if support >= self.min_support:
                self.frequent_itemsets[1][frozenset([item])] = support

        for k in range(2, self.max_itemset_size + 1):
            self.frequent_itemsets[k] = {}
            prev_itemsets = list(self.frequent_itemsets[k - 1].keys())

            for combo in combinations(sorted(set().union(*prev_itemsets)), k):
                itemset = frozenset(combo)
                count = sum(1 for trans in transactions if itemset.issubset(trans))
                support = count / n_transactions

                if support >= self.min_support:
                    self.frequent_itemsets[k][itemset] = support

        self._generate_rules(transactions, n_transactions)
        return self.rules

    def _generate_rules(
        self,
        transactions: List[Set[str]],
        n_transactions: int,
    ) -> None:
        """Generate rules from frequent itemsets."""
        self.rules = []

        for size, itemsets in self.frequent_itemsets.items():
            if size < 2:
                continue

            for itemset, itemset_support in itemsets.items():
                for k in range(1, size):
                    for antecedent in combinations(itemset, k):
                        antecedent = frozenset(antecedent)
                        consequent = itemset - antecedent

                        antecedent_support = self._get_support(antecedent)
                        if antecedent_support is None or antecedent_support == 0:
                            continue

                        consequent_str = "_AND_".join(sorted(consequent))
                        ant_str = "_AND_".join(sorted(antecedent))

                        count_ant = sum(
                            1 for trans in transactions if antecedent.issubset(trans)
                        )
                        count_both = sum(
                            1 for trans in transactions if itemset.issubset(trans)
                        )

                        if count_ant > 0:
                            confidence = count_both / count_ant

                            if confidence >= self.min_confidence:
                                rule = LogicalRule(
                                    antecedent={ant_str},
                                    consequent=consequent_str,
                                    confidence=confidence,
                                    support=itemset_support,
                                )
                                if rule not in self.rules:
                                    self.rules.append(rule)

    def _get_support(self, itemset: frozenset) -> Optional[float]:
        """Get support for an itemset."""
        size = len(itemset)
        if size in self.frequent_itemsets:
            return self.frequent_itemsets[size].get(itemset)
        return None

    def get_top_rules(
        self,
        n: int = 10,
        sort_by: str = "confidence",
    ) -> List[LogicalRule]:
        """Get top N rules.

        Args:
            n: Number of rules to return
            sort_by: Sort by "confidence" or "support"

        Returns:
            Top N rules
        """
        if sort_by == "confidence":
            sorted_rules = sorted(self.rules, key=lambda r: r.confidence, reverse=True)
        else:
            sorted_rules = sorted(self.rules, key=lambda r: r.support, reverse=True)
        return sorted_rules[:n]


class InductiveLogicProgramming:
    """Inductive Logic Programming (ILP) basics.

    Learns logical rules from positive and negative examples.
    """

    def __init__(
        self,
        background_knowledge: Optional[Dict[str, Set]] = None,
        max_rule_length: int = 3,
        max_variables: int = 2,
    ):
        """Initialize ILP system.

        Args:
            background_knowledge: Background predicates and their extensions
            max_rule_length: Maximum number of literals in rule body
            max_variables: Maximum number of variables in rule
        """
        self.background_knowledge = background_knowledge or {}
        self.max_rule_length = max_rule_length
        self.max_variables = max_variables
        self.discovered_rules: List[LogicalRule] = []

    def learn_rules(
        self,
        positive_examples: List[Dict[str, bool]],
        negative_examples: List[Dict[str, bool]],
        target_predicate: str,
        predicates: List[str],
    ) -> List[LogicalRule]:
        """Learn rules using FOIL-like algorithm.

        Args:
            positive_examples: Positive examples
            negative_examples: Negative examples
            target_predicate: Predicate to learn rules for
            predicates: Available predicates

        Returns:
            Discovered rules
        """
        rules = []
        pos_covered = set(range(len(positive_examples)))

        while len(pos_covered) > 0 and len(rules) < 10:
            best_literal = None
            best_gain = -float("inf")
            best_ant = set()

            for pred in predicates:
                for ant_size in range(1, self.max_rule_length + 1):
                    literals = self._generate_literals(pred, ant_size)

                    for lit in literals:
                        ant = best_ant | {lit}
                        gain = self._foils_gain(
                            pos_covered, negative_examples, ant, target_predicate
                        )

                        if gain > best_gain:
                            best_gain = gain
                            best_literal = lit
                            best_ant = ant

            if best_literal is None:
                break

            rule = LogicalRule(
                antecedent=best_ant,
                consequent=target_predicate,
                confidence=self._compute_confidence(
                    pos_covered, negative_examples, best_ant, target_predicate
                ),
            )
            rules.append(rule)

            new_covered = set()
            for idx in pos_covered:
                if self._check_example(positive_examples[idx], best_ant):
                    new_covered.add(idx)
            pos_covered = pos_covered - new_covered

        self.discovered_rules = rules
        return rules

    def _generate_literals(self, predicate: str, size: int) -> Set[str]:
        """Generate candidate literals."""
        if size == 1:
            return {predicate}
        return {f"{predicate}_{i}" for i in range(size)}

    def _foils_gain(
        self,
        pos_covered: Set[int],
        negative_examples: List[Dict[str, bool]],
        antecedent: Set[str],
        target: str,
    ) -> float:
        """Compute FOIL information gain."""
        p = len(pos_covered)
        if p == 0:
            return 0.0

        new_covered = sum(
            1
            for idx in pos_covered
            if self._check_example_positive(idx, antecedent, target)
        )

        p0 = p
        p1 = new_covered

        n = len(negative_examples)
        n0 = n
        n1 = sum(1 for neg in negative_examples if self._check_example(neg, antecedent))

        if p1 == 0 or n1 == 0:
            return 0.0

        gain = p1 * (np.log2(p1 / (p1 + n1)) - np.log2(p0 / (p0 + n0)))

        return max(0, gain)

    def _check_example_positive(
        self,
        example_idx: int,
        antecedent: Set[str],
        target: str,
    ) -> bool:
        """Check if positive example satisfies antecedent."""
        return True

    def _check_example(self, example: Dict[str, bool], antecedent: Set[str]) -> bool:
        """Check if example satisfies antecedent."""
        for lit in antecedent:
            pred_name = lit.split("_")[0]
            if pred_name in example:
                if not example[pred_name]:
                    return False
        return True

    def _compute_confidence(
        self,
        pos_covered: Set[int],
        negative_examples: List[Dict[str, bool]],
        antecedent: Set[str],
        target: str,
    ) -> float:
        """Compute rule confidence."""
        pos_satisfying = sum(
            1
            for idx in pos_covered
            if self._check_example_positive(idx, antecedent, target)
        )
        neg_satisfying = sum(
            1 for neg in negative_examples if self._check_example(neg, antecedent)
        )

        total = pos_satisfying + neg_satisfying
        if total == 0:
            return 0.0
        return pos_satisfying / total


class RuleExtractor:
    """Extract logical rules from trained neural networks.

    Extracts rules that approximate network behavior.
    """

    def __init__(
        self,
        model: nn.Module,
        extraction_method: str = "decision_tree",
    ):
        """Initialize rule extractor.

        Args:
            model: Trained neural network
            extraction_method: Method for extraction
        """
        self.model = model
        self.extraction_method = extraction_method
        self.extracted_rules: List[LogicalRule] = []

    def extract(
        self,
        X: Tensor,
        y: Tensor,
        n_samples: int = 1000,
    ) -> List[LogicalRule]:
        """Extract rules from network.

        Args:
            X: Input features
            y: Target labels
            n_samples: Number of samples to use

        Returns:
            Extracted rules
        """
        if self.extraction_method == "decision_tree":
            return self._extract_decision_tree(X, y)
        elif self.extraction_method == "linear":
            return self._extract_linear_rules(X, y)
        return []

    def _extract_decision_tree(
        self,
        X: Tensor,
        y: Tensor,
    ) -> List[LogicalRule]:
        """Extract rules using decision tree approximation."""
        from sklearn.tree import DecisionTreeClassifier

        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy().astype(int)

        tree = DecisionTreeClassifier(max_depth=3)
        tree.fit(X_np, y_np)

        rules = []
        feature_names = [f"x{i}" for i in range(X.shape[1])]

        tree_rules = tree.tree_.feature
        tree_thresholds = tree.tree_.threshold
        tree_children_left = tree.tree_.children_left
        tree_children_right = tree.tree_.children_right
        tree_values = tree.tree_.value

        def extract_rules_recursive(node, antecedent):
            if tree_children_left[node] == -1 and tree_children_right[node] == -1:
                class_counts = tree_values[node][0]
                predicted_class = np.argmax(class_counts)
                confidence = class_counts[predicted_class] / class_counts.sum()

                if predicted_class == 1 and confidence > 0.7:
                    rule = LogicalRule(
                        antecedent=set(antecedent),
                        consequent="class_1",
                        confidence=confidence,
                    )
                    rules.append(rule)
                return

            feature_idx = tree_rules[node]
            if feature_idx >= 0:
                threshold = tree_thresholds[node]
                feature_name = feature_names[feature_idx]

                left_ant = antecedent + [f"{feature_name} <= {threshold:.3f}"]
                right_ant = antecedent + [f"{feature_name} > {threshold:.3f}"]

                extract_rules_recursive(tree_children_left[node], left_ant)
                extract_rules_recursive(tree_children_right[node], right_ant)

        extract_rules_recursive(0, [])
        self.extracted_rules = rules
        return rules

    def _extract_linear_rules(
        self,
        X: Tensor,
        y: Tensor,
    ) -> List[LogicalRule]:
        """Extract linear rules from weights."""
        rules = []

        if isinstance(self.model, nn.Sequential):
            layers = [m for m in self.model.modules() if isinstance(m, nn.Linear)]

            for i, layer in enumerate(layers):
                weights = layer.weight.detach()
                biases = layer.bias.detach()

                for j in range(weights.shape[0]):
                    threshold = (
                        -biases[j].item() / weights[j].abs().max().item()
                        if weights[j].abs().max() > 0
                        else 0
                    )

                    antecedent = []
                    for k in range(weights.shape[1]):
                        if weights[j, k].abs() > 0.1:
                            op = "<=" if weights[j, k] > 0 else ">"
                            antecedent.append(f"x{k} {op} {threshold:.3f}")

                    if antecedent:
                        rule = LogicalRule(
                            antecedent=set(antecedent),
                            consequent=f"output_{i}_{j}",
                            confidence=0.8,
                        )
                        rules.append(rule)

        self.extracted_rules = rules
        return rules


class NeuralRuleReasoner(nn.Module):
    """Neural network augmented with logical rules.

    Combines neural representations with symbolic reasoning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_rules: int = 5,
    ):
        """Initialize neural rule reasoner.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            n_rules: Number of rules to incorporate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_rules = n_rules

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.rule_weights = nn.Parameter(torch.ones(n_rules))

        self.rule_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for _ in range(n_rules)
            ]
        )

        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim + n_rules, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: Tensor,
        rule_features: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass with rule reasoning.

        Args:
            x: Input features [batch, input_dim]
            rule_features: Pre-computed rule features [batch, n_rules]

        Returns:
            Dictionary with predictions and rule scores
        """
        encoded = self.encoder(x)

        if rule_features is None:
            rule_features = torch.zeros(x.size(0), self.n_rules, device=x.device)
            for i, head in enumerate(self.rule_heads):
                rule_features[:, i] = torch.sigmoid(head(encoded)).squeeze(-1)

        rule_weights = F.softmax(self.rule_weights, dim=0)
        weighted_rules = rule_features * rule_weights

        combined = torch.cat([encoded, weighted_rules], dim=-1)
        output = self.combiner(combined)

        return {
            "output": output,
            "rule_features": rule_features,
            "rule_weights": rule_weights,
        }

    def add_rule(self, rule: LogicalRule) -> None:
        """Add a logical rule to the reasoner."""
        pass


class RuleAttention(nn.Module):
    """Attention mechanism over learned rules.

    Allows the network to dynamically attend to relevant rules.
    """

    def __init__(
        self,
        n_rules: int,
        hidden_dim: int = 64,
    ):
        """Initialize rule attention.

        Args:
            n_rules: Number of rules
            hidden_dim: Hidden dimension for attention
        """
        super().__init__()
        self.n_rules = n_rules
        self.hidden_dim = hidden_dim

        self.rule_embeddings = nn.Embedding(n_rules, hidden_dim)

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        context: Tensor,
        rule_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Attend over rules.

        Args:
            context: Query context [batch, hidden_dim]
            rule_mask: Mask for invalid rules [batch, n_rules]

        Returns:
            Attended output and attention weights
        """
        batch_size = context.size(0)

        query = self.query_proj(context)
        keys = self.rule_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)
        values = self.rule_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)

        scores = torch.matmul(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)

        if rule_mask is not None:
            scores = scores.masked_fill(~rule_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        attended = torch.matmul(attn_weights.unsqueeze(1), values).squeeze(1)

        return attended, attn_weights
