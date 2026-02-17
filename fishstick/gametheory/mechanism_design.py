"""
Mechanism design primitives.

Implements fundamental mechanism design concepts including
VCG, Myerson mechanisms, and general mechanism framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from fishstick.gametheory.core_types import Player


@dataclass
class Allocation:
    """An allocation of items to agents.
    
    Attributes:
        agent_items: Mapping of agent IDs to allocated items
        payments: Mapping of agent IDs to payments
    """
    
    agent_items: Dict[int, Any]
    payments: Dict[int, float]
    
    def get_item(self, agent_id: int) -> Any:
        """Get item allocated to an agent."""
        return self.agent_items.get(agent_id)
    
    def get_payment(self, agent_id: int) -> float:
        """Get payment from an agent."""
        return self.payments.get(agent_id, 0.0)


@dataclass 
class Preference:
    """Agent's preference over outcomes.
    
    Attributes:
        agent_id: ID of the agent
        valuation_fn: Function that maps outcomes to utilities
    """
    
    agent_id: int
    valuation_fn: Callable[[Any], float]
    
    def value(self, outcome: Any) -> float:
        """Get utility for an outcome."""
        return self.valuation_fn(outcome)
    
    def net_value(self, outcome: Any, payment: float) -> float:
        """Get net utility (value minus payment)."""
        return self.valuation_fn(outcome) - payment


@dataclass
class Mechanism(ABC):
    """Abstract base class for mechanisms.
    
    A mechanism maps reported preferences to allocations and payments.
    
    Attributes:
        agents: Set of agent IDs
        items: Set of items to allocate
    """
    
    agents: Set[int]
    items: Set[Any]
    
    @abstractmethod
    def allocate(
        self, 
        preferences: Dict[int, Preference]
    ) -> Allocation:
        """Compute allocation given preferences."""
        pass
    
    @abstractmethod
    def compute_payments(
        self, 
        preferences: Dict[int, Preference],
        allocation: Allocation
    ) -> Dict[int, float]:
        """Compute payments for each agent."""
        pass
    
    def execute(
        self, 
        preferences: Dict[int, Preference]
    ) -> Allocation:
        """Execute the mechanism."""
        allocation = self.allocate(preferences)
        payments = self.compute_payments(preferences, allocation)
        
        return Allocation(
            agent_items=allocation.agent_items,
            payments=payments,
        )


class MechanismDesign:
    """Framework for mechanism design analysis."""
    
    def __init__(
        self,
        agents: Set[int],
        items: Set[Any],
        true_preferences: Optional[Dict[int, Preference]] = None,
    ):
        self.agents = agents
        self.items = items
        self.true_preferences = true_preferences or {}
        self.reported_preferences: Dict[int, Preference] = {}
    
    def set_true_preferences(
        self, 
        preferences: Dict[int, Preference]
    ) -> None:
        """Set the true preferences of agents."""
        self.true_preferences = preferences
    
    def report_preferences(
        self, 
        reports: Dict[int, Preference]
    ) -> Allocation:
        """Process reported preferences."""
        self.reported_preferences = reports
        return None
    
    def is_truthful(
        self, 
        mechanism: Mechanism
    ) -> bool:
        """Check if a mechanism is truthful (dominant-strategy incentive compatible)."""
        for agent in self.agents:
            true_pref = self.true_preferences[agent]
            
            true_allocation = mechanism.execute(
                {agent: true_pref, **self.true_preferences}
            )
            true_utility = true_pref.net_value(
                true_allocation.get_item(agent),
                true_allocation.get_payment(agent)
            )
            
            for other_report in self.true_preferences.values():
                other_allocation = mechanism.execute(
                    {agent: other_report, **self.true_preferences}
                )
                other_utility = true_pref.net_value(
                    other_allocation.get_item(agent),
                    other_allocation.get_payment(agent)
                )
                
                if other_utility > true_utility:
                    return False
        
        return True
    
    def is_efficient(
        self, 
        mechanism: Mechanism
    ) -> bool:
        """Check if mechanism is efficient (social choice function
        maximizes sum of valuations)."""
        allocation = mechanism.execute(self.true_preferences)
        
        max_social_welfare = float('-inf')
        best_allocation = None
        
        for agent in self.agents:
            item = allocation.get_item(agent)
            if item is not None:
                value = self.true_preferences[agent].value(item)
                social_welfare = sum(
                    self.true_preferences[a].value(allocation.get_item(a))
                    for a in self.agents
                    if allocation.get_item(a) is not None
                )
                
                if social_welfare > max_social_welfare:
                    max_social_welfare = social_welfare
                    best_allocation = allocation
        
        return best_allocation is not None


@dataclass
class VCGMechanism(Mechanism):
    """Vickrey-Clarke-Groves mechanism.
    
    A truthful mechanism for allocating items and computing payments.
    The payment from an agent equals the externality they impose on others.
    """
    
    def allocate(self, preferences: Dict[int, Preference]) -> Allocation:
        """Allocate items to maximize total social welfare."""
        n = len(self.agents)
        
        if len(self.items) == 1:
            best_agent = max(
                self.agents,
                key=lambda a: preferences[a].value(list(self.items)[0])
            )
            return Allocation(
                agent_items={best_agent: list(self.items)[0]},
                payments={},
            )
        
        agent_list = list(self.agents)
        item_list = list(self.items)
        
        if len(item_list) == len(agent_list):
            cost_matrix = np.zeros((len(agent_list), len(item_list)))
            
            for i, agent in enumerate(agent_list):
                for j, item in enumerate(item_list):
                    cost_matrix[i, j] = -preferences[agent].value(item)
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            agent_items = {
                agent_list[i]: item_list[j]
                for i, j in zip(row_ind, col_ind)
            }
            
            return Allocation(agent_items=agent_items, payments={})
        
        return Allocation(agent_items={}, payments={})
    
    def compute_payments(
        self,
        preferences: Dict[int, Preference],
        allocation: Allocation
    ) -> Dict[int, float]:
        """Compute VCG payments."""
        payments = {}
        
        for agent in self.agents:
            allocated_item = allocation.get_item(agent)
            if allocated_item is None:
                payments[agent] = 0.0
                continue
            
            without_agent = {
                a: pref for a, pref in preferences.items()
                if a != agent
            }
            
            if without_agent:
                welfare_without = self._max_social_welfare(without_agent)
            else:
                welfare_without = 0.0
            
            welfare_with = preferences[agent].value(allocated_item) + sum(
                preferences[a].value(allocation.get_item(a))
                for a in self.agents
                if a != agent and allocation.get_item(a) is not None
            )
            
            payments[agent] = welfare_without - welfare_with + preferences[agent].value(allocated_item)
        
        return payments
    
    def _max_social_welfare(
        self, 
        preferences: Dict[int, Preference]
    ) -> float:
        """Compute maximum social welfare without a specific agent."""
        agent_list = list(preferences.keys())
        
        if not agent_list:
            return 0.0
        
        total = 0.0
        for agent in agent_list:
            best_value = max(
                preferences[agent].value(item)
                for item in self.items
            )
            total += best_value
        
        return total


@dataclass
class MyersonMechanism(Mechanism):
    """Myerson's mechanism for single-parameter domains.
    
    Extends any social choice function to be truthful using
    payment transformation.
    """
    
    social_choice_function: Callable[
        [Dict[int, Preference]], 
        Any
    ]
    
    def allocate(self, preferences: Dict[int, Preference]) -> Allocation:
        """Allocate using the social choice function."""
        item = self.social_choice_function(preferences)
        
        agent_items = {}
        for agent in self.agents:
            if item is not None:
                agent_items[agent] = item
        
        return Allocation(agent_items=agent_items, payments={})
    
    def compute_payments(
        self,
        preferences: Dict[int, Preference],
        allocation: Allocation
    ) -> Dict[int, float]:
        """Compute Myerson payments."""
        payments = {}
        
        for agent in self.agents:
            allocated_item = allocation.get_item(agent)
            
            def payment_integral(t: float) -> float:
                modified_prefs = dict(preferences)
                
                def mod_val(x):
                    return preferences[agent].valuation_fn(x) - t
                
                modified_prefs[agent] = Preference(agent, mod_val)
                
                outcome = self.social_choice_function(modified_prefs)
                
                if outcome == allocated_item:
                    return 0.0
                return 0.0
            
            payments[agent] = 0.0
        
        return payments


class AuctionMechanism(Mechanism):
    """Base class for auction mechanisms."""
    
    def __init__(
        self,
        agents: Set[int],
        item: Any,
        reserve_price: float = 0.0,
    ):
        super().__init__(agents, {item})
        self.item = item
        self.reserve_price = reserve_price
    
    @abstractmethod
    def allocate(self, preferences: Dict[int, Preference]) -> Allocation:
        pass
    
    @abstractmethod
    def compute_payments(
        self,
        preferences: Dict[int, Preference],
        allocation: Allocation
    ) -> Dict[int, float]:
        pass


@dataclass
class UniformPriceMechanism(AuctionMechanism):
    """Uniform price auction where winning bidders pay
    the same clearing price.
    """
    
    def allocate(self, preferences: Dict[int, Preference]) -> Allocation:
        bids = [(agent, preferences[agent].value(self.item)) 
                for agent in self.agents]
        
        bids_sorted = sorted(bids, key=lambda x: x[1], reverse=True)
        
        if not bids_sorted or bids_sorted[0][1] < self.reserve_price:
            return Allocation(agent_items={}, payments={})
        
        winner = bids_sorted[0][0]
        
        return Allocation(
            agent_items={winner: self.item},
            payments={},
        )
    
    def compute_payments(
        self,
        preferences: Dict[int, Preference],
        allocation: Allocation
    ) -> Dict[int, float]:
        winner = list(allocation.agent_items.keys())[0]
        
        other_bids = [
            preferences[a].value(self.item)
            for a in self.agents
            if a != winner
        ]
        
        if other_bids:
            price = max(other_bids)
        else:
            price = self.reserve_price
        
(price, self        price = max.reserve_price)
        
        return {winner: price}


def create_knapsack_mechanism(
    agents: Set[int],
    items: Set[Any],
    values: Dict[int, Callable[[Any], float]],
    capacities: Dict[int, float],
) -> Mechanism:
    """Create a knapsack auction mechanism."""
    
    def allocate(prefs: Dict[int, Preference]) -> Allocation:
        agent_list = list(agents)
        
        item_values = [(item, sum(
            values[a](item) for a in agent_list
        )) for item in items]
        
        item_values.sort(key=lambda x: x[1], reverse=True)
        
        agent_items = {}
        current_caps = dict(capacities)
        
        for item, total_value in item_values:
            for agent in agent_list:
                if agent in agent_items:
                    continue
                
                if current_caps[agent] > 0:
                    agent_items[agent] = item
                    break
        
        return Allocation(agent_items=agent_items, payments={})
    
    return Mechanism(agents, items, allocate, lambda p, a: {})
