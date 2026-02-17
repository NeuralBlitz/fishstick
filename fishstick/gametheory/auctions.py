"""
Auction mechanisms.

Implements various auction formats including Vickrey (second-price),
Dutch, English, and first-price auctions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from enum import Enum

from fishstick.gametheory.mechanism_design import (
    Mechanism,
    Preference,
    Allocation,
)


class AuctionType(Enum):
    """Types of auctions."""

    VICKREY = "vickrey"
    DUTCH = "dutch"
    ENGLISH = "english"
    FIRST_PRICE = "first_price"
    DOUBLE = "double"


@dataclass
class Bid:
    """A bid in an auction.

    Attributes:
        agent_id: ID of the bidding agent
        amount: Bid amount
        quantity: Quantity desired (for multi-unit auctions)
    """

    agent_id: int
    amount: float
    quantity: int = 1


@dataclass
class AuctionResult:
    """Result of an auction.

    Attributes:
        winner: ID of winning agent
        payment: Amount paid by winner
        item: Item sold
        all_bids: All bids received
    """

    winner: Optional[int]
    payment: float
    item: any
    all_bids: List[Bid] = field(default_factory=list)


class Auction(ABC):
    """Abstract base class for auctions."""

    def __init__(
        self,
        item: any,
        reserve_price: float = 0.0,
        seller_id: int = -1,
    ):
        self.item = item
        self.reserve_price = reserve_price
        self.seller_id = seller_id
        self.bids: List[Bid] = []

    @abstractmethod
    def submit_bid(self, bid: Bid) -> None:
        """Submit a bid."""
        pass

    @abstractmethod
    def close(self) -> AuctionResult:
        """Close the auction and determine winner."""
        pass

    def reset(self) -> None:
        """Reset the auction."""
        self.bids = []


@dataclass
class VickreyAuction(Auction):
    """Vickrey (second-price sealed-bid) auction.

    The winner pays the second-highest bid.
    Truthful: bidding your true valuation is a dominant strategy.
    """

    def submit_bid(self, bid: Bid) -> None:
        """Submit a sealed bid."""
        self.bids.append(bid)

    def close(self) -> AuctionResult:
        """Determine winner and payment."""
        if not self.bids:
            return AuctionResult(winner=None, payment=0.0, item=self.item)

        sorted_bids = sorted(self.bids, key=lambda b: b.amount, reverse=True)

        if sorted_bids[0].amount < self.reserve_price:
            return AuctionResult(winner=None, payment=0.0, item=self.item)

        if len(sorted_bids) > 1:
            payment = max(sorted_bids[1].amount, self.reserve_price)
        else:
            payment = self.reserve_price

        return AuctionResult(
            winner=sorted_bids[0].agent_id,
            payment=payment,
            item=self.item,
            all_bids=self.bids,
        )

    def get_payment(self, agent_id: int, valuation: float) -> float:
        """Compute payment for an agent given their true valuation."""
        other_bids = [b.amount for b in self.bids if b.agent_id != agent_id]

        if not other_bids:
            return self.reserve_price

        return max(max(other_bids), self.reserve_price)


@dataclass
class DutchAuction(Auction):
    """Dutch (descending) auction.

    Price starts high and decreases until someone accepts.
    """

    def __init__(
        self,
        item: any,
        start_price: float,
        min_price: float,
        decrement: float,
        reserve_price: float = 0.0,
    ):
        super().__init__(item, reserve_price)
        self.start_price = start_price
        self.min_price = min_price
        self.decrement = decrement
        self.current_price = start_price
        self.winner: Optional[int] = None

    def submit_bid(self, bid: Bid) -> None:
        """Accept the current price."""
        if self.winner is None and bid.amount >= self.current_price:
            self.winner = bid.agent_id

    def step(self) -> float:
        """Decrease price by one step."""
        self.current_price = max(self.min_price, self.current_price - self.decrement)
        return self.current_price

    def close(self) -> AuctionResult:
        """Close auction and determine winner."""
        if self.winner is not None:
            return AuctionResult(
                winner=self.winner,
                payment=self.current_price,
                item=self.item,
                all_bids=self.bids,
            )

        return AuctionResult(winner=None, payment=0.0, item=self.item)


@dataclass
class EnglishAuction(Auction):
    """English (ascending) auction.

    Price increases until only one bidder remains.
    """

    def __init__(
        self,
        item: any,
        start_price: float = 0.0,
        increment: float = 1.0,
        reserve_price: float = 0.0,
    ):
        super().__init__(item, reserve_price)
        self.start_price = start_price
        self.increment = increment
        self.current_price = start_price
        self.active_bidders: Set[int] = set()
        self.current_bidder: Optional[int] = None
        self.is_open = True

    def submit_bid(self, bid: Bid) -> None:
        """Submit a bid (accept current price)."""
        if not self.is_open:
            return

        self.active_bidders.add(bid.agent_id)
        self.current_bidder = bid.agent_id
        self.current_price += self.increment

    def close(self) -> AuctionResult:
        """Close auction and determine winner."""
        if not self.active_bidders or self.current_price < self.reserve_price:
            return AuctionResult(winner=None, payment=0.0, item=self.item)

        if len(self.active_bidders) == 1:
            winner = list(self.active_bidders)[0]
            payment = max(self.current_price, self.reserve_price)

            return AuctionResult(
                winner=winner,
                payment=payment,
                item=self.item,
                all_bids=self.bids,
            )

        return AuctionResult(winner=None, payment=0.0, item=self.item)


@dataclass
class FirstPriceAuction(Auction):
    """First-price sealed-bid auction.

    Winner pays their own bid.
    """

    def submit_bid(self, bid: Bid) -> None:
        """Submit a sealed bid."""
        self.bids.append(bid)

    def close(self) -> AuctionResult:
        """Determine winner and payment."""
        if not self.bids:
            return AuctionResult(winner=None, payment=0.0, item=self.item)

        sorted_bids = sorted(self.bids, key=lambda b: b.amount, reverse=True)

        if sorted_bids[0].amount < self.reserve_price:
            return AuctionResult(winner=None, payment=0.0, item=self.item)

        return AuctionResult(
            winner=sorted_bids[0].agent_id,
            payment=sorted_bids[0].amount,
            item=self.item,
            all_bids=self.bids,
        )


@dataclass
class DoubleAuction(Auction):
    """Double auction for trading between buyers and sellers.

    Matches buyers and sellers to maximize trade volume.
    """

    def __init__(self, reserve_price: float = 0.0):
        super().__init__(item=None, reserve_price=reserve_price)
        self.buyer_bids: List[Bid] = []
        self.seller_bids: List[Bid] = []

    def submit_bid(self, bid: Bid) -> None:
        """Submit a bid (buyer or seller)."""
        if bid.amount > 0:
            self.buyer_bids.append(bid)
        else:
            self.seller_bids.append(bid)

    def submit_buyer_bid(self, bid: Bid) -> None:
        """Submit a buyer bid."""
        self.buyer_bids.append(bid)

    def submit_seller_ask(self, bid: Bid) -> None:
        """Submit a seller ask."""
        self.seller_bids.append(bid)

    def close(self) -> List[AuctionResult]:
        """Close auction and match buyers and sellers."""
        buyers = sorted(self.buyer_bids, key=lambda b: b.amount, reverse=True)
        sellers = sorted(self.seller_bids, key=lambda b: b.amount)

        results = []

        for buyer, seller in zip(buyers, sellers):
            if buyer.amount >= seller.amount and seller.amount >= self.reserve_price:
                price = (buyer.amount + seller.amount) / 2

                results.append(
                    AuctionResult(
                        winner=buyer.agent_id,
                        payment=price,
                        item=None,
                        all_bids=self.buyer_bids + self.seller_bids,
                    )
                )

        return results


@dataclass
class MultiUnitAuction(Auction):
    """Auction for multiple identical units.

    Attributes:
        num_units: Number of identical units for sale
    """

    def __init__(
        self,
        item: any,
        num_units: int,
        reserve_price: float = 0.0,
    ):
        super().__init__(item, reserve_price)
        self.num_units = num_units

    def submit_bid(self, bid: Bid) -> None:
        """Submit a bid for units."""
        self.bids.append(bid)

    def close(self) -> List[AuctionResult]:
        """Close auction and determine winners."""
        if not self.bids:
            return []

        sorted_bids = sorted(self.bids, key=lambda b: b.amount, reverse=True)

        winning_bids = sorted_bids[: self.num_units]

        if len(winning_bids) < self.num_units:
            return []

        clearing_price = winning_bids[-1].amount

        results = []
        for bid in winning_bids:
            if bid.amount >= self.reserve_price:
                results.append(
                    AuctionResult(
                        winner=bid.agent_id,
                        payment=clearing_price,
                        item=self.item,
                        all_bids=self.bids,
                    )
                )

        return results


def create_auction(auction_type: AuctionType, item: any, **kwargs) -> Auction:
    """Factory function to create auction instances."""

    if auction_type == AuctionType.VICKREY:
        return VickreyAuction(item, kwargs.get("reserve_price", 0.0))
    elif auction_type == AuctionType.DUTCH:
        return DutchAuction(
            item,
            kwargs.get("start_price", 100.0),
            kwargs.get("min_price", 1.0),
            kwargs.get("decrement", 1.0),
            kwargs.get("reserve_price", 0.0),
        )
    elif auction_type == AuctionType.ENGLISH:
        return EnglishAuction(
            item,
            kwargs.get("start_price", 0.0),
            kwargs.get("increment", 1.0),
            kwargs.get("reserve_price", 0.0),
        )
    elif auction_type == AuctionType.FIRST_PRICE:
        return FirstPriceAuction(item, kwargs.get("reserve_price", 0.0))
    else:
        raise ValueError(f"Unknown auction type: {auction_type}")


def simulate_auction(
    auction: Auction,
    valuations: Dict[int, float],
    num_bidders: int,
    noise_std: float = 0.0,
) -> AuctionResult:
    """Simulate an auction with bidders having given valuations.

    Args:
        auction: The auction mechanism
        valuations: True valuations for each bidder
        num_bidders: Number of bidders
        noise_std: Standard deviation of noise in bids

    Returns:
        AuctionResult with winner and payment
    """
    import numpy as np

    np.random.seed(42)

    for agent_id in range(num_bidders):
        true_val = valuations.get(agent_id, 0.0)

        if isinstance(auction, (VickreyAuction, FirstPriceAuction)):
            bid_amount = true_val + np.random.normal(0, noise_std)
            bid = Bid(agent_id=agent_id, amount=max(0, bid_amount))
            auction.submit_bid(bid)

    return auction.close()
