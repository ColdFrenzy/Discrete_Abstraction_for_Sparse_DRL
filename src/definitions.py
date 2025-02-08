from enum import Enum


class RewardType(Enum):
    dense = 0
    sparse = 1
    model = 2


class TransitionMode(Enum):
    deterministic = 0
    stochastic = 1
