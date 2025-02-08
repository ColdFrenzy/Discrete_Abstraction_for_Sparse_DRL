from abc import ABC, abstractmethod

import numpy as np


class DiscountScheduler(ABC):
    def __init__(self, initial_value, final_value, total_steps):
        self.initial_value = initial_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.current_step = 0

    def __call__(self):
        if self.current_step >= self.total_steps:
            return self.final_value
        self.current_step += 1
        return self.calculate_value()

    @abstractmethod
    def calculate_value(self):
        pass


class LinearDiscountScheduler(DiscountScheduler):
    def calculate_value(self):
        # Linear interpolation between initial and final value
        return self.initial_value + (self.final_value - self.initial_value) * (
            self.current_step / self.total_steps
        )


class ExponentialDiscountScheduler(DiscountScheduler):
    def calculate_value(self):
        # Exponential interpolation between initial and final value
        return self.initial_value * (
            self.final_value / self.initial_value
        ) ** (self.current_step / self.total_steps)


class SigmoidDiscountScheduler(DiscountScheduler):
    def calculate_value(self):
        # Sigmoid interpolation between initial and final value
        progress = self.current_step / self.total_steps
        sigmoid_progress = 1 / (1 + np.exp(-10 * (progress - 0.5)))
        return (
            self.initial_value
            + (self.final_value - self.initial_value) * sigmoid_progress
        )
