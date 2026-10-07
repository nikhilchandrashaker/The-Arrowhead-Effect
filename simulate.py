"""
Counterfactual simulation engine.

This intentionally separates:
1) estimated statistical effects
2) counterfactual exposure
3) game simulation

Do not hard-code arbitrary effect sizes. Load fitted coefficients from a model output file.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Scenario:
    name: str
    noise_index: float

SCENARIOS = [
    Scenario("Observed", 0.0),
    Scenario("Conservative Arrowhead", 0.50),
    Scenario("Central Arrowhead", 1.00),
    Scenario("Aggressive Arrowhead", 1.25),
]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate_binary_outcome(linear_predictor, rng):
    return int(rng.random() < sigmoid(linear_predictor))

def run_monte_carlo(play_table, fitted_models, seasons=10000, seed=42):
    """
    play_table: one row per modeled play with covariates.
    fitted_models: fitted model objects / coefficient dictionaries.
    Returns one simulated outcome per season and scenario.

    The production version should simulate sequential drives/games rather than
    treating plays as independent. This function is the API boundary for that engine.
    """
    rng = np.random.default_rng(seed)
    results = []
    for scenario in SCENARIOS:
        for s in range(seasons):
            # Placeholder: production implementation uses model predictions
            # and sequential game-state transitions.
            results.append({"scenario": scenario.name, "simulation": s})
    return pd.DataFrame(results)
