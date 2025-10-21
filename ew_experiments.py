"""
===============================================================
Exponential Weights (EW) Online Learning — Simplified Version
===============================================================

Modify configuration values in the CONFIG SECTION below and run:
    python ew_experiments_simplified.py

Generates regret curves and epsilon-sweep plots for the chosen
payoff model (A: Adversarial Fair, B: Bernoulli).
---------------------------------------------------------------
Author: ChatGPT (GPT-5)
"""

import os
import math
import json
import random
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
# CONFIGURATION SECTION — modify these values freely
# ==============================================================

MODEL = "B"          # "A" for Adversarial Fair, "B" for Bernoulli
K = 8                # number of actions
N = 800              # number of rounds
MC = 30              # Monte Carlo repetitions
OUTDIR = "./ew_outputs"   # output folder

# Learning rate sweep range (for EW only)
SWEEP_ENABLED = True
EPS_MIN = 1e-4
EPS_MAX = 2.0
EPS_POINTS = 15

# Bernoulli model probabilities (only used if MODEL="B")
PS = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]  # must have length K

# Random seed
SEED_BASE = 42

# ==============================================================


# ---------- Utility Functions ----------

def theoretical_opt_lr(k: int, n: int) -> float:
    return math.sqrt(math.log(k) / n)


# ---------- Policies ----------

class EWPolicy:
    def __init__(self, k: int, epsilon: float):
        self.k = k
        self.epsilon = epsilon
        self.weights = np.ones(k)

    def probs(self):
        wsum = self.weights.sum()
        if wsum <= 0 or not np.isfinite(wsum):
            return np.ones(self.k) / self.k
        return self.weights / wsum

    def select_action(self, rng: random.Random):
        p = self.probs()
        return int(np.searchsorted(np.cumsum(p), rng.random()))

    def update(self, action: int, reward: float):
        if self.epsilon == 0:
            return
        self.weights[action] *= math.exp(self.epsilon * reward)


class FTLPolicy:
    def __init__(self, k: int):
        self.k = k
        self.cum_rewards = np.zeros(k)

    def select_action(self, rng: random.Random):
        max_val = np.max(self.cum_rewards)
        candidates = np.flatnonzero(np.isclose(self.cum_rewards, max_val))
        return int(rng.choice(candidates))

    def update(self, action: int, reward: float):
        self.cum_rewards[action] += reward


class UniformPolicy:
    def __init__(self, k: int):
        self.k = k
    def select_action(self, rng: random.Random):
        return rng.randrange(self.k)
    def update(self, action: int, reward: float):
        pass


# ---------- Environments ----------

class EnvAAdversarialFair:
    def __init__(self, k: int, seed: int):
        self.k = k
        self.rng = random.Random(seed)
        self.cum_payoffs = np.zeros(k)

    def step(self):
        x = self.rng.random()
        j_star = int(np.argmin(self.cum_payoffs))
        payoffs = np.zeros(self.k)
        payoffs[j_star] = x
        self.cum_payoffs[j_star] += x
        return payoffs


class EnvBBernoulli:
    def __init__(self, ps, seed: int):
        self.ps = ps
        self.k = len(ps)
        self.rng = random.Random(seed)

    def step(self):
        return np.array([1.0 if self.rng.random() < p else 0.0 for p in self.ps])


# ---------- Simulation Core ----------

def simulate_once(model, k, n, policy_type, epsilon, seed, ps=None):
    rng = random.Random(seed)
    if model == "A":
        env = EnvAAdversarialFair(k, seed + 1337)
    elif model == "B":
        env = EnvBBernoulli(ps, seed + 1337)
    else:
        raise ValueError("Model must be 'A' or 'B'")

    # select policy
    if policy_type == "ew":
        policy = EWPolicy(k, epsilon)
    elif policy_type == "ftl":
        policy = FTLPolicy(k)
    elif policy_type == "uniform":
        policy = UniformPolicy(k)
    else:
        raise ValueError("invalid policy type")

    chosen_rewards = np.zeros(n)
    actions = np.zeros(n, dtype=int)
    all_action_cum = np.zeros(k)
    learner_cum = 0.0

    for t in range(n):
        a_t = policy.select_action(rng)
        payoffs = env.step()
        r_t = payoffs[a_t]

        actions[t] = a_t
        chosen_rewards[t] = r_t
        learner_cum += r_t
        all_action_cum += payoffs

        policy.update(a_t, r_t)

    best_fixed_total = np.max(all_action_cum)
    best_fixed_avg = best_fixed_total / n
    learner_cum_t = 0.0
    regret_curve = np.zeros(n)
    for t in range(n):
        learner_cum_t += chosen_rewards[t]
        regret_curve[t] = (t + 1) * best_fixed_avg - learner_cum_t
    return regret_curve


def run_mc(model, k, n, mc, policy_type, epsilon, ps=None):
    regrets = np.zeros((mc, n))
    for m in range(mc):
        regrets[m] = simulate_once(model, k, n, policy_type, epsilon,
                                   seed=SEED_BASE + m * 9973, ps=ps)
    return regrets.mean(axis=0)


# ---------- Plot Helpers ----------

def plot_regret_curves(curves, title, path):
    plt.figure()
    for label, curve in curves.items():
        plt.plot(curve, label=label)
    plt.title(title)
    plt.xlabel("Round t")
    plt.ylabel("Average Regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_final_vs_epsilon(epsilons, final_regrets, title, path):
    plt.figure()
    plt.plot(epsilons, final_regrets, marker="o")
    plt.title(title)
    plt.xlabel("ε (learning rate)")
    plt.ylabel("Final Average Regret")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ---------- Main Execution ----------

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    ps = PS if MODEL == "B" else None

    eps_theory = theoretical_opt_lr(K, N)

    # Run policies
    print(f"Running Model {MODEL}, k={K}, n={N}, mc={MC}")
    res_uniform = run_mc(MODEL, K, N, MC, "uniform", 0.0, ps)
    res_theory = run_mc(MODEL, K, N, MC, "ew", eps_theory, ps)
    res_ftl = run_mc(MODEL, K, N, MC, "ftl", 0.0, ps)

    # Save regret curves
    np.savez(os.path.join(OUTDIR, f"curves_model{MODEL}.npz"),
             uniform=res_uniform, theory=res_theory, ftl=res_ftl)

    # Plot average regret curves
    plot_regret_curves(
        {"Uniform (ε≈0)": res_uniform,
         f"EW (ε*={eps_theory:.3f})": res_theory,
         "FTL (ε→∞)": res_ftl},
        title=f"Regret Curves — Model {MODEL}",
        path=os.path.join(OUTDIR, f"regret_curves_model{MODEL}.png")
    )

    # Sweep ε values
    if SWEEP_ENABLED:
        epsilons = np.linspace(EPS_MIN, EPS_MAX, EPS_POINTS)
        finals = []
        for eps in epsilons:
            curve = run_mc(MODEL, K, N, MC, "ew", eps, ps)
            finals.append(curve[-1])
        sweep_data = {"epsilons": epsilons.tolist(),
                      "final_regrets": finals}
        with open(os.path.join(OUTDIR, f"sweep_model{MODEL}.json"), "w") as f:
            json.dump(sweep_data, f, indent=2)
        plot_final_vs_epsilon(
            epsilons, finals,
            title=f"Final Regret vs ε — Model {MODEL}",
            path=os.path.join(OUTDIR, f"final_regret_vs_epsilon_model{MODEL}.png")
        )

    # Print summary
    print("=== Summary ===")
    print(f"Uniform baseline final regret: {res_uniform[-1]:.3f}")
    print(f"Theoretical ε*={eps_theory:.4f} final regret: {res_theory[-1]:.3f}")
    print(f"FTL (ε→∞) final regret: {res_ftl[-1]:.3f}")
    print(f"Plots + data saved to: {OUTDIR}")
