"""
ew_realdata_experiment.py
-------------------------
Part 2C – Exponential Weights on Real Data

Reads 'real_payoffs.csv' (n × k matrix of scaled payoffs in [0,1])
and:
  • Runs Exponential Weights (EW) with ε≈0, ε*=√(ln k/n), and ε→∞
  • Sweeps ε over a range to find empirical optimal ε
  • Plots regret curves and Final Regret vs ε
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math, random, os, json

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "real_payoffs.csv"
OUTDIR = "./ew_real_outputs"
MC = 30             # Monte Carlo trials (resampling with replacement)
SWEEP_ENABLED = True
EPS_MIN, EPS_MAX, EPS_POINTS = 1e-4, 2.0, 25
SEED_BASE = 42

# ============================================================
# EXPONENTIAL WEIGHTS CORE
# ============================================================

class EWPolicy:
    def __init__(self, k, epsilon):
        self.k = k
        self.epsilon = epsilon
        self.weights = np.ones(k)
    def probs(self):
        wsum = self.weights.sum()
        return self.weights / wsum if wsum > 0 else np.ones(self.k)/self.k
    def select_action(self, rng):
        p = self.probs()
        return int(np.searchsorted(np.cumsum(p), rng.random()))
    def update(self, a, r):
        if self.epsilon == 0: return
        self.weights[a] *= math.exp(self.epsilon * r)

class FTLPolicy:
    def __init__(self, k): self.cum = np.zeros(k)
    def select_action(self, rng):
        m = np.max(self.cum); idx = np.flatnonzero(np.isclose(self.cum,m))
        return int(rng.choice(idx))
    def update(self,a,r): self.cum[a]+=r

class UniformPolicy:
    def __init__(self,k): self.k=k
    def select_action(self,rng): return rng.randrange(self.k)
    def update(self,a,r): pass

# ============================================================
# SIMULATION
# ============================================================

def run_once(payoffs, policy_type, epsilon, seed):
    """Simulate one pass through payoff matrix."""
    n, k = payoffs.shape
    rng = random.Random(seed)
    if policy_type=="ew": pol=EWPolicy(k,epsilon)
    elif policy_type=="ftl": pol=FTLPolicy(k)
    elif policy_type=="uniform": pol=UniformPolicy(k)
    else: raise ValueError

    learner_rewards=np.zeros(n); all_cum=np.zeros(k)
    for t in range(n):
        a=pol.select_action(rng)
        rewards=payoffs[t]; r=rewards[a]
        learner_rewards[t]=r
        all_cum+=rewards
        pol.update(a,r)

    best=np.max(all_cum)
    best_avg=best/n
    reg_curve=np.zeros(n)
    cum_l=0
    for t in range(n):
        cum_l+=learner_rewards[t]
        reg_curve[t]=(t+1)*best_avg - cum_l
    return reg_curve

def run_mc(payoffs, policy_type, epsilon, mc, seed0):
    n,k=payoffs.shape
    regrets=[]
    for m in range(mc):
        sample=payoffs[np.random.default_rng(seed0+m).integers(0,n,size=n)]
        reg=run_once(sample,policy_type,epsilon,seed0+m*101)
        regrets.append(reg)
    return np.mean(regrets,axis=0)

# ============================================================
# MAIN EXPERIMENT
# ============================================================

def theoretical_opt_lr(k,n): return math.sqrt(math.log(k)/n)

def main():
    os.makedirs(OUTDIR,exist_ok=True)
    payoffs=pd.read_csv(CSV_PATH).values
    n,k=payoffs.shape
    eps_star=theoretical_opt_lr(k,n)

    print(f"Loaded {CSV_PATH} with shape (n={n}, k={k})")
    print(f"Theoretical ε* = {eps_star:.4f}")

    res_uniform=run_mc(payoffs,"uniform",0,MC,SEED_BASE)
    res_theory=run_mc(payoffs,"ew",eps_star,MC,SEED_BASE)
    res_ftl=run_mc(payoffs,"ftl",0,MC,SEED_BASE)

    np.savez(os.path.join(OUTDIR,"curves_realdata.npz"),
             uniform=res_uniform,theory=res_theory,ftl=res_ftl)

    # Plot curves
    plt.figure()
    plt.plot(res_uniform,label="Uniform (ε≈0)")
    plt.plot(res_theory,label=f"EW (ε*={eps_star:.3f})")
    plt.plot(res_ftl,label="FTL (ε→∞)")
    plt.title("Regret over Time — Real Data")
    plt.xlabel("Round t"); plt.ylabel("Average Regret"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR,"regret_curves_realdata.png"))
    plt.close()

    # Sweep ε
    if SWEEP_ENABLED:
        eps_list=np.linspace(EPS_MIN,EPS_MAX,EPS_POINTS)
        finals=[]
        for eps in eps_list:
            r=run_mc(payoffs,"ew",eps,MC,SEED_BASE)
            finals.append(r[-1])
        out={"epsilons":eps_list.tolist(),"final_regrets":finals}
        with open(os.path.join(OUTDIR,"sweep_realdata.json"),"w") as f: json.dump(out,f,indent=2)
        plt.figure(); plt.plot(eps_list,finals,marker="o")
        plt.title("Final Regret vs ε — Real Data")
        plt.xlabel("ε"); plt.ylabel("Final Regret"); plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR,"final_regret_vs_epsilon_realdata.png"))
        plt.close()

        eps_opt=eps_list[int(np.argmin(finals))]
        print(f"Empirical ε_opt ≈ {eps_opt:.4f} (theoretical {eps_star:.4f})")

    print("✅ Results saved to",OUTDIR)

if __name__=="__main__":
    main()
