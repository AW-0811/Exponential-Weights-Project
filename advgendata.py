"""
ew_adversarial_generator.py
---------------------------
Part 2D – Adversarial Generative Model (Shifting-Best-Arm)

Simulates a randomized environment where:
  - A single "best" arm switches with probability q each round.
  - The best arm yields payoff p_hi, others p_lo = p_hi − Δ.
  - Neither FTL nor Uniform achieves vanishing regret.

Outputs:
  • Regret curves (Uniform, EW(ε*), FTL)
  • Final Regret vs ε sweep
"""

import numpy as np
import matplotlib.pyplot as plt
import random, math, os, json

# ============================================================
# CONFIGURATION
# ============================================================

K = 8                 # number of arms
N = 1000              # rounds per trial
MC = 30               # Monte Carlo runs
OUTDIR = "./ew_adversarial_outputs"

# Environment params
SWITCH_PROB = 0.02    # probability that best arm switches each round
P_HI = 0.5
DELTA = 0.2           # p_lo = 0.5 − DELTA

# EW parameters
SWEEP_ENABLED = True
EPS_MIN, EPS_MAX, EPS_POINTS = 1e-4, 2.0, 25
SEED_BASE = 42

# ============================================================
# POLICIES
# ============================================================

class EWPolicy:
    def __init__(self,k,eps): self.k=k; self.eps=eps; self.w=np.ones(k)
    def probs(self):
        s=self.w.sum(); return self.w/s if s>0 else np.ones(self.k)/self.k
    def select(self,rng):
        p=self.probs(); return int(np.searchsorted(np.cumsum(p),rng.random()))
    def update(self,a,r):
        if self.eps==0: return
        self.w[a]*=math.exp(self.eps*r)

class FTLPolicy:
    def __init__(self,k): self.k=k; self.c=np.zeros(k)
    def select(self,rng):
        m=np.max(self.c); idx=np.flatnonzero(np.isclose(self.c,m))
        return int(rng.choice(idx))
    def update(self,a,r): self.c[a]+=r

class UniformPolicy:
    def __init__(self,k): self.k=k
    def select(self,rng): return rng.randrange(self.k)
    def update(self,a,r): pass

# ============================================================
# ENVIRONMENT
# ============================================================

class ShiftingBestArm:
    def __init__(self,k,n,q,p_hi,delta,seed):
        self.k,self.n,self.q,self.p_hi,self.p_lo=k,n,q,p_hi,max(0,p_hi-delta)
        self.rng=random.Random(seed); self.j_star=self.rng.randrange(k)
    def step(self):
        # Switch with probability q
        if self.rng.random()<self.q:
            new=self.rng.randrange(self.k-1)
            if new>=self.j_star: new+=1
            self.j_star=new
        pay=np.zeros(self.k)
        for j in range(self.k):
            p=self.p_hi if j==self.j_star else self.p_lo
            pay[j]=1.0 if self.rng.random()<p else 0.0
        return pay

# ============================================================
# CORE SIMULATION
# ============================================================

def run_once(k,n,policy_type,eps,seed):
    env=ShiftingBestArm(k,n,SWITCH_PROB,P_HI,DELTA,seed+1337)
    rng=random.Random(seed)
    if policy_type=="ew": pol=EWPolicy(k,eps)
    elif policy_type=="ftl": pol=FTLPolicy(k)
    elif policy_type=="uniform": pol=UniformPolicy(k)
    else: raise ValueError

    learner_rewards=np.zeros(n); all_cum=np.zeros(k)
    for t in range(n):
        a=pol.select(rng)
        rewards=env.step()
        r=rewards[a]; learner_rewards[t]=r; all_cum+=rewards
        pol.update(a,r)
    best=np.max(all_cum); best_avg=best/n
    reg_curve=np.zeros(n); cum_l=0
    for t in range(n):
        cum_l+=learner_rewards[t]; reg_curve[t]=(t+1)*best_avg-cum_l
    return reg_curve

def run_mc(k,n,policy_type,eps,mc,seed0):
    regrets=[]
    for m in range(mc):
        r=run_once(k,n,policy_type,eps,seed0+m*101)
        regrets.append(r)
    return np.mean(regrets,axis=0)

def theoretical_opt_lr(k,n): return math.sqrt(math.log(k)/n)

# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    os.makedirs(OUTDIR,exist_ok=True)
    eps_star=theoretical_opt_lr(K,N)
    print(f"Theoretical ε* = {eps_star:.4f}")

    res_uniform=run_mc(K,N,"uniform",0,MC,SEED_BASE)
    res_theory=run_mc(K,N,"ew",eps_star,MC,SEED_BASE)
    res_ftl=run_mc(K,N,"ftl",0,MC,SEED_BASE)

    np.savez(os.path.join(OUTDIR,"curves_adversarial.npz"),
             uniform=res_uniform,theory=res_theory,ftl=res_ftl)

    # Plot baseline curves
    plt.figure()
    plt.plot(res_uniform,label="Uniform (ε≈0)")
    plt.plot(res_theory,label=f"EW (ε*={eps_star:.3f})")
    plt.plot(res_ftl,label="FTL (ε→∞)")
    plt.title("Regret over Time — Shifting Best Arm")
    plt.xlabel("Round t"); plt.ylabel("Average Regret")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR,"regret_curves_adversarial.png"))
    plt.close()

    # Sweep ε
    if SWEEP_ENABLED:
        eps_list=np.linspace(EPS_MIN,EPS_MAX,EPS_POINTS)
        finals=[]
        for eps in eps_list:
            r=run_mc(K,N,"ew",eps,MC,SEED_BASE)
            finals.append(r[-1])
        out={"epsilons":eps_list.tolist(),"final_regrets":finals}
        with open(os.path.join(OUTDIR,"sweep_adversarial.json"),"w") as f: json.dump(out,f,indent=2)
        plt.figure(); plt.plot(eps_list,finals,marker="o")
        plt.title("Final Regret vs ε — Adversarial Generator")
        plt.xlabel("ε"); plt.ylabel("Final Regret")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR,"final_regret_vs_epsilon_adversarial.png"))
        plt.close()
        eps_opt=eps_list[int(np.argmin(finals))]
        print(f"Empirical ε_opt ≈ {eps_opt:.4f} (theoretical {eps_star:.4f})")

    print("✅ Results saved to",OUTDIR)

if __name__=="__main__":
    main()
