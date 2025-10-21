"""
ew_doubling_trick.py
--------------------
Exponential Weights with the Doubling Trick
Compares against fixed ε* and FTL in the shifting-best-arm environment.
"""

import numpy as np, math, random, matplotlib.pyplot as plt, os

# ---------------- CONFIG ----------------
K, N, MC = 8, 1000, 30
SWITCH_PROB, P_HI, DELTA = 0.02, 0.5, 0.2
OUTDIR = "./ew_doubling_outputs"
SEED = 42
# ----------------------------------------

class ShiftingBestArm:
    def __init__(self,k,n,q,p_hi,delta,seed):
        self.k,self.n,self.q,self.p_hi,self.p_lo=k,n,q,p_hi,max(0,p_hi-delta)
        self.rng=random.Random(seed); self.star=self.rng.randrange(k)
    def step(self):
        if self.rng.random()<self.q:
            new=self.rng.randrange(self.k-1)
            if new>=self.star: new+=1
            self.star=new
        pay=np.zeros(self.k)
        for j in range(self.k):
            pay[j]=1.0 if self.rng.random()<(self.p_hi if j==self.star else self.p_lo) else 0.0
        return pay

def theoretical_eta(k,n): return math.sqrt(math.log(k)/n)

# ----------- fixed-ε baseline -----------
def run_fixed(k,n,eps,seed):
    env=ShiftingBestArm(k,n,SWITCH_PROB,P_HI,DELTA,seed+100)
    rng=random.Random(seed)
    w=np.ones(k); learner, allcum=np.zeros(n),np.zeros(k)
    for t in range(n):
        p=w/w.sum(); a=int(np.searchsorted(np.cumsum(p),rng.random()))
        rws=env.step(); r=rws[a]; learner[t]=r; allcum+=rws
        if eps>0: w[a]*=math.exp(eps*r)
    best=allcum.max(); bestavg=best/n
    return np.cumsum(bestavg-learner)

# ------------- FTL baseline -------------
def run_ftl(k,n,seed):
    env=ShiftingBestArm(k,n,SWITCH_PROB,P_HI,DELTA,seed+200)
    rng=random.Random(seed); c=np.zeros(k); learner=np.zeros(n); allcum=np.zeros(k)
    for t in range(n):
        mx=np.max(c); idx=np.flatnonzero(np.isclose(c,mx))
        a=int(rng.choice(idx)); rws=env.step(); r=rws[a]; learner[t]=r; allcum+=rws; c[a]+=r
    best=allcum.max(); bestavg=best/n
    return np.cumsum(bestavg-learner)

# -------- Doubling-trick variant --------
def run_doubling(k,n,seed):
    env=ShiftingBestArm(k,n,SWITCH_PROB,P_HI,DELTA,seed+300)
    rng=random.Random(seed); allcum=np.zeros(k); learner=np.zeros(n)
    t=0; epoch=0
    while t<n:
        T_epoch=2**epoch
        eta=math.sqrt(math.log(k)/T_epoch)
        w=np.ones(k)
        for _ in range(T_epoch):
            if t>=n: break
            p=w/w.sum(); a=int(np.searchsorted(np.cumsum(p),rng.random()))
            rws=env.step(); r=rws[a]; learner[t]=r; allcum+=rws
            w[a]*=math.exp(eta*r); t+=1
        epoch+=1
    best=allcum.max(); bestavg=best/n
    return np.cumsum(bestavg-learner)

# --------------- Monte Carlo ---------------
def mc(runfun,mc,seed0,*args):
    regs=[]
    for i in range(mc): regs.append(runfun(*args,seed0+i*111))
    return np.mean(regs,axis=0)

def main():
    os.makedirs(OUTDIR,exist_ok=True)
    eps_star=theoretical_eta(K,N)
    print(f"Theoretical ε* = {eps_star:.4f}")

    r_fixed=mc(lambda k,n,s:run_fixed(k,n,eps_star,s),MC,SEED,K,N)
    r_ftl=mc(run_ftl,MC,SEED,K,N)
    r_double=mc(run_doubling,MC,SEED,K,N)

    # After plotting regret curves
    plt.plot(r_fixed, label=f"EW (ε*={eps_star:.3f})")
    plt.plot(r_double, label="EW Doubling Trick")
    plt.plot(r_ftl, label="FTL")

    # --- Add vertical lines for Doubling Trick epochs ---
    epochs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    for e in epochs:
        if e < N:
            plt.axvline(x=e, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.title("Regret over Time — Doubling Trick Comparison")
    plt.xlabel("Round t"); plt.ylabel("Average Regret")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "regret_doubling_vs_baselines.png"))

    print("✅ Saved plot to",OUTDIR)

if __name__=="__main__": main()
