
from bandit_env import KArmedBandits, _NBANDITS
from agents import EpsGreedy
from runner import run_experiment, plot_fig22


_ROOT_SEED = 39 # for repeatability
nsteps = 1000
nruns = 2000


def fig22():
    # Fig 2.2
    results = {}
    epsilons = [0, 0.1, 0.01]
    print(f"Running eps-greedy agent on 10-armed bandits with {epsilons=}")

    for eps in epsilons:
        bandits = KArmedBandits(_NBANDITS, steps=nsteps, seed=_ROOT_SEED+1)
        agent = EpsGreedy(_NBANDITS, eps, alpha= None, seed = _ROOT_SEED+2)
        rewards, optimal = run_experiment(bandits, agent, runs=nruns)
        results[f"{eps=}"] = (rewards.mean(axis= 0), optimal.mean(axis=0))

    plot_fig22(results, f"Epsilon-greedy method on 10 Armed Bandits with {epsilons=}", "fig22.png")
    print("Eps-greedy agent run complete!")

def ex25():
    # Ex 2.5
    # Exercise 2.5 (programming) Design and conduct an experiment to demonstrate the diffculties 
    # that sample-average methods have for nonstationary problems. Use a modified version of 
    # the 10-armed testbed in which all the q⇤(a) start out equal and then take independent 
    # random walks (say by adding a normally distributed increment with mean zero and 
    # standard deviation 0.01 to all the q⇤(a) on each step). Prepare plots like Figure 2.2 for 
    # an action-value method using sample averages, incrementally computed, and another action-value 
    # method using a constant step-size parameter, ↵ = 0.1. Use " = 0.1 and longer runs, say of 10,000 steps.
    nsteps, nruns, eps = 10_000, 2000, 0.1
    results = {}

    for label, alpha in [('sample_average', None),("alpha=0.1",0.1)]: # iterate over two items that are varying    
            bandits = KArmedBandits(_NBANDITS, steps=nsteps, stationary=False, seed=_ROOT_SEED+1)
            agent = EpsGreedy(_NBANDITS, eps, alpha=alpha, seed=_ROOT_SEED+2)

            rewards, optimal = run_experiment(bandits, agent, runs=nruns)
            results[label] = (rewards.mean(axis= 0), optimal.mean(axis=0))

    plot_fig22(results, f"Non-stationary 10-armed bandits, eps={eps}, {nruns} runs", "ex25_nonstationary.png")



EXPERIMENTS = {"fig22": fig22, "ex25": ex25}


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:] or EXPERIMENTS:
        EXPERIMENTS[name]()