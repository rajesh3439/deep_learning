import numpy as np
from matplotlib import pyplot as plt

_NBANDITS = 10 # number of bandit machines
_ROOT_SEED = 39 # for repeatability

class KArmedBandits:
    """
    The model class to generate the armed bandits
    """
    def __init__(self, n_arms:int=10, stationary:bool=True, seed:int=None, steps:int=None) -> None:
        self.K = n_arms
        self.rng = np.random.default_rng(seed)
        self.is_stationary = stationary
        self.sim_steps = steps 
        self.reset()
    
    def reset(self):
        self.q_bandits = self.rng.standard_normal(self.K) 
        self.optimal_action = np.argmax(self.q_bandits)

        if not self.is_stationary: 
            self.q_bandits = np.zeros(self.K) # start at 0 rewards and take random-walk each step
            self.q_history = np.zeros((self.sim_steps, self.K)) # record_store for q*
            self.optimal_action = np.zeros(self.sim_steps) # record optimal action at time-step

    def noise(self):
        """ Gernerates random noise for each K-bandit machine"""
        return self.rng.normal(loc=0, scale=0.01, size=self.K)

    def play_select(self, ids: int | list[int], step:int):
        """Simulates arm-pull action and outputs reward for the given machine ids"""
        if not self.is_stationary: 
            # take a random walk before the reward At the very first time step, 
            # you start with a random walk and a selected nation instead of zero.
            # This helps with the loop and keeping track of optimal actions per time_step
            self.q_bandits += self.noise() # add IID noise to each K bandits
            # record the q* and optimal action
            self.q_history[step] = self.q_bandits
            self.optimal_action[step] = np.argmax(self.q_bandits)

        res = self.rng.normal(loc=self.q_bandits[ids], scale=1)

        return res 


def run_simple_bandit(bandits:KArmedBandits, steps:int, eps:float=0.01, alpha:float=None, seed:int=None): 
    """
    Epsilon Greedy algorithm with the incremental update for the reward
    
    Ref: Simple Bandit Algorithm from Section 2.4 of 2nd edition of RL by Sutton & Barto

    It outputs the estimate of rewards for k-bandits after the given step count
    """
    Q = np.zeros_like(bandits.q_bandits) # Initialize Value estimates of each bandit to zero 
    N = np.zeros_like(Q) # Initialize step count to zero
    rng = np.random.default_rng(seed)

    rewards = np.zeros(steps)
    actions = np.zeros(steps, dtype=int)

    # Add jitter to break ties initially. Draw from standard normal and 
    # scale down by factor of 1e-8
    jitter = rng.standard_normal(bandits.K) * 1e-8
    Q = Q + jitter 

    for i in range(steps):
        if rng.uniform(0, 1) <= eps: 
            A = rng.integers(bandits.K) # explore
        else: 
            A = np.argmax(Q) # exploit

        R = bandits.play_select(A, i)
        N[A] = N[A] + 1 # Increment the select count
        step_size = 1/N[A] if alpha is None else alpha
        Q[A] = Q[A] + (step_size * (R - Q[A]))

        rewards[i] = R
        actions[i] = A
    
    return Q, rewards, actions == bandits.optimal_action


def print_q_table(k, q, Q):
    """Prints actual q_*(a) and estimated Q(a) side by side for k arms"""
    print(f"{'k':>3} {'q_*(a)':>10} {'Q(a)':>10} {'err':>10}")
    for a in range(k):
        print(f"{a:>3} {q[a]:>10.4f} {Q[a]:>10.4f} {Q[a]-q[a]:>10.4f}")
    print(f"best arm: actual={np.argmax(q)}, estimated={np.argmax(Q)}, mae={np.abs(q-Q).sum()/Q.shape[0]}")


def plot_fig22(results: dict, title="", path=None):
    """
    Fig 2.2 style plot: average reward (top) and % optimal action (bottom) vs steps.
    results: {label: (mean_rewards_1d, mean_opt_actions_1d)}, one curve per label.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for lbl, (r, o) in results.items():
        ax1.plot(r, lw=1, label=lbl)
        ax2.plot(o * 100, lw=1, label=lbl)
    ax1.set_ylabel("Average reward")
    ax2.set_ylabel("% Optimal action")
    ax2.set_xlabel("Steps")
    ax2.set_ylim(0, 100)
    ax1.legend(); ax1.grid(alpha=0.3); ax2.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path if path.endswith(".png") else path + ".png", dpi=120)
    plt.show()

# def ten_armbandits_testbed(nruns:int, nsteps:int=1000, epsilons:list=[0,0.1,0.01], stationary=True):
def ten_armbandits_testbed(run_configs:dict):
    """ 
    Run the epsilon greedy method for each `epsilon` value for `nsteps` and `nruns` 
    times and plot average rewards & avg. optimal actions across each runs
    """
    print("Starting simulation runs for each config")
    results = {}

    for exp_key, exp_config in run_configs.items():
        nsteps = exp_config.get("steps", 1000)
        nruns = exp_config.get("runs",2000)
        epsilons = exp_config.get("epsilons", [0,0.1,0.01])
        stationary= exp_config.get("stationary", True)
        method = exp_config.get("method", "sampled_average")

        mean_rewards = np.zeros((len(epsilons), nsteps))
        mean_opt_actions = np.zeros((len(epsilons), nsteps))

        alpha = exp_config.get('alpha', None)

        for e in range(len(epsilons)):
            print(f"Epsilon={epsilons[e]}")
            rewards, optimal_actions = np.zeros((nruns, nsteps)), np.zeros((nruns, nsteps))
            bandits = KArmedBandits(_NBANDITS, stationary=stationary, seed=_ROOT_SEED+1, steps=nsteps)
            for i in range(nruns):
                bandits.reset() # pick fresh reward set each run
                _, rewards[i], optimal_actions[i] = run_simple_bandit(bandits, nsteps, epsilons[e], alpha, seed=_ROOT_SEED+i+2)
            
            mean_rewards[e] = rewards.mean(axis=0)
            mean_opt_actions[e] = optimal_actions.mean(axis=0)
            results[f"{exp_key} {method} eps={epsilons[e]}"] = (mean_rewards[e], mean_opt_actions[e])

    print("Simulation complete!")
    fig_path = "10_armed_bandits_" + ("stationary" if stationary else "non_stationary")
    plot_fig22(results, title=f"{nruns} runs, {nsteps} steps", path=fig_path)


if __name__ == '__main__':
    # Create bandits
    bandits = KArmedBandits(_NBANDITS, seed=_ROOT_SEED+1)
    print(f"Actual rewards for {_NBANDITS} bandits: \n q_*(a)={bandits.q_bandits}")

    # Evaluations
    # Eps-Greedy Algorithm 100 steps
    print("\nEstimation of Q(a) after 100 steps")
    bandits = KArmedBandits(_NBANDITS, seed=_ROOT_SEED+1)
    Q,_,_ = run_simple_bandit(bandits, 100, seed=_ROOT_SEED+2)
    print_q_table(_NBANDITS, bandits.q_bandits, Q)

    # 1000 steps
    print("\nEstimation of Q(a) after 1000 steps")
    bandits = KArmedBandits(_NBANDITS, seed=_ROOT_SEED+1)
    Q,_,_ = run_simple_bandit(bandits, 1000, seed=_ROOT_SEED+2)
    print_q_table(_NBANDITS, bandits.q_bandits, Q)

    # 10000 steps
    bandits = KArmedBandits(_NBANDITS, seed=_ROOT_SEED+1)
    print("\nEstimation of Q(a) after 10000 steps")
    Q,_,_ = run_simple_bandit(bandits, 10000, seed=_ROOT_SEED+2)
    print_q_table(_NBANDITS, bandits.q_bandits, Q)

    # 100000 steps
    print("\nEstimation of Q(a) after 100000 steps")
    bandits = KArmedBandits(_NBANDITS, seed=_ROOT_SEED+1)
    Q,_,_ = run_simple_bandit(bandits, 100000, seed=_ROOT_SEED+2)
    print_q_table(_NBANDITS, bandits.q_bandits, Q)

    # Replicate 2000 runs simulations from Fig 2.2
    _RUNS = 2000
    _STEPS = 1000
    print("Starting 2000 simulation runs on each epsilon levels")
    epsilons = [0,0.1,0.01] # choices of epsilons 
    mean_rewards, mean_opt_actions = np.zeros((len(epsilons), _STEPS)), np.zeros((len(epsilons), _STEPS))

    for e in range(len(epsilons)):
        print(f"Epsilon={epsilons[e]}")
        rewards, optimal_actions = np.zeros((_RUNS, _STEPS)), np.zeros((_RUNS, _STEPS))
        bandits = KArmedBandits(_NBANDITS, seed=_ROOT_SEED+1)
        for i in range(_RUNS):
            bandits.reset() # pick fresh reward set each run
            _, rewards[i], optimal_actions[i] = run_simple_bandit(bandits, _STEPS, epsilons[e], seed=_ROOT_SEED+i+2)
        
        mean_rewards[e] = rewards.mean(axis=0)
        mean_opt_actions[e] = optimal_actions.mean(axis=0)

    print("Simulation complete!")
    plot_fig22({f"eps={e}": (mean_rewards[i], mean_opt_actions[i]) for i, e in enumerate(epsilons)},
               title=f"{_RUNS} runs, {_STEPS} steps", path="fig22.png")
    # Ex 2.5
    # Exercise 2.5 (programming) Design and conduct an experiment to demonstrate the diffculties 
    # that sample-average methods have for nonstationary problems. Use a modified version of 
    # the 10-armed testbed in which all the q⇤(a) start out equal and then take independent 
    # random walks (say by adding a normally distributed increment with mean zero and 
    # standard deviation 0.01 to all the q⇤(a) on each step). Prepare plots like Figure 2.2 for 
    # an action-value method using sample averages, incrementally computed, and another action-value 
    # method using a constant step-size parameter, ↵ = 0.1. Use " = 0.1 and longer runs, say of 10,000 steps.
    run_configs = {
        "exp_1":{
            "method":"sampled_average",     # to_do: change it into enums 
            "stationary":False,
            "epsilons": [0.1],
            "steps":10_000,
            "runs" :2000
        },
        "exp_2":{
            "method":"constant_stepsize",     # to_do: change it into enums 
            "stationary":False,
            "epsilons": [0.1],
            "alpha" : 0.1,
            "steps":10_000,
            "runs" :2000
        }

    }

    ten_armbandits_testbed(run_configs)
    