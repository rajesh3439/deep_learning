import numpy as np
from matplotlib import pyplot as plt

from bandit_env import KArmedBandits

def run_episode(bandits:KArmedBandits, agent):
    """
    Runs the eps-greedy algorithm on the k-bandits environment 
    for the full time period specified in evnironment
    """
    rewards = np.zeros(bandits.sim_steps)
    actions = np.zeros(bandits.sim_steps, dtype=int)

    # reset env and agent befroe episode begin
    bandits.reset()
    agent.reset()

    # Loop the agent and env 
    for i in range(bandits.sim_steps):
        a = agent.choose() # choose the action
        r = bandits.pull(a) # sim action and get reward
        agent.update(a,r) # update q_est

        rewards[i] = r
        actions[i] = a
    
    # is optimal action taken in a given time_step
    optimal_actions = actions == bandits.optimal_action

    return agent.q_est, rewards, optimal_actions

def run_experiment(bandits:KArmedBandits, agent, runs:int) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs full episode of agent-env interaction and outputs rewards and optimal agent actions
    """

    run_rewards = np.zeros((runs, bandits.sim_steps))
    run_optimal_actions = np.zeros((runs, bandits.sim_steps))

    for r in range(runs):
        _, rewards, optimal_actions = run_episode(bandits, agent)
        run_rewards[r] = rewards
        run_optimal_actions[r] = optimal_actions

    return run_rewards, run_optimal_actions

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
