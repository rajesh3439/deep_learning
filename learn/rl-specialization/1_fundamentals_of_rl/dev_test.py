# from k_armed_bandits import *
# from k_armed_bandits import _ROOT_SEED, _NBANDITS
from bandit_env import KArmedBandits, _NBANDITS
from agents import EpsGreedy
from runner import run_episode, run_experiment, print_q_table
from experiments import _ROOT_SEED

bandits = KArmedBandits(_NBANDITS, steps=100, seed=_ROOT_SEED+1)
agent = EpsGreedy(_NBANDITS, 0.01, alpha= None, seed = _ROOT_SEED+2)
Q,_,_ = run_episode(bandits, agent)
print_q_table(_NBANDITS, bandits.q_bandits, Q)


results = {}
epsilons = [0, 0.1, 0.01]

for eps in epsilons:
    bandits = KArmedBandits(_NBANDITS, steps=1000, seed=_ROOT_SEED+1)
    agent = EpsGreedy(_NBANDITS, eps, alpha= None, seed = _ROOT_SEED+2)
    # Q,_,_ = run_episode(bandits, agent)
    rewards, optimal = run_experiment(bandits, agent, runs=5)
    results[f"{eps=}"] = (rewards.mean(axis= 0), optimal.mean(axis=0))
    # print_q_table(_NBANDITS, bandits.q_bandits, Q)
    print(f"{eps=:>6} \t {rewards.shape=},{optimal.shape=},{optimal.dtype=}")
