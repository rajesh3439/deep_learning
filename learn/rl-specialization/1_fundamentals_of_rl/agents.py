import numpy as np

"""
Agent Class
"""
class EpsGreedy:
    def __init__(self, n_arms:int, eps:float=0.01, alpha:float=None, seed:int=None) -> None:
        self.K = n_arms
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        self.alpha = alpha
        # self.reset()
        
    def reset(self): # Initialize the agent
        self.csteps = np.zeros(self.K) # Initialize pull counter of each bandit to zero
        self.q_est = np.zeros(self.K) # Initialize Q Value estimates of each bandit to zero 

        # Add jitter to break ties that happens to zero intialization intially. 
        # Draw from standard normal and scale down by factor of 1e-8
        jitter = self.rng.standard_normal(self.K) * 1e-8
        self.q_est += jitter 

    def choose(self): # explore or exploit
        if self.rng.uniform(0, 1) <= self.eps: 
            action = self.rng.integers(self.K) # explore
        else: 
            action = np.argmax(self.q_est) # exploit
        return action

    def update(self, a, r): #N, step size, Q
        self.csteps[a] += 1 # Increment the select count
        step_size = 1/self.csteps[a] if self.alpha is None else self.alpha
        self.q_est[a] += step_size * (r - self.q_est[a])
