import numpy as np

_NBANDITS=10

"""
Environment Class
"""
class KArmedBandits:
    """
    The model class to generate the armed bandits
    """
    def __init__(self, n_arms:int=10, steps:int=2000, stationary:bool=True, seed:int=None,) -> None:
        self.K = n_arms
        self.rng = np.random.default_rng(seed)
        self.is_stationary = stationary
        self.sim_steps = steps 
        # self.reset()
    
    def reset(self):
        # Initialize the q*(a) for each bandit
        # Stationary: assign reward picked up from standard normal
        # Non-Stationary: 0 rewards at start and take random-walk each step        
        self.q_bandits = self.rng.standard_normal(self.K) if self.is_stationary else np.zeros(self.K) 
        # record optimal action at time-step
        # Stationary case its the same for all timestep
        self.optimal_action = np.full(self.sim_steps, np.argmax(self.q_bandits), dtype=int)
        # self.optimal_action = np.argmax(self.q_bandits)
        self.t = 0 # current step count

    def noise(self):
        """ Gernerates random noise for each K-bandit machine"""
        return self.rng.normal(loc=0, scale=0.01, size=self.K)

    def pull(self, a):
        """Simulates arm-pull action and outputs reward for the given machine ids"""
        if not self.is_stationary: 
            # take a random walk before the reward At the very first time step, 
            # you start with a random walk and a selected nation instead of zero.
            # This helps with the loop and keeping track of optimal actions per time_step
            self.q_bandits += self.noise() # add IID noise to each K bandits
            # record the optimal action at each timestep
            self.optimal_action[self.t] = np.argmax(self.q_bandits)

        self.t +=1 # advance the time step

        return self.rng.normal(loc=self.q_bandits[a], scale=1)
