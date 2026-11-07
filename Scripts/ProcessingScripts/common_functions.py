import numpy as np

def initialize(N, L):
    """initialize the agent positions and directions

    Args:
        N (int): number of agents to spawn
        L (float): arena size

    Returns:
        list: list of all the agent positions
        list: lost of all the agent directions
    """
    x = np.random.rand(N, 2) * L
    a = np.random.uniform(0, 2 * np.pi, N)
    v = np.c_[np.cos(a), np.sin(a)]
    return x, v
