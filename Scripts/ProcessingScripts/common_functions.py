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
    np.random.seed(69)
    x = np.random.rand(N, 2) * L / 4
    a = np.random.uniform(0, 2 * np.pi, N)
    v = np.c_[np.cos(a), np.sin(a)]
    return x, v


# def initialize(N, L):
#     """initialize the agent positions and directions

#     Args:
#         N (int): number of agents to spawn
#         L (float): arena size

#     Returns:
#         list: list of all the agent positions
#         list: lost of all the agent directions
#     """
#     np.random.seed(69)
#     x = np.ones((N, 2)) * L / 2 + np.random.rand(N, 2)
#     a = np.zeros(N)  # all agents start facing right
#     v = np.c_[np.cos(a), np.sin(a)]
#     return x, v


def reflect_off_walls(pos, vel, L):
    """Specular reflection off the walls of the square arena [0, L] x [0, L].

    For each axis where the agent crossed a wall, the overshoot is mirrored
    back inside the arena and that velocity component is flipped
    (angle of incidence = angle of reflection).

    Args:
        pos (np.ndarray): (2,) proposed position after the step
        vel (np.ndarray): (2,) heading unit vector
        L (float): arena size

    Returns:
        tuple: (reflected position, reflected heading)
    """
    pos = pos.copy()
    vel = vel.copy()
    for ax in range(pos.shape[0]):
        if pos[ax] < 0.0:
            pos[ax] = -pos[ax]
            vel[ax] = -vel[ax]
        elif pos[ax] > L:
            pos[ax] = 2.0 * L - pos[ax]
            vel[ax] = -vel[ax]
    np.clip(pos, 0.0, L, out=pos)  # safety for steps larger than L
    return pos, vel
