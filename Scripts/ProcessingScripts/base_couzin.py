""" """

import numpy as np


def step_base_couzin(
    N,
    L,
    speed,
    dt,
    zor,
    zoo,
    zoa,
    max_turn_rate,
    x,
    v,
    current_time,
):
    """step function that computes interations

    Args:
        N (int): number of agents
        L (float): arena size, square
        speed (float): speed
        zor (float): zone of repulsion
        zoo (float): zone of orientation
        zoa (float): zone of attraction
        x (list): list of positions
        v (list): list of directions
    """
    new_v = np.zeros_like(v)  # placeholder for future vels

    for i in range(N):
        d = x - x[i]  # vector from agent to others
        d -= L * np.round(d / L)  # open arena/ change for bounded
        r = np.linalg.norm(d, axis=1)  # distance from agent to everyone else

        r[i] = np.inf  # distance to self is inflated :flush:
        unit = d / np.maximum(r[:, None], 1e-12)  # unit vector from agent to others

        close = r < zor  # agents to repel

        if close.any():  # if any repulsion do that
            desired = -unit[close].sum(axis=0)
        else:  # else do attractin plus orientation
            desired = v[(r < zoo)].sum(axis=0)  # checks and direction for orientation
            desired += unit[(r >= zoo) & (r < zoa)].sum(
                axis=0
            )  # checks and direction for attr

        length = np.linalg.norm(desired)  # check goal vector

        if length > 1e-12:  # len greateer than 0
            desired /= length

            angle = np.arctan2(
                v[i, 0] * desired[1] - v[i, 1] * desired[0],
                np.dot(v[i], desired),
            )  # compute tunring angel

            turn = np.clip(
                angle,
                -max_turn_rate * dt,
                max_turn_rate * dt,
            )  # cap the turning based on turning rate

            c = np.cos(turn)
            s = np.sin(turn)
            new_v[i] = np.array(
                [
                    c * v[i, 0] - s * v[i, 1],
                    s * v[i, 0] + c * v[i, 1],
                ]
            )  # turn agent as much as possible
        else:
            new_v[i] = v[i]

    v[:] = new_v
    x[:] = (x + speed * dt * v) % L
    return x, v
