import numpy as np


def step_w_fov(
    N,
    L,
    speed,
    dt,
    zor,
    zoo,
    zoa,
    max_turn_rate,
    fov,
    x,
    v,
    current_time,
):
    """Update agents with a limited forward field of view.

    Args:
        fov (float): viewing angle in radians.
    """
    new_v = np.zeros_like(v)

    for i in range(N):
        d = x - x[i]  # vector from agent to others
        d -= L * np.round(d / L)  # open arena/ change for bounded
        r = np.linalg.norm(d, axis=1)  # distance from agent to everyone else

        r[i] = np.inf  # distance to self is inflated :flush:
        unit = d / np.maximum(r[:, None], 1e-12)  # unit vector from agent to others

        visible = (unit @ v[i]) >= np.cos(fov)  # check if agent is within fov

        visible[i] = False  # The JOY of not self-reflecting :p

        repulsion = (r < zor) & visible  # repulsion check
        orientation = (r >= zor) & (r < zoo) & visible  # orientation check
        attraction = (r >= zoo) & (r < zoa) & visible  # attraction check

        if repulsion.any():
            # repulsion dir computaiton
            desired = -unit[repulsion].sum(axis=0)
        else:
            # checks and direction for attr and orientaiton
            desired = v[orientation].sum(axis=0) + unit[attraction].sum(axis=0)

        length = np.linalg.norm(desired)

        if length > 1e-12:
            desired /= length

            angle = np.arctan2(
                v[i, 0] * desired[1] - v[i, 1] * desired[0],
                np.dot(v[i], desired),
            )

            turn = np.clip(
                angle,
                -max_turn_rate * dt,
                max_turn_rate * dt,
            )

            c = np.cos(turn)
            s = np.sin(turn)
            new_v[i] = np.array(
                [
                    c * v[i, 0] - s * v[i, 1],
                    s * v[i, 0] + c * v[i, 1],
                ]
            )
        else:
            new_v[i] = v[i]

    v[:] = new_v
    x[:] = (x + speed * dt * v) % L
    return x, v
