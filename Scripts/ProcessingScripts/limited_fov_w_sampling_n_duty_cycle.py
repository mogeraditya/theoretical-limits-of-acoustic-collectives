import numpy as np


def step_w_fov_w_sampling_w_noise(
    N,
    L,
    speed,
    dt,
    zor,
    zoo,
    zoa,
    max_turn_rate,
    fov,
    sample_times,
    desired_directions,
    d,
    d_noise,
    duty_cycle_proportion,
    x,
    v,
    current_time,
):
    new_v = np.zeros_like(v)

    for i in range(N):

        if current_time >= sample_times[i]:
            vector_to_agents = x - x[i]  # vector from agent to others
            vector_to_agents -= L * np.round(
                vector_to_agents / L
            )  # open arena/ change for bounded
            r = np.linalg.norm(
                vector_to_agents, axis=1
            )  # distance from agent to everyone else

            r[i] = np.inf  # distance to self is inflated :flush:
            unit = vector_to_agents / np.maximum(
                r[:, None], 1e-12
            )  # unit vector from agent to others

            visible = (unit @ v[i]) >= np.cos(fov)  # check if agent is within fov

            visible[i] = False  # The JOY of not self-reflecting :p

            # adding the noisy calling here
            #   basically it is the same as if the probability that an agent/
            #   is sampled is like n bernoulli trials, where p is proportional/
            #   to duty cycle (call duration, ipi and listening time)/

            duty_cycle_based_mask = (
                np.random.rand(N) < duty_cycle_proportion
            )  # mask on duty cycle probability
            repulsion = (r < zor) & visible & duty_cycle_based_mask  # repulsion check
            orientation = (
                (r >= zor) & (r < zoo) & visible & duty_cycle_based_mask
            )  # orientation check
            attraction = (
                (r >= zoo) & (r < zoa) & visible & duty_cycle_based_mask
            )  # attraction check

            if repulsion.any():
                # repulsion dir computaiton
                desired = -unit[repulsion].sum(axis=0)
            else:
                # checks and direction for attr and orientaiton
                desired = v[orientation].sum(axis=0) + unit[attraction].sum(axis=0)

            length = np.linalg.norm(desired)  # normalize dir
            desired_directions[i] = (
                desired / length if length > 1e-12 else v[i]
            )  # normalize

            sample_times[i] = current_time + max(
                1e-6,
                d + np.random.uniform(-d_noise, d_noise),
            )

        desired = desired_directions[i]

        angle = np.arctan2(
            v[i, 0] * desired[1] - v[i, 1] * desired[0],
            np.dot(v[i], desired),
        )  # total angle agent will turn

        turn = np.clip(angle, -max_turn_rate * dt, max_turn_rate * dt)
        # turning agenl capped by the turning speed

        c, s = np.cos(turn), np.sin(turn)
        new_v[i] = [
            c * v[i, 0] - s * v[i, 1],
            s * v[i, 0] + c * v[i, 1],
        ]  # make agent turn

    v[:] = new_v
    x[:] = (x + speed * dt * v) % L
    return x, v
