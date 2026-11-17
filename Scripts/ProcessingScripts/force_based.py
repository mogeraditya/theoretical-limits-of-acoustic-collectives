#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created: 2026-11-13
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

Description
-----------
Force based model for flocking, based on Ferrante et al 2012

"""

import numpy as np
from ProcessingScripts.common_functions import reflect_off_walls


def force_based(
    N,
    L,
    v0,
    dt,
    zor,
    zoa,
    k,
    beta,
    alpha,
    x,
    v,
    current_time,
):
    """step function that computes interations

    Args:
        N (int): number of agents
        L (float): arena size, square
        v0 (float): cruise speed
        zor (float): zone of repulsion
        zoa (float): zone of attraction
        k = (float): stiffness
        beta (float): turning gain
        alpha (float): speed gain
        x (list): list of positions
        v (list): list of directions
    """
    new_v = np.zeros_like(v)  # placeholder for future vels
    new_x = np.zeros_like(x)  # placeholder for future positions

    for i in range(N):
        d = x - x[i]  # vector from agent to others
        d -= L * np.round(d / L)  # open arena/ change for bounded
        r = np.linalg.norm(d, axis=1)  # distance from agent to everyone else

        r[i] = np.inf  # distance to self is inflated :flush:
        unit = d / np.maximum(r[:, None], 1e-12)  # unit vector from agent to others

        close = r < zoa  # agents to repel

        if close.any():  # if any repulsion do that
            j = np.random.choice(np.flatnonzero(close))  # pick one True index
            desired = unit[j]  # (2,) unit vector to that neighbour

            angle = np.arctan2(
                v[i, 0] * desired[1] - v[i, 1] * desired[0],
                np.dot(v[i], desired),
            )  # compute turning angle

            f = k * (r[j] - zor) / zor
            turn = beta * f * np.sin(angle) * dt  # scale by stiffness
            c = np.cos(turn)
            s = np.sin(turn)

            new_v[i] = np.array(
                [
                    c * v[i, 0] - s * v[i, 1],
                    s * v[i, 0] + c * v[i, 1],
                ]
            )  # turn agent as much as possible

            heading_speed = alpha * f * np.dot(v[i], desired) + v0
        else:
            new_v[i] = v[i]
            heading_speed = v0

        new_x[i] = (
            x[i] + heading_speed * new_v[i] * dt
        ) % L  # wrap the position, not the step
    v[:] = new_v
    x[:] = new_x

    return x, v


def force_based_spherical(
    N,
    L,
    v0,
    dt,
    zor,
    zoa,
    max_turn_rate,
    k,
    beta,
    alpha,
    x,
    v,
    current_time,
):
    """step function that computes interations

    Args:
        N (int): number of agents
        L (float): arena size, square
        v0 (float): cruise speed
        zor (float): zone of repulsion
        zoa (float): zone of attraction
        max_turn_rate (float): maximum turning rate
        k = (float): stiffness
        beta (float): turning gain
        alpha (float): speed gain
        x (list): list of positions
        v (list): list of directions
    """
    new_v = np.zeros_like(v)  # placeholder for future vels
    new_x = np.zeros_like(x)  # placeholder for future positions

    for i in range(N):
        d = x - x[i]  # vector from agent to others
        d -= L * np.round(d / L)  # open arena/ change for bounded
        r = np.linalg.norm(d, axis=1)  # distance from agent to everyone else

        r[i] = np.inf  # distance to self is inflated :flush:
        unit = d / np.maximum(r[:, None], 1e-12)  # unit vector from agent to others

        close = r < zoa  # agents to repel\
        if close.any():  # if any repulsion do that
            j = np.random.choice(np.flatnonzero(close))  # pick one True index
            desired = unit[j]  # (2,) unit vector to that neighbour

            angle = np.arctan2(
                v[i, 0] * desired[1] - v[i, 1] * desired[0],
                np.dot(v[i], desired),
            )  # compute turning angle

            r_eff = max(r[j], 1e-2)  # avoid log(0)
            f = k * np.log(r_eff / zor)
            turn = np.clip(
                beta * f * angle * dt, -max_turn_rate * dt, max_turn_rate * dt
            )
            heading_speed = max(alpha * f * np.dot(v[i], desired) + v0, 0.0)

            # f = k * (r[j] - zor) / zor
            # turn = beta * f * np.sin(angle) * dt  # scale by stiffness
            c = np.cos(turn)
            s = np.sin(turn)

            new_v[i] = np.array(
                [
                    c * v[i, 0] - s * v[i, 1],
                    s * v[i, 0] + c * v[i, 1],
                ]
            )  # turn agent as much as possible

            # heading_speed = alpha * f * np.dot(v[i], desired) + v0
        else:
            new_v[i] = v[i]
            heading_speed = v0

        new_x[i] = (
            x[i] + heading_speed * new_v[i] * dt
        ) % L  # wrap the position, not the step
    v[:] = new_v
    x[:] = new_x

    return x, v


def force_based_spherical_fov(
    N,
    L,
    v0,
    dt,
    zor,
    zoa,
    max_turn_rate,
    fov,
    k,
    beta,
    alpha,
    x,
    v,
    current_time,
):
    """step function that computes interations

    Args:
        N (int): number of agents
        L (float): arena size, square
        v0 (float): cruise speed
        zor (float): zone of repulsion
        zoa (float): zone of attraction
        max_turn_rate (float): maximum turning rate
        fov (float): field of view
        k = (float): stiffness
        beta (float): turning gain
        alpha (float): speed gain
        x (list): list of positions
        v (list): list of directions
    """
    new_v = np.zeros_like(v)  # placeholder for future vels
    new_x = np.zeros_like(x)  # placeholder for future positions

    for i in range(N):
        d = x - x[i]  # vector from agent to others
        d -= L * np.round(d / L)  # open arena/ change for bounded
        r = np.linalg.norm(d, axis=1)  # distance from agent to everyone else

        r[i] = np.inf  # distance to self is inflated :flush:
        unit = d / np.maximum(r[:, None], 1e-12)  # unit vector from agent to others

        close = r < zoa  # agents to repel\

        visible = (unit @ v[i]) >= np.cos(fov)  # check if agent is within fov

        visible[i] = False  # The JOY of not self-reflecting :p

        if close.any() and visible.any():  # if any repulsion do that
            j = np.random.choice(np.flatnonzero(close))  # pick one True index
            desired = unit[j]  # (2,) unit vector to that neighbour

            angle = np.arctan2(
                v[i, 0] * desired[1] - v[i, 1] * desired[0],
                np.dot(v[i], desired),
            )  # compute turning angle

            r_eff = max(r[j], 1e-2)  # avoid log(0)
            f = k * np.log(r_eff / zor)
            turn = np.clip(
                beta * f * angle * dt, -max_turn_rate * dt, max_turn_rate * dt
            )
            # print(f"max_turn_rate * dt: {max_turn_rate * dt}, turn: {turn}")
            heading_speed = max(alpha * f * np.dot(v[i], desired) + v0, 0.0)

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
            heading_speed = v0

        new_x[i] = (
            x[i] + heading_speed * new_v[i] * dt
        ) % L  # wrap the position, not the step
    v[:] = new_v
    x[:] = new_x

    return x, v


def force_based_spherical_fov_w_duty_cycle(
    N,
    L,
    v0,
    dt,
    zor,
    zoa,
    max_turn_rate,
    fov,
    k,
    beta,
    alpha,
    sample_times,
    ipi,
    ipi_noise,
    duty_cycle_proportion,
    boundary_condition,
    x,
    v,
    current_time,
):
    """step function that computes interations

    Args:
        N (int): number of agents
        L (float): arena size, square
        v0 (float): cruise speed
        zor (float): zone of repulsion
        zoa (float): zone of attraction
        max_turn_rate (float): maximum turning rate
        fov (float): field of view
        sample_times (list): list of times when each agent samples the environment
        ipi (float): inter-pulse interval
        ipi_noise (float): noise in the inter-pulse interval
        duty_cycle_proportion (float): proportion of time the agent listenes
        boundary_condition (str): "open" or "bounded"
        k = (float): stiffness
        beta (float): turning gain
        alpha (float): speed gain
        x (list): list of positions
        v (list): list of directions

    """
    new_v = np.zeros_like(v)  # placeholder for future vels
    new_x = np.zeros_like(x)  # placeholder for future positions

    for i in range(N):
        # default between calls: keep heading, cruise speed
        new_v[i] = v[i]
        heading_speed = v0

        if current_time >= sample_times[i]:

            d = x - x[i]  # vector from agent to others
            if boundary_condition == "open":
                d -= L * np.round(d / L)  # open arena/ change for bounded-
            r = np.linalg.norm(d, axis=1)  # distance from agent to everyone else

            r[i] = np.inf  # distance to self is inflated :flush:
            unit = d / np.maximum(r[:, None], 1e-12)  # unit vector from agent to others

            visible = (unit @ v[i]) >= np.cos(fov)  # check if agent is within fov

            visible[i] = False  # The JOY of not self-reflecting :p
            duty_cycle_based_mask = np.random.rand(N) < duty_cycle_proportion
            candidates = (
                (r < zoa) & visible & duty_cycle_based_mask
            )  # in range AND in fov

            if candidates.any():
                j = np.random.choice(np.flatnonzero(candidates))  # pick one True index
                desired = unit[j]  # (2,) unit vector to that neighbour

                angle = np.arctan2(
                    v[i, 0] * desired[1] - v[i, 1] * desired[0],
                    np.dot(v[i], desired),
                )  # compute turning angle

                r_eff = max(r[j], 1e-2)  # avoid log(0)
                f = k * np.log(r_eff / zor)
                turn = np.clip(
                    beta * f * angle * dt, -max_turn_rate * dt, max_turn_rate * dt
                )
                heading_speed = max(alpha * f * np.dot(v[i], desired) + v0, 0.0)
                if heading_speed > 0.15:
                    # print(f"Agent {i} heading_speed: {heading_speed}")
                    heading_speed = 0.15

                c = np.cos(turn)
                s = np.sin(turn)

                new_v[i] = np.array(
                    [
                        c * v[i, 0] - s * v[i, 1],
                        s * v[i, 0] + c * v[i, 1],
                    ]
                )  # turn agent as much as possible

            # schedule next call from the planned time (no drift)
            sample_times[i] += max(
                1e-6,
                ipi + np.random.uniform(-ipi_noise, ipi_noise),
            )

        if boundary_condition == "open":
            new_x[i] = (
                x[i] + heading_speed * new_v[i] * dt
            ) % L  # wrap the position, not the step
        elif boundary_condition == "bounded":
            new_x[i], new_v[i] = reflect_off_walls(
                x[i] + heading_speed * new_v[i] * dt, new_v[i], L
            )
        else:
            raise ValueError(
                f"Invalid boundary_condition: {boundary_condition}. Must be 'open' or 'bounded'."
            )
    v[:] = new_v
    x[:] = new_x

    return x, v


def force_based_spherical_fov_w_duty_cycle_directionality(
    N,
    L,
    v0,
    dt,
    zor,
    zoa,
    max_turn_rate,
    fov,
    k,
    beta,
    alpha,
    sample_times,
    ipi,
    ipi_noise,
    duty_cycle_proportion,
    boundary_condition,
    x,
    v,
    current_time,
):
    """step function that computes interations

    Args:
        N (int): number of agents
        L (float): arena size, square
        v0 (float): cruise speed
        zor (float): zone of repulsion
        zoa (float): zone of attraction
        max_turn_rate (float): maximum turning rate
        fov (float): field of view
        sample_times (list): list of times when each agent samples the environment
        ipi (float): inter-pulse interval
        ipi_noise (float): noise in the inter-pulse interval
        duty_cycle_proportion (float): proportion of time the agent listenes
        boundary_condition (str): "open" or "bounded"
        k = (float): stiffness
        beta (float): turning gain
        alpha (float): speed gain
        x (list): list of positions
        v (list): list of directions

    """
    new_v = np.zeros_like(v)  # placeholder for future vels
    new_x = np.zeros_like(x)  # placeholder for future positions

    for i in range(N):
        # default between calls: keep heading, cruise speed
        new_v[i] = v[i]
        heading_speed = v0

        if current_time >= sample_times[i]:

            d = x - x[i]  # vector from agent to others
            if boundary_condition == "open":
                d -= L * np.round(d / L)  # open arena/ change for bounded-
            r = np.linalg.norm(d, axis=1)  # distance from agent to everyone else

            r[i] = np.inf  # distance to self is inflated :flush:
            unit = d / np.maximum(r[:, None], 1e-12)  # unit vector from agent to others

            visible = (unit @ v[i]) >= np.cos(fov)  # check if agent is within fov

            visible[i] = False  # The JOY of not self-reflecting :p
            duty_cycle_based_mask = np.random.rand(N) < duty_cycle_proportion
            # directionality_mask = (unit @ v[i]) <= np.cos(
            #     np.pi / 2
            # )  # only consider agents in back

            # probabilistic back-bias: P(keep) peaks for agents directly behind (pi)
            # and decays as a gaussian towards the front, never below front_floor
            back_sigma = np.pi / 2
            front_floor = 0.2
            bearing = np.arccos(np.clip(unit @ v[i], -1.0, 1.0))  # 0=front, pi=back
            p_keep = front_floor + (1 - front_floor) * np.exp(
                -0.5 * ((bearing - np.pi) / back_sigma) ** 2
            )
            directionality_mask = np.random.rand(N) < p_keep

            candidates = (
                (r < zoa) & visible & duty_cycle_based_mask & directionality_mask
            )  # in range AND in fov

            if candidates.any():  # if any repulsion do that
                j = np.random.choice(np.flatnonzero(candidates))  # pick one True index
                desired = unit[j]  # (2,) unit vector to that neighbour

                angle = np.arctan2(
                    v[i, 0] * desired[1] - v[i, 1] * desired[0],
                    np.dot(v[i], desired),
                )  # compute turning angle

                # print(f"Agent {i} angle: {np.rad2deg(angle):.2f} deg")

                # noisy range measurement: uniform error up to +/- 50% of true distance
                distance_noise = 0.5
                r_meas = r[j] * (1 + np.random.uniform(-distance_noise, distance_noise))
                r_eff = max(r_meas, 1e-2)  # avoid log(0)
                f = k * np.log(r_eff / zor)
                turn = np.clip(
                    beta * f * angle * dt, -max_turn_rate * dt, max_turn_rate * dt
                )
                heading_speed = max(alpha * f * np.dot(v[i], desired) + v0, 0.0)
                if heading_speed > 0.15:
                    # print(f"Agent {i} heading_speed: {heading_speed}")
                    heading_speed = 0.15

                c = np.cos(turn)
                s = np.sin(turn)

                new_v[i] = np.array(
                    [
                        c * v[i, 0] - s * v[i, 1],
                        s * v[i, 0] + c * v[i, 1],
                    ]
                )  # turn agent as much as possible

            # schedule next call from the planned time (no drift)
            sample_times[i] += max(
                1e-6,
                ipi + np.random.uniform(-ipi_noise, ipi_noise),
            )

        if boundary_condition == "open":
            new_x[i] = (
                x[i] + heading_speed * new_v[i] * dt
            ) % L  # wrap the position, not the step
        elif boundary_condition == "bounded":
            new_x[i], new_v[i] = reflect_off_walls(
                x[i] + heading_speed * new_v[i] * dt, new_v[i], L
            )
        else:
            raise ValueError(
                f"Invalid boundary_condition: {boundary_condition}. Must be 'open' or 'bounded'."
            )
    v[:] = new_v
    x[:] = new_x

    return x, v
