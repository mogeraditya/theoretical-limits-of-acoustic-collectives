import sys

import numpy as np

sys.path.append("./Scripts/ProcessingScripts/")
from base_couzin import step_base_couzin
from common_functions import initialize
from limited_fov import step_w_fov
from limited_fov_w_sampling import step_w_fov_w_sampling
from limited_fov_w_sampling_n_duty_cycle import step_w_fov_w_sampling_w_noise
from run_and_visualize_w_scores import generate_visualiation
from variable_fov_w_sampling import step_w_variable_fov_w_sampling
from variable_fov_w_sampling_n_duty_cycle import step_w_variable_fov_w_sampling_w_noise
from force_based import (
    force_based,
    force_based_spherical,
    force_based_spherical_fov,
    force_based_spherical_fov_w_duty_cycle,
    force_based_spherical_fov_w_duty_cycle_directionality,
)

# all the parameters of importance, ignore the others
N, L = 7, 5  # number of agetns, arena size
speed, dt = 0.05, 0.16  # speed, time step size
duration = 100  # duration of the video
zor, zoo, zoa = 0.3, 1, 1  # zones of repulsion, orientation, attraction in m
max_turn_rate = 3  # turning rate in rad/s
fov = 8 * np.pi / 8  # frontal range of sample, +/- fov

ipi = 0.15  # ipi in seconds
ipi_noise = 0.08  # jitter in ipi in seconds

# small_ipi = 0.1  # more freq sampling exanple
# smaller_ipi = 0.05  # very freqeunt samples ! like the bats lowk

duty_cycle_proportion = 0.6  # percent of ipi the agent listens for

boundary_condition = "bounded"  # "open" or "bounded"
# dont worry about this for now :p

print(f"Running with {N} agents, {L}m arena, {duration}s duration \n")

# print("Control start...\n")
# x, v = initialize(N, L)
# model_input = [N, L, speed, dt, 0, 0, 0, max_turn_rate]
# generate_visualiation(
#     x, v, step_base_couzin, model_input, duration=duration, name="control"
# )

print("Force based start...\n")


x, v = initialize(N, L)


# x, v = initialize(N, L)
# model_input = [N, L, speed, dt, zor, zoo, zoa, max_turn_rate]
# generate_visualiation(
#     x, v, step_base_couzin, model_input, duration=duration, name="base couzin"
# )


# model_input = [N, L, speed, dt, zor, zoo, zoa, max_turn_rate]
# x, v = initialize(N, L)
# model_input.extend([fov])
# generate_visualiation(
#     x, v, step_w_fov, model_input, duration=duration, name="limited fov"
# )
########################################################################333

# N (int): number of agents
# L (float): arena size, square
# v0 (float): cruise speed
# zor (float): zone of repulsion
# zoa (float): zone of attraction
# max_turn_rate (float): maximum turning rate
# k = (float): stiffness
# beta (float): turning gain
# alpha (float): speed gain
# x (list): list of positions
# v (list): list of directions
# boundary_condition (str): "open" or "bounded"

x, v = initialize(N, L)
# model_input = [N, L, speed, dt, zor, zoa, max_turn_rate, fov, 0, 0.05, 0.2]

# generate_visualiation(
#     x,
#     v,
#     force_based_spherical_fov,
#     model_input,
#     duration=duration,
#     name=f"force based random choice spherical fov \n fov {np.rad2deg(fov):.2f} deg, zor {zor}m, zoa {zoa}m, max turn rate {max_turn_rate} rad/s,\n stiffness (k): {model_input[8]}, turn gain (beta): {model_input[9]}, speed gain (alpha): {model_input[10]}",
# )


x, v = initialize(N, L)
sample_times = np.random.uniform(0, ipi, N)
model_input = [N, L, speed, dt, zor, zoa, max_turn_rate, fov, 1, 0.05, 0.2]
model_input_copy = model_input.copy()
model_input_copy.extend(
    [sample_times, ipi, ipi_noise, duty_cycle_proportion, boundary_condition]
)

generate_visualiation(
    x,
    v,
    force_based_spherical_fov_w_duty_cycle,
    model_input_copy,
    duration=duration,
    name=f"force based random choice spherical fov and duty cycle \n fov {np.rad2deg(fov):.2f} deg, zor {zor}m, zoa {zoa}m, max turn rate {max_turn_rate} rad/s,\n stiffness (k): {model_input[8]}, turn gain (beta): {model_input[9]}, speed gain (alpha): {model_input[10]}, ipi: {ipi}s, jitter: {ipi_noise}s",
)


x, v = initialize(N, L)
sample_times = np.random.uniform(0, ipi, N)
model_input = [N, L, speed, dt, zor, zoa, max_turn_rate, fov, 1, 0.05, 0.2]
model_input_copy = model_input.copy()
model_input_copy.extend(
    [sample_times, ipi, ipi_noise, duty_cycle_proportion, boundary_condition]
)

generate_visualiation(
    x,
    v,
    force_based_spherical_fov_w_duty_cycle_directionality,
    model_input_copy,
    duration=duration,
    name=f"force based random choice spherical fov and duty cycle \n fov {np.rad2deg(fov):.2f} deg, zor {zor}m, zoa {zoa}m, max turn rate {max_turn_rate} rad/s,\n stiffness (k): {model_input[8]}, turn gain (beta): {model_input[9]}, speed gain (alpha): {model_input[10]}, ipi: {ipi}s, jitter: {ipi_noise}s",
)


# x, v = initialize(N, L)
# sample_times = np.random.uniform(0, ipi, N)
# desired_directions = v.copy()

# model_input_copy = model_input.copy()
# model_input_copy.extend([sample_times, desired_directions, ipi, ipi_noise])

# generate_visualiation(
#     x,
#     v,
#     step_w_fov_w_sampling,
#     model_input_copy,
#     duration=duration,
#     name=f"limited fov with sparse sampling, ipi {ipi}s",
# )


# x, v = initialize(N, L)
# sample_times = np.random.uniform(0, small_ipi, N)
# desired_directions = v.copy()

# model_input_copy = model_input.copy()
# model_input_copy.extend([sample_times, desired_directions, small_ipi, ipi_noise])

# generate_visualiation(
#     x,
#     v,
#     step_w_fov_w_sampling,
#     model_input_copy,
#     duration=duration,
#     name=f"limited fov with sparse sampling, ipi {small_ipi}s",
# )

# x, v = initialize(N, L)
# sample_times = np.random.uniform(0, smaller_ipi, N)
# desired_directions = v.copy()

# model_input_copy = model_input.copy()
# model_input_copy.extend([sample_times, desired_directions, smaller_ipi, ipi_noise])

# generate_visualiation(
#     x,
#     v,
#     step_w_fov_w_sampling,
#     model_input_copy,
#     duration=duration,
#     name=f"limited fov with sparse sampling, ipi {smaller_ipi}s",
# )


# #### WITH NOISEEEE #######


# x, v = initialize(N, L)
# sample_times = np.random.uniform(0, ipi, N)
# desired_directions = v.copy()

# model_input_copy = model_input.copy()
# model_input_copy.extend(
#     [sample_times, desired_directions, ipi, ipi_noise, duty_cycle_proportion]
# )

# generate_visualiation(
#     x,
#     v,
#     step_w_fov_w_sampling_w_noise,
#     model_input_copy,
#     duration=duration,
#     name=f"limited fov with sampling n noise, ipi {ipi}s",
# )


# x, v = initialize(N, L)
# sample_times = np.random.uniform(0, small_ipi, N)
# desired_directions = v.copy()

# model_input_copy = model_input.copy()
# model_input_copy.extend(
#     [sample_times, desired_directions, small_ipi, ipi_noise, duty_cycle_proportion]
# )

# generate_visualiation(
#     x,
#     v,
#     step_w_fov_w_sampling_w_noise,
#     model_input_copy,
#     duration=duration,
#     name=f"limited fov with sampling n noise, ipi {small_ipi}s",
# )

# x, v = initialize(N, L)
# sample_times = np.random.uniform(0, smaller_ipi, N)
# desired_directions = v.copy()

# model_input_copy = model_input.copy()
# model_input_copy.extend(
#     [sample_times, desired_directions, smaller_ipi, ipi_noise, duty_cycle_proportion]
# )

# generate_visualiation(
#     x,
#     v,
#     step_w_fov_w_sampling_w_noise,
#     model_input_copy,
#     duration=duration,
#     name=f"limited fov with sampling n noise, ipi {smaller_ipi}s",
# )


# #### VAIRABLE FOV


# x, v = initialize(N, L)
# sample_times = np.random.uniform(0, ipi, N)
# desired_directions = v.copy()

# model_input_copy = model_input.copy()
# model_input_copy.extend(
#     [sample_times, desired_directions, ipi, ipi_noise, duty_cycle_proportion]
# )

# generate_visualiation(
#     x,
#     v,
#     step_w_variable_fov_w_sampling_w_noise,
#     model_input_copy,
#     duration=duration,
#     name=f"variable fov with sampling n noise, ipi {ipi}s",
# )


# x, v = initialize(N, L)
# sample_times = np.random.uniform(0, small_ipi, N)
# desired_directions = v.copy()

# model_input_copy = model_input.copy()
# model_input_copy.extend(
#     [sample_times, desired_directions, small_ipi, ipi_noise, duty_cycle_proportion]
# )

# generate_visualiation(
#     x,
#     v,
#     step_w_variable_fov_w_sampling_w_noise,
#     model_input_copy,
#     duration=duration,
#     name=f"variable fov with sampling n noise, ipi {small_ipi}s",
# )

# x, v = initialize(N, L)
# sample_times = np.random.uniform(0, smaller_ipi, N)
# desired_directions = v.copy()

# model_input_copy = model_input.copy()
# model_input_copy.extend(
#     [sample_times, desired_directions, smaller_ipi, ipi_noise, duty_cycle_proportion]
# )

# generate_visualiation(
#     x,
#     v,
#     step_w_variable_fov_w_sampling_w_noise,
#     model_input_copy,
#     duration=duration,
#     name=f"variable fov with sampling n noise, ipi {smaller_ipi}s",
# )

# x, v = initialize(N, L)
# sample_times = np.random.uniform(0, ipi, N)
# desired_directions = v.copy()

# model_input_copy = model_input.copy()
# model_input_copy.extend([sample_times, desired_directions, ipi, ipi_noise])

# generate_visualiation(
#     x,
#     v,
#     step_w_variable_fov_w_sampling,
#     model_input_copy,
#     duration=duration,
#     name=f"variable fov with sparse sampling, ipi {ipi}s",
# )

# x, v = initialize(N, L)
# sample_times = np.random.uniform(0, small_ipi, N)
# desired_directions = v.copy()

# model_input_copy = model_input.copy()
# model_input_copy.extend([sample_times, desired_directions, small_ipi, ipi_noise])

# generate_visualiation(
#     x,
#     v,
#     step_w_variable_fov_w_sampling,
#     model_input_copy,
#     duration=duration,
#     name=f"variable fov with sparse sampling, ipi {small_ipi}s",
# )

# x, v = initialize(N, L)
# sample_times = np.random.uniform(0, smaller_ipi, N)
# desired_directions = v.copy()

# model_input_copy = model_input.copy()
# model_input_copy.extend([sample_times, desired_directions, smaller_ipi, ipi_noise])

# generate_visualiation(
#     x,
#     v,
#     step_w_variable_fov_w_sampling,
#     model_input_copy,
#     duration=duration,
#     name=f"variable fov with sparse sampling, ipi {smaller_ipi}s",
# )
