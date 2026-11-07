import sys

import numpy as np

sys.path.append("./Scripts/ProcessingScripts/")
from base_couzin import step_base_couzin
from common_functions import initialize
from limited_fov import step_w_fov
from limited_fov_w_sampling import step_w_fov_w_sampling
from limited_fov_w_sampling_n_duty_cycle import step_w_fov_w_sampling_w_noise
from run_and_visualize_w_scores import generate_visualiation

# all the parameters of importance, ignore the others
N, L = 20, 20  # number of agetns, arena size
speed, dt = 5, 0.01  # speed, time step size
duration = 50  # duration of the video
zor, zoo, zoa = 2, 2, 4  # zones of repulsion, orientation, attraction in m
max_turn_rate = 3  # turning rate in rad/s
fov = np.pi / 4  # frontal range of sample, +/- fov

ipi = 1  # ipi in seconds
ipi_noise = 0.01  # jitter in ipi in seconds

small_ipi = 0.1  # more freq sampling exanple
smaller_ipi = 0.05  # very freqeunt samples ! like the bats lowk

duty_cycle_proportion = 0.5  # percent of ipi the agent listens for

# dont worry about this for now :p

x, v = initialize(N, L)
model_input = [N, L, speed, dt, 0, 0, 0, max_turn_rate]
generate_visualiation(
    x, v, step_base_couzin, model_input, duration=duration, name="control"
)

x, v = initialize(N, L)
model_input = [N, L, speed, dt, zor, zoo, zoa, max_turn_rate]
generate_visualiation(
    x, v, step_base_couzin, model_input, duration=duration, name="base couzin"
)

x, v = initialize(N, L)
model_input.extend([fov])
generate_visualiation(
    x, v, step_w_fov, model_input, duration=duration, name="limited fov"
)

x, v = initialize(N, L)
sample_times = np.random.uniform(0, ipi, N)
desired_directions = v.copy()

model_input_copy = model_input.copy()
model_input_copy.extend([sample_times, desired_directions, ipi, ipi_noise])

generate_visualiation(
    x,
    v,
    step_w_fov_w_sampling,
    model_input_copy,
    duration=duration,
    name=f"limited fov with sparse sampling, ipi {ipi}s",
)


x, v = initialize(N, L)
sample_times = np.random.uniform(0, small_ipi, N)
desired_directions = v.copy()

model_input_copy = model_input.copy()
model_input_copy.extend([sample_times, desired_directions, small_ipi, ipi_noise])

generate_visualiation(
    x,
    v,
    step_w_fov_w_sampling,
    model_input_copy,
    duration=duration,
    name=f"limited fov with sparse sampling, ipi {small_ipi}s",
)

x, v = initialize(N, L)
sample_times = np.random.uniform(0, smaller_ipi, N)
desired_directions = v.copy()

model_input_copy = model_input.copy()
model_input_copy.extend([sample_times, desired_directions, smaller_ipi, ipi_noise])

generate_visualiation(
    x,
    v,
    step_w_fov_w_sampling,
    model_input_copy,
    duration=duration,
    name=f"limited fov with sparse sampling, ipi {smaller_ipi}s",
)


##### WITH NOISEEEE #######


x, v = initialize(N, L)
sample_times = np.random.uniform(0, ipi, N)
desired_directions = v.copy()

model_input_copy = model_input.copy()
model_input_copy.extend(
    [sample_times, desired_directions, ipi, ipi_noise, duty_cycle_proportion]
)

generate_visualiation(
    x,
    v,
    step_w_fov_w_sampling_w_noise,
    model_input_copy,
    duration=duration,
    name=f"limited fov with sampling n noise, ipi {ipi}s",
)


x, v = initialize(N, L)
sample_times = np.random.uniform(0, small_ipi, N)
desired_directions = v.copy()

model_input_copy = model_input.copy()
model_input_copy.extend(
    [sample_times, desired_directions, small_ipi, ipi_noise, duty_cycle_proportion]
)

generate_visualiation(
    x,
    v,
    step_w_fov_w_sampling_w_noise,
    model_input_copy,
    duration=duration,
    name=f"limited fov with sampling n noise, ipi {small_ipi}s",
)

x, v = initialize(N, L)
sample_times = np.random.uniform(0, smaller_ipi, N)
desired_directions = v.copy()

model_input_copy = model_input.copy()
model_input_copy.extend(
    [sample_times, desired_directions, smaller_ipi, ipi_noise, duty_cycle_proportion]
)

generate_visualiation(
    x,
    v,
    step_w_fov_w_sampling_w_noise,
    model_input_copy,
    duration=duration,
    name=f"limited fov with sampling n noise, ipi {smaller_ipi}s",
)
