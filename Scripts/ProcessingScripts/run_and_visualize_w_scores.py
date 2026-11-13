import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon


def generate_visualiation(
    x, v, model, model_input, duration=10000, name="couzin_simulation", size=0.5
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.set(xlim=(0, model_input[1]), ylim=(0, model_input[1]), aspect="equal")
    ax1.set_title(f"couzin model simulation: {name}")
    ax2.set_xlim(0, duration)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("pol timesreis")
    ax2.set_xlabel("time")
    ax2.set_ylabel("polarization")

    frames = int(duration / model_input[3])
    triangles = [
        plt.Polygon(np.zeros((3, 2)), color="blue") for _ in range(model_input[0])
    ]
    for triangle in triangles:
        ax1.add_patch(triangle)

    time_steps = []
    global_polarization = []
    local_polarization = []

    def draw_the_triangle(i):
        front = v[i]
        side = np.array([-front[1], front[0]])
        triangles[i].set_xy(
            [
                x[i] + size * front,
                x[i] - size * front / 2 + size * side / 2,
                x[i] - size * front / 2 - size * side / 2,
            ]
        )

    def animate(_):
        nonlocal current_time
        model(*model_input, x, v, current_time)

        for i in range(model_input[0]):
            draw_the_triangle(i)

        global_p = np.linalg.norm(v.sum(axis=0)) / model_input[0]
        global_polarization.append(global_p)

        local_p = []
        zoa = model_input[6]
        for i in range(model_input[0]):
            dist = np.linalg.norm(x - x[i], axis=1)
            neighbors = dist < zoa if zoa > 1e-12 else dist < 2
            neighbors[i] = False
            if np.any(neighbors):
                local_p.append(
                    np.linalg.norm(v[neighbors].sum(axis=0)) / np.sum(neighbors)
                )
            else:
                local_p.append(0.0)
        local_polarization.append(np.mean(local_p))

        time_steps.append(current_time)

        ax2.clear()
        ax2.plot(time_steps, global_polarization, "b-", label="global")
        ax2.plot(time_steps, local_polarization, "r-", label="local; r=zoa")
        ax2.set_xlim(0, current_time + 10)
        ax2.set_ylim(0, 1.1)
        ax2.set_title("pol timesries")
        ax2.set_xlabel("time")
        ax2.set_ylabel("polarization")
        ax2.legend()

        current_time += model_input[3]
        return triangles

    current_time = 0
    for i in range(model_input[0]):
        draw_the_triangle(i)

    animation = FuncAnimation(
        fig, animate, frames=frames, interval=30, blit=False, cache_frame_data=False
    )
    animation.save(
        f"./Output/w_scores/{model_input[0]}_agents/{name}.mp4", writer="ffmpeg", fps=30
    )
    plt.close(fig)
