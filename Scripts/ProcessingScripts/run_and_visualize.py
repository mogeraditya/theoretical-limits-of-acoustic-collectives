import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon


def generate_visualiation(
    x, v, model, model_input, duration=10000, name="couzin_simulation"
):
    fig, ax = plt.subplots()
    ax.set(xlim=(0, model_input[1]), ylim=(0, model_input[1]), aspect="equal")
    ax.set_title(f"couzin model dimulation: version {name}")
    frames = int(duration / model_input[3])
    size = 1
    triangles = [
        Polygon(np.zeros((3, 2)), color="blue") for agent in range(model_input[0])
    ]
    current_time = 0
    for triangle in triangles:
        ax.add_patch(triangle)

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
        current_time += model_input[3]
        return triangles

    for i in range(model_input[0]):
        draw_the_triangle(i)

    animation = FuncAnimation(
        fig,
        animate,
        frames=frames,
        interval=30,
        cache_frame_data=False,
    )
    animation.save(
        f"./Output/w_o_scores/{name}.mp4",
        writer="ffmpeg",
        fps=30,
    )
    plt.clf()
    # plt.show()
