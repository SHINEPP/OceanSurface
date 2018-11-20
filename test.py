# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
from OceanSurface import OceanSurface

ocean = OceanSurface()
m, n = ocean.get_size()
count = m * n

x_data = [0] * count
y_data = [0] * count
z_data = [0] * count

time_i = [0]


def update_graph(num):
    time_i[0] += 1
    ocean.build_field(time_i[0])

    for i in range(0, count):
        p = ocean.height_field[i]
        x_data[i] = p[0] / 1000.0 + 0.5
        y_data[i] = p[2] / 1000.0 + 0.5
        z_data[i] = p[1] / 1000.0 + 0.5

    graph.set_data(x_data, y_data)
    graph.set_3d_properties(z_data)
    title.set_text('SPH Test, time={}'.format(num))
    return title, graph,


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('SPH Test')
ax.set_xlabel('x')
ax.set_ylabel('y')

graph, = ax.plot([], [], [], linestyle="", marker=".")

ani = matplotlib.animation.FuncAnimation(fig, update_graph, 10, interval=100, blit=True)

plt.show()
