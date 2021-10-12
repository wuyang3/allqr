import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import subprocess
import sys


def main(argv):
    # process = subprocess.Popen("./main", shell=False)
    state, output_string = subprocess.getstatusoutput('./main')

    df = pd.read_csv('states.txt', header=None)
    states = df.to_numpy()

    time_s = states[:, 0]
    x = states[:, 1]
    y = states[:, 2]
    h = states[:, 3]
    v = states[:, 4]

    df = pd.read_csv('controls.txt', header=None)
    controls = df.to_numpy()

    time_c = controls[:, 0]
    delta = controls[:, 1]
    acc = controls[:, 2]

    figure, ax, = plt.subplots(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 3)
    ax.set_title('Extended LQR Motion Planner', pad=20)
    ax.title.set_fontsize(20)
    ax.axis('off')

    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=0.5, hspace=0.5)

    ax1 = figure.add_subplot(gs[0, 0:2])
    ax1.title.set_text('vehicle trajectory')
    ax1.title.set_fontsize(12)
    ax1.grid()
    ax1.set_aspect('equal')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_xlim([0, np.max(x)])
    ax1.set_ylim([-7.4, 7.4])

    traj_plot, = ax1.plot([], [], 'r.', markersize=5)
    traj_plot.set_xdata(x)
    traj_plot.set_ydata(y)

    ax2 = figure.add_subplot(gs[1, 0:1])
    ax2.title.set_text('vehicle heading')
    ax2.title.set_fontsize(12)
    ax2.grid()
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('heading angle (rad)')
    ax2.set_xlim([0, np.max(time_s)])
    ax2.set_ylim([-0.3, 0.3])

    heading_plot, = ax2.plot([], [], 'r.', markersize=5)
    heading_plot.set_xdata(time_s)
    heading_plot.set_ydata(h)

    ax3 = figure.add_subplot(gs[1, 1:2])
    ax3.title.set_text('vehicle velocity')
    ax3.title.set_fontsize(12)
    ax3.grid()
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('velocity (m/s)')
    ax3.set_xlim([0, np.max(time_s)])
    ax3.set_ylim([-1, 11])

    speed_plot, = ax3.plot([], [], 'r.', markersize=5)
    speed_plot.set_xdata(time_s)
    speed_plot.set_ydata(v)

    ax4 = figure.add_subplot(gs[0, 2:3])
    ax4.title.set_text('front wheel angle')
    ax4.title.set_fontsize(12)
    ax4.grid()
    ax4.set_xlabel('time (s)')
    ax4.set_ylabel('front wheel angle (rad)')
    ax4.set_xlim([0, np.max(time_c)])
    ax4.set_ylim([-0.3, 0.3])

    steer_plot, = ax4.plot([], [], 'r.', markersize=2)
    steer_plot.set_xdata(time_c)
    steer_plot.set_ydata(delta)

    ax5 = figure.add_subplot(gs[1, 2:3])
    ax5.title.set_text('acceleration')
    ax5.title.set_fontsize(12)
    ax5.grid()
    ax5.set_xlabel('time (s)')
    ax5.set_ylabel('acceleration (m/s^2)')
    ax5.set_xlim([0, np.max(time_c)])
    ax5.set_ylim([-5, 5])

    acc_plot, = ax5.plot([], [], 'r.', markersize=2)
    acc_plot.set_xdata(time_c)
    acc_plot.set_ydata(acc)

    plt.show()
    return


if __name__ == '__main__':
    main(sys.argv)
