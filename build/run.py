import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import subprocess
import sys


def main(argv):
    process = subprocess.Popen("reset", shell=False)
    process.wait()
    process = subprocess.Popen("./main", shell=False)
    process.wait()

    if process.returncode == 1:
        return

    df = pd.read_csv('states.txt', header=None)
    states = df.to_numpy()

    time_s = states[:, 0]
    x = states[:, 1]
    y = states[:, 2]
    h = states[:, 3]
    v = states[:, 4]
    lat_acc = states[:, 5]
    lat_error = states[:, 6]

    df = pd.read_csv('controls.txt', header=None)
    controls = df.to_numpy()

    time_c = controls[:, 0]
    delta = controls[:, 1]
    acc = controls[:, 2]
    d_delta_f = controls[:, 3]
    d_acc = controls[:, 4]

    df = pd.read_csv('ref_traj.txt', header=None)
    ref_traj = df.to_numpy()

    ref_t = ref_traj[:, 0]
    ref_x = ref_traj[:, 1]
    ref_y = ref_traj[:, 2]
    ref_h = ref_traj[:, 3]
    ref_v = ref_traj[:, 4]

    df = pd.read_csv('ref_traj_left_bounds.txt', header=None)
    ref_traj_left_bounds = df.to_numpy()

    ref_left_x = ref_traj_left_bounds[:, 1]
    ref_left_y = ref_traj_left_bounds[:, 2]

    df = pd.read_csv('ref_traj_right_bounds.txt', header=None)
    ref_traj_right_bounds = df.to_numpy()

    ref_right_x = ref_traj_right_bounds[:, 1]
    ref_right_y = ref_traj_right_bounds[:, 2]

    df = pd.read_csv('optimization_parameters.txt', header=None)
    parameters = df.to_numpy()
    dt = parameters[0]
    horizon = int(parameters[1])
    delta_f_min = parameters[2]
    delta_f_max = parameters[4]
    acc_min = parameters[3]
    acc_max = parameters[5]
    lat_acc_limit = parameters[6]
    lat_dist_limit = parameters[7]
    d_delta_f_min = parameters[8]
    d_delta_f_max = parameters[10]
    d_acc_min = parameters[9]
    d_acc_max = parameters[11]

    figure, ax, = plt.subplots(figsize=(25, 12))
    gs = gridspec.GridSpec(2, 5)
    ax.set_title('Extended LQR Motion Planner', pad=20)
    ax.title.set_fontsize(18)
    ax.axis('off')

    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=0.5, hspace=0.5)

    ax1 = figure.add_subplot(gs[0, 0:3])
    ax1.title.set_text('vehicle trajectory')
    ax1.title.set_fontsize(15)
    ax1.grid()
    ax1.set_aspect('equal')
    ax1.set_xlabel('x (m)', fontsize=12)
    ax1.set_ylabel('y (m)', fontsize=12)
    ax1.set_xlim([-10.0, 120.0])
    ax1.set_ylim([-10.0, 40.0])

    traj_plot, = ax1.plot([], [], 'r.', markersize=5)
    traj_plot.set_xdata(x)
    traj_plot.set_ydata(y)

    ref_traj_plot, = ax1.plot([], [], 'g.', markersize=3)
    ref_traj_plot.set_xdata(ref_x)
    ref_traj_plot.set_ydata(ref_y)

    ref_traj_left_bound_plot, = ax1.plot([], [], 'g.', markersize=1)
    ref_traj_left_bound_plot.set_xdata(ref_left_x)
    ref_traj_left_bound_plot.set_ydata(ref_left_y)

    ref_traj_right_bound_plot, = ax1.plot([], [], 'g.', markersize=1)
    ref_traj_right_bound_plot.set_xdata(ref_right_x)
    ref_traj_right_bound_plot.set_ydata(ref_right_y)

    start_pos_plot, = ax1.plot([], [], 'mx', markersize=10)
    initial_pos_x = []
    initial_pos_y = []
    N = len(x)
    num_runs = int(math.ceil(N/horizon))

    for i in range(num_runs):
        initial_pos_x.append(x[i*horizon])
        initial_pos_y.append(y[i*horizon])

    start_pos_plot.set_xdata(initial_pos_x)
    start_pos_plot.set_ydata(initial_pos_y)

    end_pos_plot, = ax1.plot([], [], 'y*', markersize=10)
    goal_pos_x = []
    goal_pos_y = []

    for i in range(1, num_runs):
        goal_pos_x.append(ref_x[i*horizon+1])
        goal_pos_y.append(ref_y[i*horizon+1])
    goal_pos_x.append(ref_x[-1])
    goal_pos_y.append(ref_y[-1])

    end_pos_plot.set_xdata(goal_pos_x)
    end_pos_plot.set_ydata(goal_pos_y)

    ax1.legend(['planned trajectory', 'reference trajectory',
                'reference trajectory left bound', 'reference trajectory right bound',
                'initial position', 'goal position'], loc='upper left',
               fontsize=10, bbox_to_anchor=(-0.4, 1.0))

    ax2 = figure.add_subplot(gs[1, 0:1])
    ax2.title.set_text('vehicle heading')
    ax2.title.set_fontsize(15)
    ax2.grid()
    ax2.set_xlabel('time (s)', fontsize=12)
    ax2.set_ylabel('heading angle (rad)', fontsize=12)
    ax2.set_xlim([0, np.max(time_s)])
    ax2.set_ylim([min(ref_h)-math.pi, max(ref_h)+math.pi])

    heading_plot, = ax2.plot([], [], 'r.', markersize=4)
    heading_plot.set_xdata(time_s)
    heading_plot.set_ydata(h)

    ref_heading_plot, = ax2.plot([], [], 'g.', markersize=4)
    ref_heading_plot.set_xdata(ref_t)
    ref_heading_plot.set_ydata(ref_h)

    ax2.legend(['planned heading', 'reference heading'],
               loc='upper left', fontsize=10, bbox_to_anchor=(0.0, 1.3))

    ax3 = figure.add_subplot(gs[1, 1:2])
    ax3.title.set_text('vehicle velocity')
    ax3.title.set_fontsize(15)
    ax3.grid()
    ax3.set_xlabel('time (s)', fontsize=12)
    ax3.set_ylabel('velocity (m/s)', fontsize=12)
    ax3.set_xlim([0, np.max(time_s)])
    ax3.set_ylim([-1, max(v)+1.0])

    speed_plot, = ax3.plot([], [], 'r.', markersize=4)
    speed_plot.set_xdata(time_s)
    speed_plot.set_ydata(v)

    ref_speed_plot, = ax3.plot([], [], 'g.', markersize=4)
    ref_speed_plot.set_xdata(ref_t)
    ref_speed_plot.set_ydata(ref_v)

    ax3.legend(['planned velocity', 'reference velocity'],
               loc='upper left', fontsize=10, bbox_to_anchor=(0.0, 1.3))

    ax4 = figure.add_subplot(gs[1, 2:3])
    ax4.title.set_text('vehicle lateral acceleration')
    ax4.title.set_fontsize(15)
    ax4.grid()
    ax4.set_xlabel('time (s)', fontsize=12)
    ax4.set_ylabel('lat acc (m/s^2)', fontsize=12)
    ax4.set_xlim([0, np.max(time_s)])
    ax4.set_ylim([min(-lat_acc_limit, min(lat_acc))-0.5,
                  max(lat_acc_limit, max(lat_acc))+0.5])

    lat_acc_plot, = ax4.plot([], [], 'r.', markersize=4)
    lat_acc_plot.set_xdata(time_s)
    lat_acc_plot.set_ydata(lat_acc)

    min_lat_acc_plot, = ax4.plot([], [], 'g.-', markersize=3)
    min_lat_acc_plot.set_xdata(time_s)
    min_lat_acc_plot.set_ydata(-lat_acc_limit*np.ones([1, len(time_s)]))

    max_lat_acc_plot, = ax4.plot([], [], 'g.-', markersize=3)
    max_lat_acc_plot.set_xdata(time_s)
    max_lat_acc_plot.set_ydata(lat_acc_limit*np.ones([1, len(time_s)]))

    ax4.legend(['planned lateral acceleration', 'bounds'],
               loc='upper left', fontsize=10, bbox_to_anchor=(0.0, 1.3))

    # ax5 = figure.add_subplot(gs[1, 3:4])
    # ax5.title.set_text('vehicle lateral error')
    # ax5.title.set_fontsize(15)
    # ax5.grid()
    # ax5.set_xlabel('time (s)', fontsize=12)
    # ax5.set_ylabel('lat acc (m/s^2)', fontsize=12)
    # ax5.set_xlim([0, np.max(time_s)])
    # ax5.set_ylim([min(-lat_dist_limit, min(lat_error))-0.5,
    #               max(lat_dist_limit, max(lat_error))+0.5])

    # lat_error_plot, = ax5.plot([], [], 'r.', markersize=4)
    # lat_error_plot.set_xdata(time_s)
    # lat_error_plot.set_ydata(lat_error)

    # min_lat_error_plot, = ax5.plot([], [], 'g.-', markersize=3)
    # min_lat_error_plot.set_xdata(time_s)
    # min_lat_error_plot.set_ydata(-lat_dist_limit*np.ones([1, len(time_s)]))

    # max_lat_error_plot, = ax5.plot([], [], 'g.-', markersize=3)
    # max_lat_error_plot.set_xdata(time_s)
    # max_lat_error_plot.set_ydata(lat_dist_limit*np.ones([1, len(time_s)]))

    # ax5.legend(['planned lateral error', 'bounds'],
    #            loc='upper left', fontsize=10, bbox_to_anchor=(0.0, 1.3))

    ax6 = figure.add_subplot(gs[0, 3:4])
    ax6.title.set_text('control - front wheel angle')
    ax6.title.set_fontsize(15)
    ax6.grid()
    ax6.set_xlabel('time (s)', fontsize=12)
    ax6.set_ylabel('front wheel angle (rad)', fontsize=12)
    ax6.set_xlim([0, np.max(time_c)+dt])
    ax6.set_ylim([delta_f_min-0.1, delta_f_max+0.1])

    steer_plot, = ax6.plot([], [], 'r.', markersize=3)
    steer_plot.set_xdata(time_c)
    steer_plot.set_ydata(delta)

    min_steer_plot, = ax6.plot([], [], 'g.-', markersize=3)
    min_steer_plot.set_xdata(time_c)
    min_steer_plot.set_ydata(delta_f_min*np.ones([1, len(time_c)]))

    max_steer_plot, = ax6.plot([], [], 'g.-', markersize=3)
    max_steer_plot.set_xdata(time_c)
    max_steer_plot.set_ydata(delta_f_max*np.ones([1, len(time_c)]))

    ax6.legend(['planned front wheel angle', 'bounds'],
               loc='upper left', fontsize=10, bbox_to_anchor=(1.0, 1.0))

    ax8 = figure.add_subplot(gs[0, 4:5])
    ax8.title.set_text('control - delta front wheel')
    ax8.title.set_fontsize(15)
    ax8.grid()
    ax8.set_xlabel('time (s)', fontsize=12)
    ax8.set_ylabel('delta front wheel (rad)', fontsize=12)
    ax8.set_xlim([0, np.max(time_c)+dt])
    ax8.set_ylim([d_delta_f_min-0.015, d_delta_f_max+0.015])
    d_steer_plot, = ax8.plot([], [], 'r.', markersize=3)
    d_steer_plot.set_xdata(time_c)
    d_steer_plot.set_ydata(d_delta_f)

    min_d_steer_plot, = ax8.plot([], [], 'g.-', markersize=3)
    min_d_steer_plot.set_xdata(time_c)
    min_d_steer_plot.set_ydata(d_delta_f_min*np.ones([1, len(time_c)]))

    max_d_steer_plot, = ax8.plot([], [], 'g.-', markersize=3)
    max_d_steer_plot.set_xdata(time_c)
    max_d_steer_plot.set_ydata(d_delta_f_max*np.ones([1, len(time_c)]))

    ax8.legend(['planned delta front wheel', 'bounds'],
               loc='upper left', fontsize=10, bbox_to_anchor=(1.0, 1.0))

    ax7 = figure.add_subplot(gs[1, 3:4])
    ax7.title.set_text('control - acc')
    ax7.title.set_fontsize(15)
    ax7.grid()
    ax7.set_xlabel('time (s)', fontsize=12)
    ax7.set_ylabel('acceleration (m/s^2)', fontsize=12)
    ax7.set_xlim([0, np.max(time_c)+dt])
    ax7.set_ylim([acc_min-0.5, acc_max+0.5])

    acc_plot, = ax7.plot([], [], 'r.', markersize=3)
    acc_plot.set_xdata(time_c)
    acc_plot.set_ydata(acc)

    min_acc_plot, = ax7.plot([], [], 'g.-', markersize=3)
    min_acc_plot.set_xdata(time_c)
    min_acc_plot.set_ydata(acc_min*np.ones([1, len(time_c)]))

    max_acc_plot, = ax7.plot([], [], 'g.-', markersize=3)
    max_acc_plot.set_xdata(time_c)
    max_acc_plot.set_ydata(acc_max*np.ones([1, len(time_c)]))

    ax7.legend(['planned acc', 'bounds'],
               loc='upper left', fontsize=10, bbox_to_anchor=(1.0, 1.0))

    ax9 = figure.add_subplot(gs[1, 4:5])
    ax9.title.set_text('control - delta acc')
    ax9.title.set_fontsize(15)
    ax9.grid()
    ax9.set_xlabel('time (s)', fontsize=12)
    ax9.set_ylabel('delta acc (m/s^2)', fontsize=12)
    ax9.set_xlim([0, np.max(time_c)+dt])
    ax9.set_ylim([d_acc_min-0.2, d_acc_max+0.2])

    d_acc_plot, = ax9.plot([], [], 'r.', markersize=3)
    d_acc_plot.set_xdata(time_c)
    d_acc_plot.set_ydata(d_acc)

    d_min_acc_plot, = ax9.plot([], [], 'g.-', markersize=3)
    d_min_acc_plot.set_xdata(time_c)
    d_min_acc_plot.set_ydata(d_acc_min*np.ones([1, len(time_c)]))

    d_max_acc_plot, = ax9.plot([], [], 'g.-', markersize=3)
    d_max_acc_plot.set_xdata(time_c)
    d_max_acc_plot.set_ydata(d_acc_max*np.ones([1, len(time_c)]))

    ax9.legend(['planned delta acceleration', 'bounds'],
               loc='upper left', fontsize=10, bbox_to_anchor=(1.0, 1.0))

    plt.show()
    return


if __name__ == '__main__':
    main(sys.argv)
