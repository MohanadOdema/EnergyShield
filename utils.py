import types
import os

import cv2
import numpy as np
import scipy.signal
import tensorflow
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

class VideoRecorder():
    def __init__(self, filename, frame_size, fps=30):
        self.video_writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*"MPEG"), int(fps),
            (frame_size[1], frame_size[0]))

    def add_frame(self, frame):
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def release(self):
        self.video_writer.release()

    def __del__(self):
        self.release()

def build_mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def create_counter_variable(name):
    counter = types.SimpleNamespace()
    counter.var = tf.Variable(0, name=name, trainable=False)
    counter.inc_op = tf.assign(counter.var, counter.var + 1)
    return counter

def create_mean_metrics_from_dict(metrics):
    # Set up summaries for each metric
    update_metrics_ops = []
    summaries = []
    for name, (value, update_op) in metrics.items():
        summaries.append(tf.summary.scalar(name, value))
        update_metrics_ops.append(update_op)
    return tf.summary.merge(summaries), tf.group(update_metrics_ops)

def compute_gae(rewards, values, bootstrap_values, terminals, gamma, lam):
    rewards = np.array(rewards)
    values = np.array(list(values) + [bootstrap_values])
    terminals = np.array(terminals)
    deltas = rewards + (1.0 - terminals) * gamma * values[1:] - values[:-1]
    return scipy.signal.lfilter([1], [1, -gamma * lam], deltas[::-1], axis=0)[::-1]

def count_flags(miss_flag):
    previous = 0
    counter = []
    for flag in miss_flag:
        current = previous + flag
        counter.append(current)
        previous = current
    assert len(counter) == len(miss_flag)
    return counter

# plots
def plot_trajectories(ego_x, ego_y, completed_x, completed_y, obstacle_x, obstacle_y, plot_dir, episode_idx):
    plt.figure()
    plt.plot(ego_x, ego_y, 'g--')
    if len(obstacle_x) > 0:
        plt.plot(obstacle_x, obstacle_y, 'bo')
    plt.plot(completed_x, completed_y, 'r-.')
    # plt.xlim([180, 430])
    # plt.ylim([-410,-120])
    plt.savefig(plot_dir + '/train_' + str(episode_idx) + '_xy.png')
    plt.close()

def plot_energy_stats(exp_latency, exp_energy, exp_tu, miss_flag, plot_dir, episode_idx, params):
    accum_miss = count_flags(miss_flag)
    index_list = np.arange(len(accum_miss))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(index_list, exp_energy, 'g-')
    ax2.plot(index_list, accum_miss, 'b-')
    ax1.set_xlabel("Deadline %d index" % params['deadline'])
    ax1.set_ylabel("Energy")
    ax2.set_ylabel("Missed deadlines")
    plt.savefig(plot_dir + '/train_' + str(episode_idx) + '_ergy.png')
    plt.close()

# create directories
class dir_manager():
    def __init__(self, model_dir="./", subdirs=""):
        self.ep_counter = 0
        self.model_dir = model_dir
        self.subdirs = subdirs
        self.checkpoint_dir = "{}/checkpoints/".format(self.model_dir)
        self.log_dir        = "{}/logs/".format(self.subdirs)
        self.video_dir      = "{}/videos/".format(self.subdirs)
        self.plot_dir       = "{}/plots/".format(self.subdirs)
        self.dirs = [self.checkpoint_dir, self.log_dir, self.video_dir, self.plot_dir]
        for d in self.dirs: os.makedirs(d, exist_ok=True)

    def get_episode_idx(self):
        return self.ep_counter                # hard-coded for automated navigation

    def inc_count(self):
        self.ep_counter = self.ep_counter + 1