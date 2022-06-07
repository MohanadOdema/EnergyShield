import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

data_path = 'results/training/'
agents = ['off_off', 'on_off', 'on_on']
colors = ['tab:blue', 'tab:orange', 'tab:green']
plt.figure(dpi=150, figsize=(10,4))
for i in range(len(agents)):
    df = pd.read_csv(data_path + agents[i] + '_plots/run-.-tag-train_reward1.csv', delimiter=',')
    step_list   = df.Step.tolist()
    reward_list = df.Value.tolist()
    smoothed_reward_list = df.Value.ewm(alpha=0.001).mean().tolist()
    std_list = df.Value.rolling(10).std().tolist()
    plt.plot(step_list, smoothed_reward_list, color=colors[i])
    plt.fill_between(step_list, smoothed_reward_list,  [sum(x) for x in zip(smoothed_reward_list, std_list)], color=colors[i], alpha='0.25')
    plt.fill_between(step_list, smoothed_reward_list, [sum(x) for x in zip(smoothed_reward_list, [x * -1 for x in std_list])], color=colors[i], alpha='0.25')
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend( ['Obstacle OFF + Filter OFF', 'Obstacle ON  + Filter OFF', 'Obstacle ON  + Filter ON'])
plt.xlim([0,6000])
plt.ylim([0,10000])
plt.savefig('results/figs/plots/reward.png')
plt.figure(dpi=150, figsize=(10, 4))
for i in range(len(agents)):
    df = pd.read_csv(data_path + agents[i] + '_plots/run-.-tag-train_curb_hit.csv', delimiter=',')
    step_list   = df.Step.tolist()
    df.Value = df.Value * 100
    reward_list = df.Value.tolist()
    smoothed_reward_list = df.Value.ewm(alpha=0.001).mean().tolist()
    std_list = df.Value.rolling(10).std().tolist()
    plt.plot(step_list, smoothed_reward_list, color=colors[i])
    plt.fill_between(step_list, smoothed_reward_list,  [sum(x) for x in zip(smoothed_reward_list, std_list)], color=colors[i], alpha='0.25')
    plt.fill_between(step_list, smoothed_reward_list, [sum(x) for x in zip(smoothed_reward_list, [x * -1 for x in std_list])], color=colors[i], alpha='0.25')
plt.legend( ['Obstacle OFF + Filter OFF', 'Obstacle ON  + Filter OFF', 'Obstacle ON  + Filter ON'])
plt.xlabel("Episodes")
plt.ylabel("Curb Hitting Rate (%)")
plt.xlim([0,6000])
plt.ylim([0,100])
plt.savefig('results/figs/plots/curb_hitting.png')
plt.figure(dpi=150, figsize=(10, 4))
for i in range(1, len(agents)):
    df = pd.read_csv(data_path + agents[i] + '_plots/run-.-tag-train_obstacle_hit.csv', delimiter=',')
    step_list   = df.Step.tolist()
    df.Value = df.Value * 100
    reward_list = df.Value.tolist()
    smoothed_reward_list = df.Value.ewm(alpha=0.001).mean().tolist()
    std_list = df.Value.rolling(10).std().tolist()
    plt.plot(step_list, smoothed_reward_list, color=colors[i])
    plt.fill_between(step_list, smoothed_reward_list,  [sum(x) for x in zip(smoothed_reward_list, std_list)], color=colors[i], alpha='0.25')
    plt.fill_between(step_list, smoothed_reward_list, [sum(x) for x in zip(smoothed_reward_list, [x * -1 for x in std_list])], color=colors[i], alpha='0.25')
plt.legend( ['Obstacle ON  + Filter OFF', 'Obstacle ON  + Filter ON'])
plt.xlabel("Episodes")
plt.ylabel("Obstacle Hitting Rate (%)")
plt.xlim([0,6000])
plt.ylim([0,80])
plt.savefig('results/figs/plots/obstacle_hitting.png')
plt.figure(dpi=150 ,figsize=(10, 4))
for i in range(1,len(agents)):
    df = pd.read_csv(data_path + agents[i] + '_plots/run-.-tag-train_min_distance_to_obstacle.csv', delimiter=',')
    step_list   = df.Step.tolist()
    reward_list = df.Value.tolist()
    smoothed_reward_list = df.Value.ewm(alpha=0.001).mean().tolist()
    std_list = df.Value.rolling(10).std().tolist()
    plt.plot(step_list, smoothed_reward_list, color=colors[i])
    plt.fill_between(step_list, smoothed_reward_list,  [sum(x) for x in zip(smoothed_reward_list, std_list)], color=colors[i], alpha='0.25')
    plt.fill_between(step_list, smoothed_reward_list, [sum(x) for x in zip(smoothed_reward_list, [x * -1 for x in std_list])], color=colors[i], alpha='0.25')
plt.legend( ['Obstacle ON + Filter OFF', 'Obstacle ON + Filter ON'])
plt.xlabel("Episodes")
plt.ylabel("Minimum Distance to Obstacles")
plt.xlim([0,6000])
plt.ylim([0,7])
plt.savefig('results/figs/plots/min_dist.png')
