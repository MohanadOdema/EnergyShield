import pandas as pd
import numpy as np
import os
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

dir_path = os.getcwd()
for exp in range (1,3):
    output_path = dir_path + '\\results\exp' + str(exp) + '\\' + str(exp) + '_results.csv'
    reward_mean, reward_std, min_dst_mean, min_dst_std, track_comp_perc, track_comp_count, curb_hit, obstacle_hit = [],[],[],[],[],[],[],[]
    df = []
    valid_df = []
    valid_indeces = []
    for config in range (1,7):
        input_path = dir_path + '\\results\exp'+ str(exp) +'\\' + str(config) +'\\'+ 'valid_data.csv'
        if os.path.exists(input_path):
            df.append(pd.read_csv(input_path, delimiter=','))
            valid_df.append(pd.read_csv(input_path, delimiter=','))
    for i in range(len(df[0])):
        valid = True
        for config in range (0,6):
            if df[config]['curb_hit'][i] == False and df[config]['obstacle_hit'][i] == False  and df[config]['track_percentage'][i] < 0.85:
                valid = False
                break
            if df[config]['curb_hit'][i] == True:
                valid = False
                break
        if valid == False:
            for config in range (0,6):
                valid_df[config] = valid_df[config].drop([i])
        else:
            valid_indeces.append(i)
    length = 200
    hist_path = dir_path + '\\results\\figs\\hists\\'
    fig1 = plt.figure(1,figsize=(10, 3))
    fig2 = plt.figure(2,figsize=(10, 3))
    for config in range(0, 6):
        reward_mean.append(int(valid_df[config]['reward'][:length].mean()))
        reward_std.append(int(valid_df[config]['reward'][:length].std(ddof=0)))
        min_dst_mean.append(valid_df[config]['dist_to_obstacle'][:length].mean())
        min_dst_std.append(valid_df[config]['dist_to_obstacle'][:length].std())
        track_comp_perc.append(valid_df[config]['track_percentage'][:length].mean()*100)
        track_comp_count.append(np.sum(valid_df[config]['track_percentage'][:length]>0.99)*100/length)
        curb_hit.append(valid_df[config]['curb_hit'][:length].sum()*100/length)
        obstacle_hit.append(valid_df[config]['obstacle_hit'][:length].sum()*100/length)
        if config % 2 == 0:
            plt.figure(1)
        else:
            plt.figure(2)
        np_array = valid_df[config]['dist_to_obstacle'][:length].to_numpy()
        bins, edges = np.histogram(np_array, 10)
        left, right = edges[:-1], edges[1:]
        X = np.array([left, right]).T.flatten()
        X = np.insert(X, 0, X[0], axis=0)
        X = np.append(X, [X[-1]], axis=0)
        Y = np.array([bins, bins]).T.flatten()
        Y = np.insert(Y, 0, 0, axis=0)
        Y = np.append(Y, [0], axis=0)
        plt.plot(X, Y)
        plt.xlim([1, 5.5])
        plt.ylim([0, 85])
        ax = plt.gca()
        ax.axvline(x=4, color='tab:brown', ls='--',label='_nolegend_')
        ax.axvline(x=2.3, color='r', ls='-.',label='_nolegend_')
        if config == 5:
            plt.legend(['Config 2', 'Config 4', 'Config 6'])
            plt.savefig(hist_path + str(exp) + '_ON.png')
            fig = ax.get_figure()
            fig.clf()
        if config == 4:
            plt.legend(['Config 1', 'Config 3', 'Config 5'])
            plt.savefig(hist_path + str(exp) + '_OFF.png')
            fig = ax.get_figure()
            fig.clf()



        #ax = None
        #ax = valid_df[config]['dist_to_obstacle'][:length].plot.hist(bins=10)
        #ax.set_xlabel("")
        #ax.set_ylabel("")
        #ax.set_xlim([0, 6])
        #ax.set_ylim([0, 50])
        #fig = ax.get_figure()
        #fig.savefig(hist_path + str(exp) + '_' + str(config+1) + '.png')
        #fig.clf()
    #plt.show()
    output_df = pd.DataFrame({'Configuration': ['1', '2', '3', '4', '5', '6'], 'reward_mean': pd.Series(reward_mean), 'reward_std': pd.Series(reward_std),
                              'min_dst_mean': pd.Series(min_dst_mean), 'min_dst_std': pd.Series(min_dst_std)
                              , 'track_comp_perc': pd.Series(track_comp_perc), 'track_comp_count': pd.Series(track_comp_count)
                              , 'curb_hit': pd.Series(curb_hit), 'obstacle_hit': pd.Series(obstacle_hit)})
    output_df.to_csv(output_path, index=False)


#Experiment 3B
exp = 3
output_path = dir_path + '\\results\exp' + str(exp) + '\\' + str(exp) + '_results.csv'
reward_mean, reward_std, min_dst_mean, min_dst_std, track_comp_perc, track_comp_count, curb_hit, obstacle_hit = [],[],[],[],[],[],[],[]
df = []
valid_df = []
valid_indeces = []
for config in range (1,7):
    input_path = dir_path + '\\results\exp'+ str(exp) +'\\' + str(config) +'\\'+ 'valid_data.csv'
    if os.path.exists(input_path):
        df.append(pd.read_csv(input_path, delimiter=','))
        valid_df.append(pd.read_csv(input_path, delimiter=','))
for i in range(len(df[0])):
    valid = True
    for config in range (0,6):
        if df[config]['curb_hit'][i] == False and df[config]['obstacle_hit'][i] == False  and df[config]['track_percentage'][i] < 0.85:
            valid = False
            break
        if df[config]['curb_hit'][i] == True:
            valid = False
            break
    if valid == False:
        for config in range (0,6):
            valid_df[config] = valid_df[config].drop([i])
    else:
        valid_indeces.append(i)
length = 200
hist_path = dir_path + '\\results\\figs\\hists\\'
fig1 = plt.figure(1,figsize=(10, 3))
for config in range(0, 6):
    reward_mean.append(int(valid_df[config]['reward'][:length].mean()))
    reward_std.append(int(valid_df[config]['reward'][:length].std(ddof=0)))
    min_dst_mean.append(valid_df[config]['dist_to_obstacle'][:length].mean())
    min_dst_std.append(valid_df[config]['dist_to_obstacle'][:length].std())
    track_comp_perc.append(valid_df[config]['track_percentage'][:length].mean()*100)
    track_comp_count.append(np.sum(valid_df[config]['track_percentage'][:length]>0.99)*100/length)
    curb_hit.append(valid_df[config]['curb_hit'][:length].sum()*100/length)
    obstacle_hit.append(valid_df[config]['obstacle_hit'][:length].sum()*100/length)
    #if config % 2 == 0:
    #    plt.figure(1)
    #else:
    #    plt.figure(2)
    if config == 3 or config == 5:
        np_array = valid_df[config]['dist_to_obstacle'][:length].to_numpy()
        bins, edges = np.histogram(np_array, 10)
        left, right = edges[:-1], edges[1:]
        X = np.array([left, right]).T.flatten()
        X = np.insert(X, 0, X[0], axis=0)
        X = np.append(X, [X[-1]], axis=0)
        Y = np.array([bins, bins]).T.flatten()
        Y = np.insert(Y, 0, 0, axis=0)
        Y = np.append(Y, [0], axis=0)
        plt.plot(X, Y)
        plt.xlim([1, 5.5])
        plt.ylim([0, 85])
        ax = plt.gca()
        ax.axvline(x=4, color='tab:brown', ls='--', label='_nolegend_')
        ax.axvline(x=2.3, color='r', ls='-.', label='_nolegend_')
        if config == 5:
            plt.legend(['Config 4', 'Config 6'])
            plt.savefig(hist_path + str(exp) + '_ON.png')
            fig = ax.get_figure()
            fig.clf()
output_df = pd.DataFrame({'Configuration': ['1', '2', '3', '4', '5', '6'], 'reward_mean': pd.Series(reward_mean),
                          'reward_std': pd.Series(reward_std),
                          'min_dst_mean': pd.Series(min_dst_mean), 'min_dst_std': pd.Series(min_dst_std)
                             , 'track_comp_perc': pd.Series(track_comp_perc),
                          'track_comp_count': pd.Series(track_comp_count)
                             , 'curb_hit': pd.Series(curb_hit), 'obstacle_hit': pd.Series(obstacle_hit)})
output_df.to_csv(output_path, index=False)