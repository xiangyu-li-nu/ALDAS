import pandas as pd
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
SEED_NUM = 3
ENV = "LC"

if __name__ == '__main__':
#    file = os.path.join("ppo/ppo_s0", "progress.txt")
    algo = [ "5000_" + ENV, "4000_" + ENV, "3000_" + ENV]
    data = []
    label = ['Traffic Flow = 5000','Traffic Flow = 4000', 'Traffic Flow = 3000']
    for i in range(len(algo)):
        for seed in range(SEED_NUM):
            file = os.path.join(os.path.join(algo[i], algo[i] + "_s" + str(seed*10)), "reward.txt")
            pd_data = pd.read_table(file,sep=' ')
            pd_data.insert(len(pd_data.columns), "Unit", seed)
            pd_data.insert(len(pd_data.columns), "Condition", label[i])
            data.append(pd_data)

    data = pd.concat(data, ignore_index=True)
#    print(data)
    sns.set(style="whitegrid", font_scale=1.5)
    sns.tsplot(data=data, time="Epoch", value="AverageEpRet", condition="Condition", unit="Unit", legend=True)

    xscale = np.max(data["Epoch"]) > 5e3
    if xscale:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.ylabel("Average Cumulative Reward", fontsize=16)
    plt.xlabel("Epoch", fontsize=16)
#    plt.title("Total reward to Epoch", fontsize=20)
    plt.legend(loc='best').set_draggable(True)
    plt.tight_layout(pad=0.5)
    plt.grid

#    my_x_ticks = [0,2000,4000,6000,8000,10000]
#    my_y_ticks = [0,0.2,0.4,0.6,0.8,1.0]

#    plt.xticks(my_x_ticks)
#    plt.yticks(my_y_ticks)

    plt.show()