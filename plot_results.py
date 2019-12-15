import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict as ddict
from scipy.ndimage.filters import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    control_types = os.listdir('main_results')
    for control_type in control_types:
        reward_types = os.listdir('main_results' + os.sep + control_type)
        for reward_type in reward_types:
            fpath = 'main_results' + os.sep + control_type + os.sep + reward_type
            files = os.listdir(fpath)

            entcoeff = ddict(list); xdata = ddict(list); ydata = ddict(list)
            colors = ['b', 'g', 'r', 'm'] #Add more if needed

            for file in files:
                if file.endswith('.dat'):
                    data = pickle.load(open(fpath + os.sep + file, "rb"))
                    clip = data['params']['clip']
                    entcoeff[clip].append(data['params']['entcoeff'])
                    xdata[clip].append(data['xarr'])
                    ydata[clip].append(data['yarr'])
            plt.figure(figsize = (14,5))
            for i, c in enumerate(entcoeff.keys()):
                plt.subplot(1, len(entcoeff.keys()), i+1)
                plt.xlabel('Learning steps')
                plt.ylabel('Reward')
                for j in range(len(xdata[c])):
                    y_smooth = gaussian_filter1d(ydata[c][j], sigma=2)
                    plt.plot(xdata[c][j], ydata[c][j], colors[j] + '.')
                    plt.plot(xdata[c][j], y_smooth, colors[j] + '-', \
                        label = 'ent = %.4f' % entcoeff[c][j])
                    plt.title('$\epsilon$ = %.2f' % c)
                plt.legend()
            plt.tight_layout()
            os.makedirs('paper_figures', exist_ok=True)
            plt.savefig('paper_figures/' + 'PPO_' + control_type + '_' + reward_type + '.png')
            plt.clf()