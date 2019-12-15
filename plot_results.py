import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from collections import defaultdict as ddict
from scipy.ndimage.filters import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    matplotlib.rcParams.update({'font.size': 17})
    control_types = os.listdir('main_results')
    for control_type in control_types:
        reward_types = os.listdir('main_results' + os.sep + control_type)
        for reward_type in reward_types:
            fpath = 'main_results' + os.sep + control_type + os.sep + reward_type
            files = os.listdir(fpath)
            entcoeff = ddict(list); xdata = ddict(list); ydata = ddict(list)
            colors = ['b', 'g', 'r', 'm'] #Add more if needed
            max_r = -1e10; min_r = 1e10
            for file in files:
                if file.endswith('.dat'):
                    data = pickle.load(open(fpath + os.sep + file, "rb"))
                    clip = data['params']['clip']
                    entcoeff[clip].append(data['params']['entcoeff'])
                    xdata[clip].append(data['xarr'])
                    ydata[clip].append(data['yarr'])
                    if max(data['yarr']) > max_r: max_r = max(data['yarr'])
                    if min(data['yarr']) < min_r: min_r = min(data['yarr'])
            plt.figure(figsize = (22,7.5))
            clips = np.sort(list(entcoeff.keys()))
            for i, c in enumerate(clips):
                ax = plt.subplot(1, len(clips), i+1)
                ax.set_ylim([min_r, max_r])
                plt.xlabel('Learning steps')
                for j in range(len(xdata[c])):
                    y_smooth = gaussian_filter1d(ydata[c][j], sigma=2)
                    ax.plot(xdata[c][j], ydata[c][j], colors[j] + '.')
                    ax.plot(xdata[c][j], y_smooth, colors[j] + '-', \
                        label = 'ent = %.4f' % entcoeff[c][j])
                    plt.title('$\epsilon$ = %.2f' % c)
                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=2)
            os.makedirs('paper_figures', exist_ok=True)
            plt.savefig('paper_figures/' + 'PPO_' + control_type + '_' + reward_type + '.png', bbox_inches='tight')
            plt.clf()