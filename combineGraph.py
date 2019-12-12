import os

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.ndimage.filters import gaussian_filter1d

if __name__ == "__main__":
    folder = 'results2/'
    files = ['ppo_20191208T21_38_33.dat', 'ppo_20191208T22_06_38.dat'
             ,'ppo_20191208T22_33_58.dat', 'ppo_20191208T23_00_16.dat']
    data = []
    clip = []
    entcoeff = []
    xdata = []
    ydata = []
    colors = ['b', 'g', 'r', 'm'] #Add more if needed

    for i in range(len(files)):
        data = pickle.load(open(folder + files[i], "rb"))
        clip.append(data['params']['clip'])
        entcoeff.append(data['params']['entcoeff'])
        xdata.append(data['xarr'])
        ydata.append(data['yarr'])

    legend = []
    for i in range(len(files)):
        xarr = np.array(xdata[i])
        yarr = np.array(ydata[i])
        y_smooth = gaussian_filter1d(yarr, sigma=2)
        plt.plot(xarr, yarr, colors[i] + '.')
        plt.plot(xarr, y_smooth, colors[i] + '-', label='Eps=' + str(clip[i]) + ',Ent=' + str(entcoeff[i]))

    plt.legend()
    plt.title('PPO comparison')
    plt.xlabel('timesteps during learning')
    plt.ylabel('reward')
    plt.show()
        #plt.savefig('results/figures/' + timestamp_name('ppo', 'png'))
        #plt.clf()

