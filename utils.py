import pickle
import os
from datetime import datetime

def save_ppo_results(log_dir, fname, steps, clip, gamma, lam, entcoeff, time_steps, xarr, yarr, true_observations, true_actions, pi):
    os.makedirs(log_dir, exist_ok=True)

    sim = {
        'params' : {
            'clip' : clip,
            'gamma' : gamma,
            'lam' : lam,
            'entcoeff' : entcoeff,
            'steps' : steps,
        },
        'true_observations' : true_observations,
        'true_actions' : true_actions,
        'pi' : pi,
        'xarr' : xarr,
        'yarr' : yarr
    }
    with open(log_dir+fname+'.dat', 'wb') as f:
        pickle.dump(sim, f)

def timestamp_name(msg, extension):
    now = datetime.now()
    return '{msg}_{date}.{ext}'.format(
        msg = msg, date = str(now.strftime("%Y%m%dT%H_%M_%S")), ext = extension)