import pickle
import os
from datetime import datetime

def save_ppo_results(clip, gamma, lam, entcoeff, steps, xarr, yarr):
    log_dir = "results/"
    os.makedirs(log_dir, exist_ok=True)

    sim = {
        'params' : {
            'clip' : clip,
            'gamma' : gamma,
            'lam' : lam,
            'entcoeff' : entcoeff,
            'steps' : steps,
        },
        'xarr' : xarr,
        'yarr' : yarr
    }
    with open('results/'+timestamp_name('ppo', 'dat'), 'wb') as f:
        pickle.dump(sim, f)

def timestamp_name(msg, extension):
    now = datetime.now()
    return '{msg}_{date}.{ext}'.format(
        msg = msg, date = str(now.strftime("%Y%m%dT%H_%M_%S")), ext = extension)
