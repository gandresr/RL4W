import pickle
from datetime import datetime

def save_ppo_results(clip, gamma, lam, entcoeff, steps, xarr, yarr):
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
    now = datetime.now()
    pickle.dump(sim, 'results/' + timestamp_name('ppo', '.dat'))

def timestamp_name(msg, extension):
    return '{msg}_{date}.{ext}'.format(
        msg = msg, date = str(now.strftime("%Y%m%dT%H_%M_%S"), ext = extension))
