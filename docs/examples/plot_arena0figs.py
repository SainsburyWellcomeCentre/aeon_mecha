

#%% Setup
import sys
from os.path import expanduser
sys.path.append(expanduser('~/repos/aeon_mecha_de'))


import numpy as np
import pandas as pd
import aeon.analyze.patches as patches
import os 
import aeon.preprocess.api as api
import matplotlib.pyplot as plt
import aeon.util.helpers as helpers
import aeon.util.plotting as plotting

# As long as you have cloned the aeon_mecha_de folder into 
# repos in your home filter

# This path is only true if you are on pop.erlichlab.org
dataroot = '/var/www/html/delab/data/arena0.1/socialexperiment0/'
figpath = '/var/www/html/delab/figures/'

#%% Load session data.

sessdf = api.sessiondata(dataroot)
sessdf = api.sessionduration(sessdf)                                     # compute session duration
sessdf = sessdf[~sessdf.id.str.contains('test')]
sessdf = sessdf[~sessdf.id.str.contains('jeff')]
sessdf = sessdf[~sessdf.id.str.contains('OAA')]
sessdf = sessdf[~sessdf.id.str.contains('rew')]
sessdf = sessdf[~sessdf.id.str.contains('Animal')]

sessdf.reset_index(inplace=True, drop=True)

df = sessdf.copy()
helpers.merge(df)
helpers.merge(df,first=[15])
helpers.merge(df,first=[32,35], merge_id=True)
helpers.markSessionEnded(df)
session_list = df.itertuples()
print('Data loaded and merged.')
#%% save all the patch figures.

for session in session_list:
    try:
        meta = {'session':session}
        filename = plotting.plotFileName(os.path.join(figpath,'patch'), 'patch', meta)
        if not os.path.exists(filename):
            data = helpers.getWheelData(dataroot, session.start, session.end)
            data['meta'] = meta
            data['filename'] = filename
            data['total_dur'] = 60 # show 70 minutes around change.
            fig = plotting.plotWheelData(**data)
            print(f'{filename} saved.')
        else:
            print(f'{filename} exists. Skipping.')
    except IndexError as e:
        print(f'Failed to save {filename}. {e}')
        
# %%
