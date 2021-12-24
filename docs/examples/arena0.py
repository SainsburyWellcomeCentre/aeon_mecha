

#%% Setup
import sys
import os
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
#from multiprocessing import Pool

# As long as you have cloned the aeon_mecha_de folder into 
# repos in your home filter

# This path is only true if you are on pop.erlichlab.org
env = os.environ
dataroot = env.get('aeon_dataroot', '/var/www/html/delab/data/arena0.1/socialexperiment0/')
figpath = env.get('aeon_figpath','/var/www/html/delab/figures/')
exportpath = env.get('aeon_exportpath','/var/www/html/delab/data/arena0.1/exported/')
export_format = env.get('aeon_dataformat','parquet')
fig_format = env.get('aeon_figformat','png')


#%% 
def makeWheelPlots(df):
    try:
        fileformat = sys.argv[2]
    except IndexError:
        fileformat = 'png'

    print(f'Saving files as {fileformat} in {figpath}')

    session_list = df.itertuples()
    print('Data loaded and merged.')
    #%% save all the patch figures.

    for session in session_list:
        try:
            meta = {'session':session}
            filename = plotting.plotFileName(os.path.join(figpath,'patch_flip_full',fileformat), 'patch', meta, type=fileformat)
            if not os.path.exists(filename):
                data = helpers.getWheelData(dataroot, session.start, session.end)
                data['meta'] = meta
                data['filename'] = filename
                data['change_in_red'] = True
                # data['total_dur'] = 60 # show 70 minutes around change.
                plotting.plotWheelData(**data)

                print(f'{filename} saved.')
            else:
                print(f'{filename} exists. Skipping.')
        except IndexError as e:
            print(f'Failed to save {filename}. {e}')

def exportDataToParquet(limit=1e6):
    done = 0
    
    for session in df.itertuples(): # This is easily parallelized :shrug:
        print(f'Exporting {helpers.getSessionID(session)}')
        helpers.exportWheelData(dataroot, session, 
        datadir = exportpath, format = 'parquet', force=False)
        if done >= limit:
            return
        else:
            done += 1


def exportDataToCSV(limit=1e6):
    done = 0
    for session in df.itertuples(): # This is easily parallelized :shrug:
        print(f'Exporting {helpers.getSessionID(session)}')
        helpers.exportWheelData(dataroot, session, 
        datadir = exportpath, format = 'csv', force=False)
        if done >= limit:
            return
        else:
            done += 1
    
# %%
#df = helpers.loadSessions(dataroot)
#exportDataToParquet(1)

if __name__ == "__main__":
# if False:
    funclist = ['makeWheelPlots', 'exportDataToParquet']
    if len(sys.argv) == 1:
        print(f"""
        This function is a wrapper for some common arena0 activities.
        
        The way to call it is:

        python arena0 func arg1 arg2

        Available functions are:
        {funclist}
        
        df, a dataframe of the sessions is availabe by default.

        Some settings should be set using environment variables:
        ```
        dataroot = env.get('aeon_dataroot', '/var/www/html/delab/data/arena0.1/socialexperiment0/')
        figpath = env.get('aeon_figpath','/var/www/html/delab/figures/')
        exportpath = env.get('aeon_exportpath','/var/www/html/delab/data/arena0.1/exported/')
        export_format = env.get('aeon_dataformat','parquet')
        fig_format = env.get('aeon_figformat','png')
        ```
        """)
        sys.exit()
        
    func = sys.argv[1]
    df = helpers.loadSessions(dataroot)
    eval(func)(*sys.argv[2:])
    

#%%
