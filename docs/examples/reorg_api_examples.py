from pathlib import Path
from datetime import datetime, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotmap import DotMap
from aeon import api


# <s Get position data, pellet delivery data, and patch wheel data

exp01_root = Path('/ceph/aeon/test2/experiment0.1')
exp02_root = Path('/ceph/aeon/test2/experiment0.2')
start_ts1 = pd.Timestamp('2021-11-26')
end_ts1 = pd.Timestamp('2021-11-27')
start_ts2 = pd.Timestamp('2022-02-24')
end_ts2 = pd.Timestamp('2022-02-26')

exp01_data_dict = api.gen_data_dict(exp01_root)
exp02_data_dict = api.gen_data_dict(exp02_root)

# example of data_dict
exp02_data_dict = DotMap({
    'position': 'CameraTop_200',
    'pellet_triggered_patch1': 'Patch1_35',
    'pellet_delivered_patch1': 'Patch1_32',
    'pellet_triggered_patch2': 'Patch2_35',
    'pellet_delivered_patch2': 'Patch2_32',
    'wheel_encoder_patch1': 'Patch1_90',
    'wheel_encoder_patch2': 'Patch2_90',
    'weight': 'Nest_200'})

# Start with data from exp02

position_data = api.load(path=exp_root, start_ts=start_ts, end_ts=end_ts,
                         data=exp02_data_dict.position, pos=None)
patch_names = ['patch1', 'patch2']
for p in patch_names:
    pellet_data = p + ''

pellet_patch1_data = api.load(path=exp_root, start_ts=start_ts, end_ts=end_ts,
                               data=exp02_data_dict.pellet, pos='patch1')
pellet_patch2_data = data.load(path=exp_root, start_ts=start_ts, end_ts=end_ts,
                               pos=None,
                          data=DATA.position)
wheel_patch1_data = data.load(path=exp_root, start_ts=start_ts, end_ts=end_ts, pos=None,
                          data=DATA.position)
wheel_patch2_data = data.load(path=exp_root, start_ts=start_ts, end_ts=end_ts, pos=None,
                          data=DATA.position)
all_data = api.load(path=exp02_root, start_ts=start_ts, end_ts=end_ts, data='all',
                    pos=None)  # `'all'` returns dict of dataframes














# <s `chunk` functions


# /s>

root = '/ceph/aeon/test2/experiment0.1'
data = aeon.sessiondata(root)
annotations = aeon.annotations(root)

data = data[data.id.str.startswith('BAA')]  # take only proper sessions
if len(data) % 2 != 0:  # if number of sessions don't pair up
    data = data.drop(data.index[-1])  # drop last session (might be ongoing)
data = aeon.sessionduration(data)  # compute session duration

for session in data.itertuples():  # for all sessions
    print('{0} on {1}...'.format(session.id,
                                 session.Index))  # print progress report
    start = session.Index  # session start time is session index
    end = start + session.duration  # end time = start time + duration
    position = aeon.positiondata(root, start=start,
                                 end=end)  # get position data between start and end
    position = position[
        position.area < 2000]  # filter for objects of the correct size

    encoder1 = aeon.encoderdata(root, 'Patch1', start=start,
                                end=end)  # get encoder data for patch1 between start and end
    encoder2 = aeon.encoderdata(root, 'Patch2', start=start,
                                end=end)  # get encoder data for patch2 between start and end
    pellets1 = aeon.pelletdata(root, 'Patch1', start=start,
                               end=end)  # get pellet events for patch1 between start and end
    pellets2 = aeon.pelletdata(root, 'Patch2', start=start,
                               end=end)  # get pellet events for patch2 between start and end

    wheel1 = aeon.distancetravelled(
        encoder1.angle)  # compute total distance travelled on patch1 wheel
    wheel2 = aeon.distancetravelled(
        encoder2.angle)  # compute total distance travelled on patch2 wheel
    pellets1 = pellets1[
        pellets1.event == 'TriggerPellet']  # get timestamps of pellets delivered at patch1
    pellets2 = pellets2[
        pellets2.event == 'TriggerPellet']  # get timestamps of pellets delivered at patch2

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,
                                                 2)  # create a figure with subplots

    ax1.plot(position.x, position.y,
             alpha=0.4)  # plot position data as a path trajectory
    forage = position.reindex(pellets1.index,
                              method='nearest')  # get position data when a pellet is delivered at patch1
    forage.plot.scatter('x', 'y', s=1, c='red',
                        ax=ax1)  # plot mouse positions when pellets were delivered

    for trial in pellets1.itertuples():  # for each pellet delivery
        before = trial.Index - pd.to_timedelta(10,
                                               's')  # get the previous 10 seconds
        path = position.loc[
               before:trial.Index]  # get position data in the time before pellet delivery
        ax1.plot(path.x, path.y)  # plot path traces preceding pellet delivery

    ax2.hist(position.area, bins=100)  # plot histogram of tracked object size

    wheel1.plot(ax=ax3)  # plot distance travelled on patch1 wheel
    wheel1.plot(ax=ax4)  # plot distance travelled on patch2 wheel
    ax3.set_ylabel('distance (cm)')  # set axis label
    ax4.set_ylabel('distance (cm)')  # set axis label

    fig.savefig('{0}_{1}.png'.format(session.id,
                                     start.date()))  # save figure tagged with id and date
    plt.close(fig)  # close figure
