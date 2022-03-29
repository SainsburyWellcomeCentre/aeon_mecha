import datetime
import pandas as pd
import aeon.io.api as aeon
import matplotlib.pyplot as plt

root = '/ceph/aeon/test2/experiment0.1'
data = aeon.sessiondata(root)
annotations = aeon.annotations(root)

data = data[data.id.str.startswith('BAA')]                            # take only proper sessions
if len(data) % 2 != 0:                                                # if number of sessions don't pair up
    data = data.drop(data.index[-1])                                  # drop last session (might be ongoing)
data = aeon.sessionduration(data)                                     # compute session duration

for session in data.itertuples():                                     # for all sessions
    print('{0} on {1}...'.format(session.id, session.Index))          # print progress report
    start = session.Index                                             # session start time is session index
    end = start + session.duration                                    # end time = start time + duration
    position = aeon.positiondata(root, start=start, end=end)          # get position data between start and end
    position = position[position.area < 2000]                         # filter for objects of the correct size

    encoder1 = aeon.encoderdata(root, 'Patch1', start=start, end=end) # get encoder data for patch1 between start and end
    encoder2 = aeon.encoderdata(root, 'Patch2', start=start, end=end) # get encoder data for patch2 between start and end
    pellets1 = aeon.pelletdata(root, 'Patch1', start=start, end=end)  # get pellet events for patch1 between start and end
    pellets2 = aeon.pelletdata(root, 'Patch2', start=start, end=end)  # get pellet events for patch2 between start and end

    wheel1 = aeon.distancetravelled(encoder1.angle)                   # compute total distance travelled on patch1 wheel
    wheel2 = aeon.distancetravelled(encoder2.angle)                   # compute total distance travelled on patch2 wheel
    pellets1 = pellets1[pellets1.event == 'TriggerPellet']            # get timestamps of pellets delivered at patch1
    pellets2 = pellets2[pellets2.event == 'TriggerPellet']            # get timestamps of pellets delivered at patch2

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)                # create a figure with subplots

    ax1.plot(position.x, position.y, alpha=0.4)                       # plot position data as a path trajectory
    forage = position.reindex(pellets1.index, method='nearest')       # get position data when a pellet is delivered at patch1
    forage.plot.scatter('x','y',s=1,c='red',ax=ax1)                   # plot mouse positions when pellets were delivered

    for trial in pellets1.itertuples():                               # for each pellet delivery
        before = trial.Index - pd.to_timedelta(10, 's')               # get the previous 10 seconds
        path = position.loc[before:trial.Index]                       # get position data in the time before pellet delivery
        ax1.plot(path.x, path.y)                                      # plot path traces preceding pellet delivery

    ax2.hist(position.area, bins=100)                                 # plot histogram of tracked object size
    
    wheel1.plot(ax=ax3)                                               # plot distance travelled on patch1 wheel
    wheel1.plot(ax=ax4)                                               # plot distance travelled on patch2 wheel
    ax3.set_ylabel('distance (cm)')                                   # set axis label
    ax4.set_ylabel('distance (cm)')                                   # set axis label

    fig.savefig('{0}_{1}.png'.format(session.id,start.date()))        # save figure tagged with id and date
    plt.close(fig)                                                    # close figure