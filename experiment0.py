import aeon
import datetime
import matplotlib.pyplot as plt

root = '/ceph/aeon/test2/data'
data = aeon.sessiondata(root)

# fill missing data (crash on the 3rd day due to storage overflow)
oddsession = data[data.id == 'BAA-1099592'].iloc[4,:]             # take the start time of 3rd session
oddsession.name = oddsession.name + datetime.timedelta(hours=3)   # add three hours
oddsession.event = 'End'                                          # manually insert End event
data.loc[oddsession.name] = oddsession                            # insert the new row in the data frame
data.sort_index(inplace=True)                                     # sort chronologically

data = data[data.id.str.startswith('BAA')]                        # take only proper sessions
data = aeon.sessionduration(data)                                 # compute session duration
print(data.groupby('id').apply(lambda g:g[:].drop('id', axis=1))) # print session summary grouped by id

for session in data.itertuples():                                 # for all sessions
    print('{0} on {1}...'.format(session.id, session.Index))      # print progress report
    start = session.Index                                         # session start time is session index
    end = start + session.duration                                # end time = start time + duration
    encoder = aeon.encoderdata(root, start=start, end=end)        # get encoder data between start and end
    distance = aeon.distancetravelled(encoder.angle)              # compute total distance travelled
    fig = plt.figure()                                            # create figure
    ax = fig.add_subplot(1,1,1)                                   # with subplot
    distance.plot(ax=ax)                                          # plot distance travelled
    ax.set_ylim(-1, 12000)                                        # set fixed scale range
    ax.set_ylabel('distance (cm)')                                # set axis label
    fig.savefig('{0}_{1}.png'.format(session.id,start.date()))    # save figure tagged with id and date
    plt.close(fig)                                                # close figure
