import os
import aeon
import datetime
import subprocess

root = '/ceph/aeon/test2/data'
output = os.path.expanduser('~/aeon/data/experiment0')
bonsai = os.path.expanduser('~/bonsai/Bonsai.Player/bin/Debug/net5.0/Bonsai.Player')
tracking = 'bonsai/tracking.bonsai'
export = 'bonsai/export.bonsai'
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

def exportvideo(path, prefix, workflow, framerate, start, end, **kwargs):
    """Exports and analyses a continuous session video segment."""
    videoclip = aeon.videoclip(root, prefix, start, end)          # get top video clip between start and end
    videoinfo = '{0}/{1}-Clips.csv'.format(path, prefix)          # get clip segment info file name
    videoclip.to_csv(videoinfo)                                   # save top video clip segment info
    args = [                                                      # assemble required workflow arguments
        bonsai,
        workflow,
        '-p:ChunkFile={0}'.format(videoinfo),
        '-p:TrackingFile={0}'.format('{0}/{1}.csv'.format(path, prefix)),
        '-p:VideoFile={0}'.format('{0}/{1}.avi'.format(path, prefix)),
        '-p:FrameRate={0}'.format(framerate)]
    for key, value in kwargs.items():                             # add any extra keyword arguments
        args.append('-p:{0}={1}'.format(key, value))
    subprocess.call(args)                                         # call bonsai passing all workflow arguments

for session in data.itertuples():                                 # for all sessions
    start = session.Index                                         # session start time is session index
    end = start + session.duration                                # end time = start time + duration
    fname = start.floor('s').isoformat().replace(':','-')         # get a path safe representation of time
    path = '{0}/{1}/{2}'.format(output, session.id, fname)        # format the full name of the output folder
    print('Exporting {0}...'.format(path))                        # print progress report
    os.makedirs(path, exist_ok=True)                              # ensure output 
    exportvideo(path, 'FrameTop', tracking, 50, start, end,       # track and export top video
                Threshold=36)
    exportvideo(path, 'FrameSide', export, 125, start, end)       # export side video
