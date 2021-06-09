import os
import aeon
import datetime
import subprocess

rawdata = '/ceph/aeon/test2/data'
preprocess = '/ceph/aeon/aeon/preprocess'
output = os.path.expanduser('~/aeon/data/experiment0')
bonsai = '/ceph/aeon/aeon/code/bonsai/Bonsai.Player/bin/Debug/net5.0/Bonsai.Player'
tracking = 'tracking.bonsai'
export = 'export.bonsai'
data = aeon.sessiondata(rawdata)

# fill missing data (crash on the 3rd day due to storage overflow)
oddsession = data[data.id == 'BAA-1099592'].iloc[4,:]             # take the start time of 3rd session
oddsession.name = oddsession.name + datetime.timedelta(hours=3)   # add three hours
oddsession.event = 'End'                                          # manually insert End event
data.loc[oddsession.name] = oddsession                            # insert the new row in the data frame
data.sort_index(inplace=True)                                     # sort chronologically

data = data[data.id.str.startswith('BAA')]                        # take only proper sessions
data = aeon.sessionduration(data)                                 # compute session duration
print(data.groupby('id').apply(lambda g:g[:].drop('id', axis=1))) # print session summary grouped by id

def exportvideo(srcpath, dstpath, prefix, workflow, **kwargs):
    """Exports and analyses a continuous session video segment."""
    args = [                                                      # assemble required workflow arguments
        bonsai,
        workflow,
        '-p:TrackingFile={0}'.format('{0}/{1}.csv'.format(dstpath, prefix)),
        '-p:VideoFile={0}'.format('{0}/{1}.avi'.format(srcpath, prefix))]
    for key, value in kwargs.items():                             # add any extra keyword arguments
        args.append('-p:{0}={1}'.format(key, value))
    subprocess.call(args)                                         # call bonsai passing all workflow arguments

chunks = aeon.timebin_glob(rawdata)
for chunk in chunks:
    exportvideo(rawdata, preprocess, chunk, tracking, Threshold=36)
