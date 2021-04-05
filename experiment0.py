import aeon
import datetime

root = '/ceph/aeon/test2/data'
data = aeon.sessiondata(root)

# fill missing data (crash on the 3rd day due to storage overflow)
oddsession = data[data.id == 'BAA-1099592'].iloc[4,:]
oddsession.name = oddsession.name + datetime.timedelta(hours=3)
oddsession.event = 'End'
data.loc[oddsession.name] = oddsession
data.sort_index(inplace=True)

data = data[data.id.str.startswith('BAA')]
data = aeon.sessionduration(data)
print(data.groupby('id').apply(lambda g:g[:].drop('id', axis=1)))