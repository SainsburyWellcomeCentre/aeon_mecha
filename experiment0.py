import aeon

root = '/ceph/aeon/test2/data'
data = aeon.sessiondata(root)
data = data[data.id.str.startswith('BAA')].groupby('id')
print(data.apply(lambda g:g[:].drop('id', axis=1)))