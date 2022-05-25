from os import path
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import aeon.analyze.patches as patches
import aeon.preprocess.api as api
import aeon.util.plotting as aplot
from functools import cache
import jellyfish as jl

DEBUG = True

if DEBUG:
    eprint = print
else:
    def eprint(*args):
        pass


def fixID(subjid, valid_ids=None, valid_id_file=None):
    """
    Attempt to correct the id entered by the technician
    Attempt to correct the subjid entered by the technician
    Input:
    subjid              The subjid to fix
    (
    valid_ids           A list of valid_ids
    OR
    valid_id_file       A fully qualified filename of a csv file with `id`, and `roomid` columns.
    )
    """

    if not valid_ids:
        if not valid_id_file:
            valid_id_file = path.expanduser('~/mnt/delab/conf/valid_ids.csv')

        df = pd.read_csv(valid_id_file)
        valid_ids = df.id.values

    # The id is good
    # The subjid is good
    if subjid in valid_ids:
        return subjid

    # The id has a comment
    # The subjid has a comment
    if '/' in subjid:
        return fixID(subjid.split('/')[0], valid_ids=valid_ids)

    # The id is a combo id.
    # The id is a combo id.
    # The subjid is a combo subjid.
    if ';' in subjid:
        subjidA, subjidB = subjid.split(';')
        return f"{fixID(subjidA.strip(), valid_ids=valid_ids)};{fixID(subjidB.strip(), valid_ids=valid_ids)}"

    if 'vs' in subjid:
        subjidA, tmp, subjidB = subjid.split(' ')[1:]
        return f"{fixID(subjidA.strip(), valid_ids=valid_ids)};{fixID(subjidB.strip(), valid_ids=valid_ids)}"

    try:
        ld = [jl.levenshtein_distance(subjid, x[-len(subjid):]) for x in valid_ids]
        return valid_ids[np.argmin(ld)]
    except:
        return subjid


def loadSessions(dataroot, valid_ids=None):
    """
    loadSessions is a helper function to analyse data from arena0.
    It removes some test sessions, fixes the subject ids and
    merges interrupted and social sessions.
    """
    sessdf = api.sessiondata(dataroot)
    sessdf = api.sessionduration(sessdf)  # compute session duration
    sessdf = sessdf[~sessdf.id.str.contains('test')]
    sessdf = sessdf[~sessdf.id.str.contains('jeff')]
    sessdf = sessdf[~sessdf.id.str.contains('OAA')]
    sessdf = sessdf[~sessdf.id.str.contains('rew')]
    sessdf = sessdf[~sessdf.id.str.contains('Animal')]
    sessdf = sessdf[~sessdf.id.str.contains('white')]

    sessdf.reset_index(inplace=True, drop=True)

    df = sessdf.copy()

    if valid_ids is None:
        valid_id_file = path.expanduser('~/mnt/delab/conf/valid_ids.csv')
        vdf = pd.read_csv(valid_id_file)
        valid_ids = list(vdf.id.values[vdf.real.values == 1])

    fix = lambda x: fixID(x, valid_ids=valid_ids)
    df.id = df.id.apply(fix)
    mergeSocial(df)
    merge(df)
    # %%
    markSessionEnded(df)
    return df


def merge(df, first=[], merge_id=False):
    """
    merge(df, first=[]):

    Modifies the input! if you want to save it make a copy first
    df: a dataframe that is the output of sessiondata
    If `first` is empty, then merge tries match adjacent sessions by ID.
    It's called `first` because it is the index of the first of two sessions to merge.

    Note: if a session was interrupted twice, then it will not handle it properly.
    Also fixes missing end Time

    Note: if you have removed rows from your dataframe you are likely to get
        keyerrors. To avoid this reset_index.
    """

    id = df.id.values
    # You need the values, because you are otherwise still in a pandas type
    if not first:
        for i in range(0, len(id) - 1):
            if ((df.loc[i, 'id'] == df.loc[i + 1, 'id']) and
                    (df.loc[i, 'start'].date() == df.loc[i + 1, 'start'].date())):
                first.append(i)

    for i in first:
        df.loc[i, 'end'] = df.loc[i + 1, 'end']
        df.loc[i, 'weight_end'] = df.loc[i + 1, 'weight_end']
        if merge_id:
            df.loc[i, 'id'] = '{};{}'.format(df.loc[i, 'id'], df.loc[i + 1, 'id'])

    second = [i + 1 for i in first]
    df.drop(index=second, inplace=True)
    df.reset_index(inplace=True, drop=True)


def mergeSocial(df):
    """
    mergeSocial
    modifies input!!!
    finds sessions that started close together and had different IDs.
    Assumes they are a social session.
    """
    first = _findSocial(df)
    merge(df, first=first, merge_id=True)


def _findSocial(df, threshold_in_minutes=15):
    started_close = [df.start[x] - df.start[x - 1] < pd.Timedelta(threshold_in_minutes, "min") for x in
                     range(1, len(df.start))]  # < pd.Timedelta(10, "min")
    # not_ended = np.where(pd.isnull(df.end))[0]
    not_same_id = [df.id[x] != df.id[x - 1] for x in range(1, len(df.start))]
    return list(np.where(np.bitwise_and(started_close, not_same_id))[0])


def markSessionEnded(df, offset=pd.DateOffset(minutes=100)):
    bad_end = df.end.isnull()
    df.loc[bad_end, "end"] = df.loc[bad_end, 'start'] + offset
    df.duration = df.end - df.start


def getSessionID(session):
    return f"{session.id.split('/')[0].replace(';', ':').replace(' ', '')}_{session.start:%Y-%m-%dT%H:%M:%S}"


def stateChangeRows(df, patchid=1):
    """
    Extract state change times from the patches.
    I'm sure there is a more elegant way to do this.
    """
    switchind = np.diff(df.threshold).nonzero()[0] + 1
    switchind = np.append(np.insert(switchind, 0, 0), len(df.threshold) - 1)

    return pd.DataFrame({
        "time": [pd.Timestamp(x) for x in df.index[switchind]],
        "threshold": df.threshold[switchind].values,
        "patch": df.threshold[switchind].values * 0 + patchid
    })


def getSwitchTime(sdf):
    patches = sdf.patch.unique()
    stime = []
    for p in patches:
        tdf = sdf.loc[sdf.patch == p]
        if tdf.shape[0] < 3:
            sdf.drop(sdf.patch == p)

        sdf.time[1:-1]


def splitOnStateChange(root, start, end):
    """
    startts, endts = splitOnStateChange(root, start, end)
    split up the session into sections that have the same state.
    ## Output
    list startts, endts, data.
    If there is no switch return [start,],[end,] for consistency.
    """
    state1 = api.patchdata(root, 'Patch1', start=start, end=end)  # get patch state for patch1 between start and end
    state2 = api.patchdata(root, 'Patch2', start=start, end=end)
    s1 = stateChangeRows(state1, 1)
    s2 = stateChangeRows(state2, 2)
    sdf = pd.concat([s1, s2]).set_index('time').sort_index()
    eprint(state1.shape, state2.shape, sdf.shape)

    switchlist = [pd.Timestamp(x) for x in sdf.index[2:-2]]
    startts = switchlist.copy()
    startts.insert(0, start)
    switchlist.append(end)  # For some reason using np.append gives the wrong type

    return startts, switchlist, sdf


# @cache
def getWheelData(root, start, end):
    """
    Helper function that gets the wheel data from both patches in arena0
    and returns the data in a dictionary of dataframes.
    """
    encoder1 = api.encoderdata(root, 'Patch1', start=start,
                               end=end)  # get encoder data for patch1 between start and end
    encoder2 = api.encoderdata(root, 'Patch2', start=start,
                               end=end)  # get encoder data for patch2 between start and end
    pellets1 = api.pelletdata(root, 'Patch1', start=start,
                              end=end)  # get pellet events for patch1 between start and end
    pellets2 = api.pelletdata(root, 'Patch2', start=start,
                              end=end)  # get pellet events for patch2 between start and end
    state1 = api.patchdata(root, 'Patch1', start=start, end=end)  # get patch state for patch1 between start and end
    state2 = api.patchdata(root, 'Patch2', start=start, end=end)  # get patch state for patch2 between start and end

    wheel1 = api.distancetravelled(encoder1.angle)  # compute total distance travelled on patch1 wheel
    wheel2 = api.distancetravelled(encoder2.angle)  # compute total distance travelled on patch2 wheel
    pellets1 = pellets1[pellets1.event == 'TriggerPellet']  # get timestamps of pellets delivered at patch1
    pellets2 = pellets2[pellets2.event == 'TriggerPellet']
    return {"pellets1": pellets1,
            "pellets2": pellets2,
            "state1": state1,
            "state2": state2,
            "wheel1": wheel1,
            "wheel2": wheel2,
            }


def exportWheelData(root, session, *,
                    datadir,
                    format='parquet',
                    force=False
                    ):
    data = getWheelData(root, session.start, session.end)
    sessid = getSessionID(session)
    for k, v in data.items():
        fullfile = path.join(datadir, f"{k}_{sessid}.{format}")
        if path.exists(fullfile) and not force:
            print(f'{fullfile} exists. Skipping')
        else:
            if isinstance(v, pd.core.series.Series):
                v = v.to_frame()
            v['time_in_session'] = (v.index - session.start).total_seconds().array

            if format == 'parquet':
                v.to_parquet(fullfile, index=False)
            elif format == 'csv':
                v.to_csv(fullfile, index=False)

            print(f'Saved {fullfile}')


def getPositionData(root, start, end, duration=None):
    """
    Returns a list of dictionaries with keys:
    position       a dataframe with the data
    frequency      Samples/sec (seems this could be computed from the data)
    positionrange  the x/y lims of the data? for plotting?
    """
    try:
        return list(map(lambda x, y: _getPositionData(root, x, y, duration=duration), start, end))
    except TypeError:
        return [_getPositionData(root, start, end, duration=None), ]


# @cache
def _getPositionData(root, start, end, duration=None):
    if duration:
        end = start + duration
    frequency = 50  # frame rate in Hz
    pixelscale = 0.00192  # JCE: This hasn't been calibrated for our arena.
    # Shouldn't it go into metadata ??                                             # 1 px = 1.92 mm
    positionrange = [[0, 1440 * pixelscale], [0, 1080 * pixelscale]]  # frame position range in metric units
    position = api.positiondata(root, start=start, end=end)  # get position data between start and end
    # if start > pd.Timestamp('20210621') and \
    #    start < pd.Timestamp('20210701'):                              # time offset to account for abnormal drop event
    #     position.index = position.index + pd.Timedelta('22.57966s')   # exact offset extracted from video timestamps
    valid_position = (position.area > 0) & (position.area < 10000)  # filter for objects of the correct size
    position = position[valid_position]  # get only valid positions
    position.x = position.x * pixelscale  # scale position data to metric units
    position.y = position.y * pixelscale
    return {"position": position, "frequency": frequency, "positionrange": positionrange}


def ethogram(root, start, end):
    frequency = 50  # frame rate in Hz
    pixelscale = 0.00192  # 1 px = 1.92 mm
    positionrange = [[0, 1440 * pixelscale], [0, 1080 * pixelscale]]  # frame position range in metric units
    position = api.positiondata(root, start=start, end=end)  # get position data between start and end
    # if start > pd.Timestamp('20210621') and \
    #    start < pd.Timestamp('20210701'):                              # time offset to account for abnormal drop event
    #     position.index = position.index + pd.Timedelta('22.57966s')   # exact offset extracted from video timestamps
    valid_position = (position.area > 0) & (position.area < 10000)  # filter for objects of the correct size
    position = position[valid_position]  # get only valid positions
    position.x = position.x * pixelscale  # scale position data to metric units
    position.y = position.y * pixelscale
    # compute ethogram based on distance to patches, nest and corridor
    radius = 0.95  # middle
    inner = 0.93  # inner
    outer = 0.97  # outer
    patchradius = 0.21  # patch radius
    x0, y0 = 1.475, 1.075  # center
    p1x, p1y = 1050 * pixelscale, 580 * pixelscale  # 1.13, 1.59 # patch1 1050 580
    p2x, p2y = pixelscale * 384, pixelscale * 588  # 1.19, 0.50 # patch2
    #  p1x, p1y = p1x-0.02, p1y+0.07 # offset1 what do these do?!?
    #  p2x, p2y = p2x-0.04, p2y-0.04 # offset2
    dist0 = patches.distance(position, (x0, y0))
    distp1 = patches.distance(position, (p1x, p1y))
    distp2 = patches.distance(position, (p2x, p2y))
    in_patch1 = patches.activepatch(wheel1, distp1 < patchradius)
    in_patch2 = patches.activepatch(wheel2, distp2 < patchradius)
    in_arena = ~in_corridor & ~in_nest & ~in_patch1 & ~in_patch2
    ethogram = pd.Series('other', index=position.index)
    ethogram[in_patch1] = 'patch1'
    ethogram[in_patch2] = 'patch2'
    ethogram[in_arena] = 'arena'
    return ethogram