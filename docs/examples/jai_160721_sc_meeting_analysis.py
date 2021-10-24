from datetime import datetime
dt = datetime.today()
import seaborn as sns
import os
from pathlib import Path
import math
import datetime
import numpy as np
import pandas as pd
from pandas.tseries import frequencies
import aeon.query.api as aeon
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# For each session, we want: mouse id, start datetime, end datetime, duration,
# start weight, end weight, mouse position, patch1 wheel trace, patch2 wheel
# trace, patch1 food, delivery times, patch2 food delivery times, behavior
# ethogram (patch1, patch2, arena, corridor, nest)

# Helper functions.
def distance(position, target):
    return np.sqrt(np.square(position[['x', 'y']] - target).sum(axis=1))


def activepatch(wheel, patch, position):
    exit_patch = patch.astype(np.int8).diff() < 0
    in_wheel = (wheel.diff().rolling('1s').sum() > 1).reindex(position.index,
                                                              method='pad')
    epochs = exit_patch.cumsum()
    return in_wheel.groupby(epochs).apply(lambda x: x.cumsum()) > 0


# Constants.
CAM_HZ = 50                # sampling rate of overhead camera
PIXEL_SCALE = 0.00192      # 1 px = 1.92 mm
ARENA_INNER_RADIUS = 0.93  # radius to the inner wall (m)
ARENA_OUTER_RADIUS = 0.97  # radius to the outer wall (m)
ARENA_RADIUS = 0.95        # radius to the middle of the corridor (m)
PATCH_RADIUS = 0.1         # radius around the patch to detect 'in patch' (m)
X0, Y0 = 1.475, 1.075      # center of arena
P1X, P1Y = 1.13, 1.59      # center of patch1
P2X, P2Y = 1.19, 0.50      # center of patch2

# Get good sessions (mouse sessions after 14 June greater than 1 hour)
# (returns id, start datetime, end datetime, duration, weight_start, weight_end)
output = 'exp01_figures'
root = '/ceph/aeon/test2/experiment0.1'
metadata = aeon.sessiondata(root)
annotations = aeon.annotations(root)
metadata = aeon.sessionduration(metadata)
metadata = metadata[metadata.id.str.startswith('BAA')]
metadata = metadata[metadata.start >= pd.Timestamp('20210614')]
metadata = metadata[(metadata.duration > pd.Timedelta('1h'))]

# df = pd.DataFrame(index=metadata.start)
# df['id'] = metadata.id.values
# for session, idx in zip(metadata.itertuples(), df.index):
#     start, end = session.start, session.end
#     # total distance traveled on wheel
#     encoder_1 = aeon.encoderdata(path=root, device='Patch1', start=start,
#                                  end=end)
#     encoder_2 = aeon.encoderdata(path=root, device='Patch2', start=start,
#                                  end=end)
#     df.loc[idx, 'patch1_dist'] = aeon.distancetravelled(encoder_1.angle)[-1]
#     df.loc[idx, 'patch2_dist'] = aeon.distancetravelled(encoder_2.angle)[-1]
#     # total pellets
#     fed_1 = aeon.pelletdata(root, 'Patch1', start=start, end=end)
#     fed_2 = aeon.pelletdata(root, 'Patch2', start=start, end=end)
#     df.loc[idx, 'patch1_pellets'] = len(fed_1[fed_1.values == 'TriggerPellet'])
#     df.loc[idx, 'patch2_pellets'] = len(fed_2[fed_2.values == 'TriggerPellet'])
#     # pellets patch 1 pre-threshold change
#     # pellets patch 2 post-threshold change

# Add keys for: mouse position, patch1 wheel trace, patch2 wheel trace,
# patch1 food delivery times, patch2 food delivery times, ethogram times (
# patch1, patch2, arena, corridor, nest)
# Get values for new columns from session data
data = {}
for session in metadata.itertuples():
    # Get session start, end, duration, mouse id, weight start, and weight
    # end from metadata
    start, end, duration, id, weight_start, weight_end = (
        session.start, session.end, session.duration, session.id,
        session.weight_start, session.weight_end)
    data[start] = {}
    data[start]['end'] = end
    data[start]['duration'] = duration
    data[start]['id'] = id
    data[start]['weight_start'] = weight_start
    data[start]['weight_end'] = weight_end

    # Get mouse position from online tracking.
    pos = aeon.positiondata(root, start=start, end=end)
    # Time offset to account for abnormal drop event
    if start > pd.Timestamp('20210621'):
        pos.index += pd.Timedelta('22.57966s')
    # Filter out NaNs and positions outside the arena.
    valid_pos = (pos.area > 0) & (pos.area < 1000)
    pos = pos[valid_pos]
    # Convert from pixel space to meter space.
    pos.x *= PIXEL_SCALE
    pos.y *= PIXEL_SCALE
    data[start]['pos'] = pos

    # Get distance traveled on wheel from encoder data.
    encoder_1 = aeon.encoderdata(path=root, device='Patch1', start=start,
                                 end=end)
    encoder_2 = aeon.encoderdata(path=root, device='Patch2', start=start,
                                 end=end)
    data[start]['wheel_1'] = aeon.distancetravelled(encoder_1.angle)
    data[start]['wheel_2'] = aeon.distancetravelled(encoder_2.angle)

    # Get food delivery from FED data.
    fed_1 = aeon.pelletdata(root, 'Patch1', start=start, end=end)
    fed_2 = aeon.pelletdata(root, 'Patch2', start=start, end=end)
    data[start]['fed_1'] = fed_1[fed_1.event == 'TriggerPellet']
    data[start]['fed_2'] = fed_2[fed_2.event == 'TriggerPellet']

    # Get ethogram info from known positions in arena and mouse position data:
    # In corridor if the distance from the center is between the inner and
    # outer walls.
    dist_from_0 = distance(pos, (X0, Y0))
    in_corridor = ( (dist_from_0 < ARENA_OUTER_RADIUS)
                    & (dist_from_0 > ARENA_INNER_RADIUS) )
    # In nest if distance from outer is greater than the outer wall radius
    in_nest = dist_from_0 > ARENA_OUTER_RADIUS
    # In patch if the distance to the patch center is less than the patch
    # radius.
    in_patch_1 = ( (np.abs(P1X - pos.x) <= PATCH_RADIUS)
                   & (np.abs(P1Y - pos.y) <= PATCH_RADIUS) )
    in_patch_2 = ( (np.abs(P2X - pos.x) <= PATCH_RADIUS)
                 & (np.abs(P2Y - pos.y) <= PATCH_RADIUS) )
    # In arena if not in any other area.
    in_arena = ~in_corridor & ~in_nest & ~in_patch_1 & ~in_patch_2
    ethogram = np.zeros(len(in_corridor))
    ethogram[in_patch_1] = 1
    ethogram[in_patch_2] = 2
    ethogram[in_arena] = 3
    ethogram[in_corridor] = 6
    ethogram[in_nest] = 10
    data[start]['ethogram'] = ethogram

    # Get time and pellet distributions for both patches for each `feeding`:
    # Wheel bout if in patch and wheel has moved more than 5 cm in 2 sec
    # rolling window.  (reindex df to put in same sampling rate as position (
    # i.e. camera) data
    wheel_bout_1 = (data[start]['wheel_1'].diff().rolling('2s').sum() >
                 5).reindex(pos.index, method='pad')
    wheel_bout_1.iloc[0] = False
    wheel_bout_2 = (data[start]['wheel_2'].diff().rolling('2s').sum() >
                 5).reindex(pos.index, method='pad')
    wheel_bout_2.iloc[0] = False

    # Get `enter_patch` -> `leave_patch` and 'feed_start' -> 'feed_end' "bout"
    # times (array of arrays): bouts are separated by more than 2s between
    # 'in' times
    in_patch_1_times = np.array(in_patch_1[in_patch_1.values].index)
    diff_patch_1_times = np.diff(in_patch_1_times).astype(np.float64) / 1e9
    in_patch_1_bout_idxs = np.where(diff_patch_1_times > 2)[0]
    in_patch_1_bout_end_times = (
        np.append(in_patch_1_times[in_patch_1_bout_idxs],
                  in_patch_1_times[-1]))
    in_patch_1_bout_start_times = (
        np.append(in_patch_1_times[0],
                  in_patch_1_times[(in_patch_1_bout_idxs + 1)]))
    data[start]['p_1_bout_duration'] = (
        (in_patch_1_bout_end_times - in_patch_1_bout_start_times).astype(
            np.float64) / 1e9)

    in_patch_2_times = np.array(in_patch_2[in_patch_2.values].index)
    diff_patch_2_times = np.diff(in_patch_2_times).astype(np.float64) / 1e9
    in_patch_2_bout_idxs = np.where(diff_patch_2_times > 2)[0]
    in_patch_2_bout_end_times = (
        np.append(in_patch_2_times[in_patch_2_bout_idxs],
                  in_patch_2_times[-1]))
    in_patch_2_bout_start_times = (
        np.append(in_patch_2_times[0],
                  in_patch_2_times[(in_patch_2_bout_idxs + 1)]))
    data[start]['p_2_bout_duration'] = (
        (in_patch_2_bout_end_times - in_patch_2_bout_start_times).astype(
            np.float64) / 1e9)

    wheel_bout_1_times = np.array(wheel_bout_1[wheel_bout_1.values].index)
    diff_wheel_bout_1_times = (
            np.diff(wheel_bout_1_times).astype(np.float64) / 1e9)
    wheel_bout_1_idxs = np.where(diff_wheel_bout_1_times > 10)[0]
    wheel_bout_1_end_times = (
        np.append(wheel_bout_1_times[wheel_bout_1_idxs],
                  wheel_bout_1_times[-1]))
    wheel_bout_1_start_times = (
        np.append(wheel_bout_1_times[0],
                  wheel_bout_1_times[(wheel_bout_1_idxs + 1)]))
    data[start]['wheel_1_bout_duration'] = (
        (wheel_bout_1_end_times - wheel_bout_1_start_times).astype(
            np.float64) / 1e9)
    pellets_in_feed_1 = np.zeros(len(wheel_bout_1_end_times))
    pellet_times_p_1 = data[start]['fed_1'].index
    for bout in range(len(wheel_bout_1_end_times)):
        pellets_in_feed_1[bout] = len(
            np.where((pellet_times_p_1 >= wheel_bout_1_start_times[bout])
                   & (pellet_times_p_1 <= wheel_bout_1_end_times[bout]))[0])
    data[start]['pellets_in_wheel_bout_1'] = pellets_in_feed_1
    data[start]['pellets_in_feed_1'] = pellets_in_feed_1[pellets_in_feed_1 != 0]
    data[start]['wheel_1_pct_engaged'] = (
        len(wheel_bout_1_times) / len(data[start]['wheel_1']))

    wheel_bout_2_times = np.array(wheel_bout_2[wheel_bout_2.values].index)
    diff_wheel_bout_2_times = (
            np.diff(wheel_bout_2_times).astype(np.float64) / 1e9)
    wheel_bout_2_idxs = np.where(diff_wheel_bout_2_times > 10)[0]
    wheel_bout_2_end_times = (
        np.append(wheel_bout_2_times[wheel_bout_2_idxs],
                  wheel_bout_2_times[-1]))
    wheel_bout_2_start_times = (
        np.append(wheel_bout_2_times[0],
                  wheel_bout_2_times[(wheel_bout_2_idxs + 1)]))
    data[start]['wheel_2_bout_duration'] = (
        (wheel_bout_2_end_times - wheel_bout_2_start_times).astype(
            np.float64) / 1e9)
    pellets_in_feed_2 = np.zeros(len(wheel_bout_2_end_times))
    pellet_times_p_2 = data[start]['fed_2'].index
    for bout in range(len(wheel_bout_2_end_times)):
        pellets_in_feed_2[bout] = len(
            np.where((pellet_times_p_2 >= wheel_bout_2_start_times[bout])
                   & (pellet_times_p_2 <= wheel_bout_2_end_times[bout]))[0])
    data[start]['pellets_in_wheel_bout_2'] = pellets_in_feed_2
    data[start]['pellets_in_feed_2'] = pellets_in_feed_2[pellets_in_feed_2 != 0]
    data[start]['wheel_2_pct_engaged'] = (
            len(wheel_bout_2_times) / len(data[start]['wheel_2']))

    # Get % time `in_patch` but not `feeding`
    in_patch_1_not_feeding = (
        1 - (len(wheel_bout_1_times) / len(in_patch_1_times)))
    in_patch_2_not_feeding = (
        1 - (len(wheel_bout_2_times) / len(in_patch_2_times)))
    data[start]['in_p_1_not_feeding'] = in_patch_1_not_feeding
    data[start]['in_p_2_not_feeding'] = in_patch_2_not_feeding

    # Get transition for patch1 -> 1/2 and patch2 -> 1/2
    eth_diff = np.diff(ethogram)
    p1_to_arena = np.where(eth_diff == 2)[0]
    arena_to_p1 = np.where(eth_diff == -2)[0]
    p2_to_arena = np.where(eth_diff == 1)[0]
    arena_to_p2 = np.where(eth_diff == -1)[0]
    full_paths = []

    p1_to_p1_ct = 0
    p1_to_p2_ct = 0
    p2_to_p1_ct = 0
    p2_to_p2_ct = 0
    p_same_trans_no_nest = 0
    p_diff_trans_no_nest = 0
    p_same_trans_nest = 0
    p_diff_trans_nest = 0
    for p1_trans in p1_to_arena:
        p1_to_p1 = np.argmax(ethogram[(p1_trans + 1):] == 1)
        p1_to_p1 = len(ethogram) if (p1_to_p1 == 0) else p1_to_p1
        p1_to_p2 = np.argmax(ethogram[(p1_trans + 1):] == 2)
        p1_to_p2 = len(ethogram) if (p1_to_p2 == 0) else p1_to_p2
        full_path = (
            np.unique(ethogram[(p1_trans - 1)
                      : (p1_trans + (np.min([p1_to_p1, p1_to_p2]) + 2))],
                      return_index=True))
        full_path = full_path[0][np.argsort(full_path[1])]
        full_paths.append(full_path)
        if (p1_to_p1 < p1_to_p2):
            p1_to_p1_ct += 1
            if 10 in full_path:
                p_same_trans_nest += 1
            else:
                p_same_trans_no_nest += 1
        else:
            p1_to_p2_ct += 1
            if 10 in full_path:
                p_diff_trans_nest += 1
            else:
                p_diff_trans_no_nest += 1
    for p2_trans in p2_to_arena:
        p2_to_p2 = np.argmax(ethogram[(p2_trans + 1):] == 2)
        p2_to_p2 = len(ethogram) if (p2_to_p2 == 0) else p2_to_p2
        p2_to_p1 = np.argmax(ethogram[(p2_trans + 1):] == 1)
        p2_to_p1 = len(ethogram) if (p2_to_p1 == 0) else p2_to_p1
        full_path = (
            np.unique(ethogram[(p2_trans - 1)
                      : (p2_trans + (np.min([p2_to_p1, p2_to_p2]) + 2))],
                      return_index=True))
        full_path = full_path[0][np.argsort(full_path[1])]
        full_paths.append(full_path)
        if (p2_to_p2 < p2_to_p1):
            p2_to_p2_ct += 1
            if 6 in full_path:
                p_same_trans_nest += 1
            else:
                p_same_trans_no_nest += 1
        else:
            p2_to_p1_ct += 1
            if 6 in full_path:
                p_diff_trans_nest += 1
            else:
                p_diff_trans_no_nest += 1

    total_trans = p1_to_p1_ct + p1_to_p2_ct + p2_to_p1_ct + p2_to_p2_ct
    p1_to_p1_pct = (p1_to_p1_ct / total_trans)
    p1_to_p2_pct = (p1_to_p2_ct / total_trans)
    p2_to_p2_pct = (p2_to_p2_ct / total_trans)
    p2_to_p1_pct = (p2_to_p1_ct / total_trans)

    data[start]['p1_to_p1_ct'] = p1_to_p1_ct
    data[start]['p1_to_p2_ct'] = p1_to_p2_ct
    data[start]['p2_to_p2_ct'] = p2_to_p2_ct
    data[start]['p2_to_p1_ct'] = p2_to_p1_ct
    data[start]['total_p_trans'] = total_trans
    data[start]['p_same_trans_no_nest'] = p_same_trans_no_nest
    data[start]['p_diff_trans_no_nest'] = p_diff_trans_no_nest
    data[start]['p_same_trans_nest'] = p_same_trans_nest
    data[start]['p_diff_trans_nest'] = p_diff_trans_nest
    data[start]['p_trans_full_paths'] = full_paths

dkeys = list(data.keys())
data_exp0 = {}
for i in range(13):
	data_exp0[dkeys[i]] = data[dkeys[i]]
dkeys_exp0 = list(data_exp0.keys())
# Get all patch transitions
p1_to_p1_ct_tot = 0
p1_to_p2_ct_tot = 0
p2_to_p1_ct_tot = 0
p2_to_p2_ct_tot = 0
p_same_trans_no_nest_tot = 0
p_diff_trans_no_nest_tot = 0
p_same_trans_nest_tot = 0
p_diff_trans_nest_tot = 0
p_total_trans_ct = 0
for dkey in dkeys_exp0:
    p1_to_p1_ct_tot += data[dkey]['p1_to_p1_ct']
    p1_to_p2_ct_tot += data[dkey]['p1_to_p2_ct']
    p2_to_p1_ct_tot += data[dkey]['p2_to_p1_ct']
    p2_to_p2_ct_tot += data[dkey]['p2_to_p2_ct']
    p_same_trans_no_nest_tot += data[dkey]['p_same_trans_no_nest']
    p_diff_trans_no_nest_tot += data[dkey]['p_diff_trans_no_nest']
    p_same_trans_nest_tot += data[dkey]['p_same_trans_nest']
    p_diff_trans_nest_tot += data[dkey]['p_diff_trans_nest']
    p_total_trans_ct += data[dkey]['total_p_trans']

labels = [f'P1 -> P1 ({p1_to_p1_ct_tot})', f'P1 -> P2 ({p1_to_p2_ct_tot})',
          f'P2 -> P2 ({p2_to_p2_ct_tot})', f'P2 -> P1 ({p2_to_p1_ct_tot})']
p1_to_p1_pct = (p1_to_p1_ct_tot / p_total_trans_ct)
p1_to_p2_pct = (p1_to_p2_ct_tot / p_total_trans_ct)
p2_to_p2_pct = (p2_to_p2_ct_tot / p_total_trans_ct)
p2_to_p1_pct = (p2_to_p1_ct_tot / p_total_trans_ct)
fig, ax = plt.subplots()
ax.bar(labels[0], p1_to_p1_pct, width=0.3, label='p1_to_p1', color=[0, 0, 1])
ax.bar(labels[1], p1_to_p2_pct, width=0.3, label='p1_to_p2',
       color=[0.2, 0.2, 0.8])
ax.bar(labels[2], p2_to_p2_pct, width=0.3, label='p2_to_p2', color=[1, 0, 0])
ax.bar(labels[3], p2_to_p1_pct, width=0.3, label='p2_to_p1', color=[0.8, 0.2,
                                                                  0.2])
ax.legend()
ax.set_title('Patch-to-Patch Transition Probabilities: Symmetric Sessions')
fig.savefig('/ceph/aeon/aeon/code/scratchpad/jai_figs/patch_trans_probs.svg',
            dpi=300, format='svg', transparent=True)

labels = [f'same via nest ({p_same_trans_nest_tot})',
          f'diff via nest {p_diff_trans_nest_tot})',
          f'same no nest ({p_same_trans_no_nest_tot})',
          f'diff no nest ({p_diff_trans_no_nest_tot})']
p1_to_p1_pct = (p_same_trans_nest_tot / p_total_trans_ct)
p1_to_p2_pct = (p_diff_trans_nest_tot / p_total_trans_ct)
p2_to_p2_pct = (p_same_trans_no_nest_tot / p_total_trans_ct)
p2_to_p1_pct = (p_diff_trans_no_nest_tot / p_total_trans_ct)
fig2, ax2 = plt.subplots()
ax2.bar(labels[0], p1_to_p1_pct, width=0.3, label='same via nest',
       color=[0.1, 0.1, 0.1])
ax2.bar(labels[1], p1_to_p2_pct, width=0.3, label='diff via nest',
       color=[0.3, 0.3, 0.3])
ax2.bar(labels[2], p2_to_p2_pct, width=0.3, label='same no nest',
       color=[0.2, 0.7, 0.2])
ax2.bar(labels[3], p2_to_p1_pct, width=0.3, label='diff no nest',
       color=[0.1, 0.8, 0.1])
ax2.legend()
ax2.set_title('Patch via Nest Transition Probabilities: Symmetric Sessions')
fig2.savefig('/ceph/aeon/aeon/code/scratchpad/jai_figs/patch_nest_trans_probs'
            '.svg', dpi=300, format='svg', transparent=True)

# Patch full routes back to next patch
# blue=patch1, red=patch2, black=corridor/nest, green=arena
fp = data[dkey]['p_trans_full_paths']

pel_tot_1 = np.array(())
pel_tot_2 = np.array(())
feed_time_tot_1 = np.array(())
feed_time_tot_2 = np.array(())
pel_per_ses_tot_1 = np.array(())
pel_per_ses_tot_2 = np.array(())
time_per_ses_tot_1 = np.array(())
time_per_ses_tot_2 = np.array(())
for dkey in dkeys[:-1]:
    pel_tot_1 = np.append(pel_tot_1, data[dkey]['pellets_in_feed_1'])
    pel_tot_2 = np.append(pel_tot_2, data[dkey]['pellets_in_feed_2'])
    feed_time_tot_1 = np.append(feed_time_tot_1,
                                data[dkey]['wheel_1_bout_duration'])
    feed_time_tot_2 = np.append(feed_time_tot_2,
                                data[dkey]['wheel_2_bout_duration'])
    pel_per_ses_tot_1 = np.append(pel_per_ses_tot_1,
                                  len(data[dkey]['fed_1']))
    pel_per_ses_tot_2 = np.append(pel_per_ses_tot_2,
                                  len(data[dkey]['fed_2']))

# Pellet distribution per feed bout
fig3, ax3 = plt.subplots()
# ax2.hist(data[start]['pellets_in_feed_1'], color='b', label='patch1')
# ax2.hist(data[start]['pellets_in_feed_2'], color='r', label='patch2')
sns.kdeplot(pel_tot_1, clip=[0, 1000], color='blue',
            label=f'patch1 ({len(pel_tot_1)} bouts)', ax=ax3)
sns.kdeplot(pel_tot_2, clip=[0, 1000], color='red',
            label=f'patch2 ({len(pel_tot_2)} bouts)', ax=ax3)
ax3.set_title('Distribution of Pellets per Feeding Bout per Patch')
ax3.set_xlabel('Pellet Count')
ax3.legend()
fig3.savefig('/ceph/aeon/aeon/code/scratchpad/jai_figs'
             '/pellets_feeding_bout_dist.svg', dpi=300, format='svg',
             transparent=True)

# Time distribution per wheel bout
fig4, ax4 = plt.subplots()
sns.kdeplot(feed_time_tot_1, clip=[0, 2000], color='blue',
            label=f'patch1 ({len(pel_tot_1)} bouts)', ax=ax4)
sns.kdeplot(feed_time_tot_2, clip=[0, 1000], color='red',
            label=f'patch2 ({len(pel_tot_2)} bouts)', ax=ax4)
ax4.set_title('Distribution of Time Durations of Feeding Bout per Patch')
ax4.set_xlabel('Time Duration of Feeding Bout')
ax4.legend()
fig4.savefig('/ceph/aeon/aeon/code/scratchpad/jai_figs'
             '/time_feeding_bout_dist.svg', dpi=300, format='svg',
             transparent=True)

# Pellet distribution per patch per session
fig5, ax5 = plt.subplots()
sns.kdeplot(pel_per_ses_tot_1, clip=[0, 1000], color='blue',
            label=f'patch1: ('
                  f'{np.int(np.sum(pel_per_ses_tot_1))} total pellets)', ax=ax5)
sns.kdeplot(pel_per_ses_tot_2, clip=[0, 1000], color='red',
            label=f'patch1: ('
                  f'{np.int(np.sum(pel_per_ses_tot_2))} total pellets)', ax=ax5)
ax5.set_title('Distribution of Pellets per Patch over Sessions')
ax5.set_xlabel('Pellets per Session')
ax5.legend()
fig5.savefig('/ceph/aeon/aeon/code/scratchpad/jai_figs'
             '/pellets_per_sesh_dist.svg', dpi=300, format='svg',
             transparent=True)

# Ethogram distribution per session
in_patch1_tot = np.array(())
in_patch2_tot = np.array(())
in_arena_tot = np.array(())
in_corridor_tot = np.array(())
in_nest_tot = np.array(())

in_patch1_tot_dur = pd.Timedelta(seconds=0)
in_patch2_tot_dur = pd.Timedelta(seconds=0)
in_arena_tot_dur = pd.Timedelta(seconds=0)
in_corridor_tot_dur = pd.Timedelta(seconds=0)
in_nest_tot_dur = pd.Timedelta(seconds=0)

# Distribution of feed bouts per session per patch
feed_bouts_1_tot = np.array(())
feed_bouts_2_tot = np.array(())
for dkey in dkeys[:-1]:
    feed_bouts_1_tot = np.append(feed_bouts_1_tot, len(data[dkey][
        'pellets_in_feed_1']))
    feed_bouts_2_tot = np.append(feed_bouts_2_tot, len(data[dkey][
        'pellets_in_feed_2']))

fig5, ax5 = plt.subplots()
sns.kdeplot(feed_bouts_1_tot, clip=[0, 100], color='blue',
            label=f'patch1: ('
                  f'{np.int(np.sum(feed_bouts_1_tot))} total feed bouts)',
            ax=ax5)
sns.kdeplot(feed_bouts_2_tot, clip=[0, 100], color='red',
            label=f'patch2: ('
                  f'{np.int(np.sum(feed_bouts_2_tot))} total feed bouts)',
            ax=ax5)
ax5.set_title('Distribution of Feed Bouts per Patch over Sessions')
ax5.set_xlabel('Feed Bouts per Session')
ax5.legend()
fig5.savefig('/ceph/aeon/aeon/code/scratchpad/jai_figs'
             '/feed_bouts_per_sesh_dist.svg', dpi=300, format='svg',
             transparent=True)

# red, blue, green, gray, black
for dkey in dkeys[:-1]:
    eth = data[dkey]['ethogram']
    in_patch1_tot = np.append(in_patch1_tot,
                              (len(np.where(eth == 1)[0]) / len(eth)))
    in_patch2_tot = np.append(in_patch2_tot,
                              (len(np.where(eth == 2)[0]) / len(eth)))
    in_arena_tot = np.append(in_arena_tot,
                            (len(np.where(eth == 3)[0]) / len(eth)))
    in_corridor_tot = np.append(in_corridor_tot,
                               (len(np.where(eth == 6)[0]) / len(eth)))
    in_nest_tot = np.append(in_nest_tot,
                            (len(np.where(eth == 10)[0]) / len(eth)))
    in_patch1_tot_dur += in_patch1_tot[-1] * data[dkey]['duration']
    in_patch2_tot_dur += in_patch2_tot[-1] * data[dkey]['duration']
    in_arena_tot_dur += in_arena_tot[-1] * data[dkey]['duration']
    in_corridor_tot_dur += in_corridor_tot[-1] * data[dkey]['duration']
    in_nest_tot_dur += in_nest_tot[-1] * data[dkey]['duration']

fig6, ax6 = plt.subplots()
sns.kdeplot(in_patch1_tot, clip=[0, 1], color='blue',
            label=f'patch1:'
                  f' {np.round((in_patch1_tot_dur.total_seconds() / 3600), 2)}'
                  f' hours', ax=ax6)
sns.kdeplot(in_patch2_tot, clip=[0, 1], color='red',
            label=f'patch2:'
                  f' {np.round((in_patch2_tot_dur.total_seconds() / 3600), 2)}'
                  f' hours', ax=ax6)
sns.kdeplot(in_arena_tot, clip=[0, 1], color='green',
            label=f'arena:'
                  f' {np.round((in_arena_tot_dur.total_seconds() / 3600), 2)}'
                  f' hours', ax=ax6)
sns.kdeplot(in_corridor_tot, clip=[0, 1], color=[0.5, 0.5, 0.5],
            label=f'corridor:'
                  f' '
                  f'{np.round((in_corridor_tot_dur.total_seconds() / 3600), 2)}'
                  f' hours', ax=ax6)
sns.kdeplot(in_nest_tot, clip=[0, 1], color=[0.1, 0.1, 0.1],
            label=f'nest:'
                  f' {np.round((in_nest_tot_dur.total_seconds() / 3600), 2)}'
                  f' hours', ax=ax6)
ax6.set_title('Distributions of Locations, Normalized by Session Time')
ax6.set_xlabel('Session Normalized Time')
ax6.legend()
fig6.savefig('/ceph/aeon/aeon/code/scratchpad/jai_figs'
             '/ethogram_distribution.svg', dpi=300, format='svg',
             transparent=True)


ax3.hist(data[start]['wheel_1_bout_duration'], color='b', label='patch1')
ax3.hist(data[start]['wheel_2_bout_duration'], color='r', label='patch2')
ax3.set_title('Distribution of Time Duration per Feed per Patch')
ax3.set_xlabel('Time Duration')
ax3.legend()

len(ethogram[in_corridor.values]) + len(ethogram[in_nest.values]) + len(
    ethogram[in_patch_1.values]) + len(ethogram[in_patch_2.values]) + len(ethogram[in_arena.values])


# For threshold switch sessions, get ratio of pellets between
# patches before/after threshold switch

# Speed of the animal overlaid on ethogram overlaid with feeding bout times
