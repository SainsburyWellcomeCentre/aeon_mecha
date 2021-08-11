# Look at position traveled values

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

import aeon.query.api as aeon

# Get session metadata
root = '/ceph/aeon/test2/experiment0.1'
metadata = aeon.sessiondata(root)
# Add session End event for known sessions that weren't ended properly:
# 2021/7/6 BAA-1099791, 2021/7/16 BAA-1099791
missing_ends = [
    '2021-07-06 14:20:41.840610027',
    '2021-07-15 08:02:24.902180195']
for start in missing_ends:
    add_event = metadata[metadata.index == start]
    add_event.index += datetime.timedelta(hours=3)
    add_event.event = 'End'
    metadata = metadata.append(add_event)
metadata.sort_index(inplace=True)
annotations = aeon.annotations(root)
metadata = aeon.sessionduration(metadata)
# Get non-test sessions after 2021-06-24 greater than 1 hour in duration
metadata = metadata[metadata.id.str.startswith('BAA')]
metadata = metadata[metadata.start >= pd.Timestamp('20210624')]
metadata = metadata[(metadata.duration > pd.Timedelta('1h'))]
# Throwaway known bad sessions:
# 24/6/2021 BAA-1099795, 20/7/2021 BAA-1099790, 21/7/2021 BAA-1099791,
# 27/7/2021 BAA-1099793
bad_sessions = (
    ((metadata['start'].dt.date == datetime.date(2021, 6, 24))
       & (metadata['id'] == 'BAA-1099795'))
    | ((metadata['start'].dt.date == datetime.date(2021, 7, 20))
     & (metadata['id'] == 'BAA-1099790'))
    | ((metadata['start'].dt.date == datetime.date(2021, 7, 21))
     & (metadata['id'] == 'BAA-1099791'))
    | ((metadata['start'].dt.date == datetime.date(2021, 7, 27))
     & (metadata['id'] == 'BAA-1099793'))
)
metadata.drop(metadata[bad_sessions].index, inplace=True)

# We want to compare threshold values, and figure out what is threshold value at
# which mice move to easier patch.
# (range of [0,1]: 0 means no preference, 1 means complete preference)

# Create a df which will hold all required values for each session (row)
mouse_scores = (
    pd.DataFrame(
        index=range(45),
        columns=[
            'id', 'start', 'end', 'new_thresh_ts', 'new_thresh_val',
            'hard_patch', 'easy_patch', 'n_easy_pellets_pre_thresh_change',
            'n_hard_pellets_pre_thresh_change',
            'n_easy_pellets_post_thresh_change',
            'n_hard_pellets_post_thresh_change', 'init_hard_pref_score',
            'end_easy_pref_score']))

# For each session
for i, session in enumerate(metadata.itertuples()):
    # Get session start, end, duration, and mouse id from metadata
    id, start, end = (session.id, session.start, session.end)
    # Get threshold values for each pellet delivery
    p1 = aeon.patchdata(root, 'Patch1', start=start, end=end)
    p2 = aeon.patchdata(root, 'Patch2', start=start, end=end)
    # fed_1 = aeon.pelletdata(root, 'Patch1', start=start, end=end)
    # fed_2 = aeon.pelletdata(root, 'Patch2', start=start, end=end)
    # Search for threshold change timestamp and value
    p1_ut = np.unique(p1.threshold)
    p2_ut = np.unique(p2.threshold)
    if len(p1_ut) > len(p2_ut):
        new_thresh_ts = (
            p1.iloc[np.where(p1.threshold == p1_ut[-1])[0][0] - 1].name)
        new_thresh_val = p1_ut[-1]
        hard_patch = 1
    elif len(p2_ut) > len(p1_ut):
        new_thresh_ts = (
            p2.iloc[np.where(p2.threshold == p2_ut[-1])[0][0] - 1].name)
        new_thresh_val = p2_ut[-1]
        hard_patch = 2
    else:
        new_thresh_val = np.NaN
        if p1.iloc[-1].name < p2.iloc[-1].name:
            new_thresh_ts = p1.iloc[-1].name
            hard_patch = 1
        else:
            new_thresh_ts = p2.iloc[-1].name
            hard_patch = 2
    easy_patch = 1 if hard_patch == 2 else 2
    # Compute 'init_hard_pref_score' and 'end_easy_pref_score'

    # init_hard_pref = (hard_pellet_ct_pre_change / tot_pellet_ct_pre_change)
    p1_pellet_ct_pre_change = len(np.where(p1.index <= new_thresh_ts)[0])
    p2_pellet_ct_pre_change = len(np.where(p2.index <= new_thresh_ts)[0])
    tot_pellet_ct_pre_change = p1_pellet_ct_pre_change + p2_pellet_ct_pre_change
    if hard_patch == 1:
        hard_pellet_ct_pre_change = p1_pellet_ct_pre_change
    else:
        hard_pellet_ct_pre_change = p2_pellet_ct_pre_change
    init_hard_pref_score = hard_pellet_ct_pre_change / tot_pellet_ct_pre_change

    # end_easy_pref = (easy_pellet_ct_post_change / tot_pellet_ct_post_change)
    p1_pellet_ct_post_change = len(p1) - p1_pellet_ct_pre_change
    p2_pellet_ct_post_change = len(p2) - p2_pellet_ct_pre_change
    tot_pellet_ct_post_change = (
        p1_pellet_ct_post_change + p2_pellet_ct_post_change)
    if easy_patch == 1:
        easy_pellet_ct_post_change = p1_pellet_ct_post_change
    else:
        easy_pellet_ct_post_change = p2_pellet_ct_post_change
    end_easy_pref_score = easy_pellet_ct_post_change / tot_pellet_ct_post_change

    assert ((p1_pellet_ct_pre_change + p1_pellet_ct_post_change
            + p2_pellet_ct_pre_change + p2_pellet_ct_post_change)
            == (len(p1) + len(p2)))

    n_easy_pellets_pre_thresh_change = (
        tot_pellet_ct_pre_change - hard_pellet_ct_pre_change)
    n_hard_pellets_pre_thresh_change = hard_pellet_ct_pre_change
    n_easy_pellets_post_thresh_change = easy_pellet_ct_post_change
    n_hard_pellets_post_thresh_change = (
        tot_pellet_ct_post_change - easy_pellet_ct_post_change)
    # Assign values to dataframe
    mouse_scores.iloc[i] = [
        id, start, end, new_thresh_ts, new_thresh_val, hard_patch,
        easy_patch, n_easy_pellets_pre_thresh_change,
        n_hard_pellets_pre_thresh_change, n_easy_pellets_post_thresh_change,
        n_hard_pellets_post_thresh_change, init_hard_pref_score,
        end_easy_pref_score]

# For each mouse, create a scatter with init_hard vs. end_easy scores,
# color-coded by threshold change value, where 'x' marker means init_hard was
# patch1, and 'o' marker means end_easy was patch2
# where each point on the scatter is color-coded by patch
fig, axes = plt.subplots(nrows=5, ncols=1)
ids = np.unique(mouse_scores.id)

for ax, id in enumerate(ids):
    init_hard_1500 = mouse_scores[
        ((mouse_scores.id == id) &
         (mouse_scores.new_thresh_val == 1500))].init_hard_pref_score
    hard_patch_1500 = mouse_scores[
        ((mouse_scores.id == id) &
         (mouse_scores.new_thresh_val == 1500))].hard_patch
    end_easy_1500 = mouse_scores[
        ((mouse_scores.id == id) &
         (mouse_scores.new_thresh_val == 1500))].end_easy_pref_score

    init_hard_750 = mouse_scores[
        ((mouse_scores.id == id) &
         (mouse_scores.new_thresh_val == 750))].init_hard_pref_score
    hard_patch_750 = mouse_scores[
        ((mouse_scores.id == id) &
         (mouse_scores.new_thresh_val == 750))].hard_patch
    end_easy_750 = mouse_scores[
        ((mouse_scores.id == id) &
         (mouse_scores.new_thresh_val == 750))].end_easy_pref_score

    init_hard_187 = mouse_scores[
        ((mouse_scores.id == id) &
         (mouse_scores.new_thresh_val == 187.5))].init_hard_pref_score
    hard_patch_187 = mouse_scores[
        ((mouse_scores.id == id) &
         (mouse_scores.new_thresh_val == 187.5))].hard_patch
    end_easy_187 = mouse_scores[
        ((mouse_scores.id == id) &
         (mouse_scores.new_thresh_val == 187.5))].end_easy_pref_score
    # Scatter point-by-point to set marker based on hard patch
    for s, hard_patch in enumerate(hard_patch_1500):
        m = 'x' if hard_patch == 1 else 'o'
        axes[ax].scatter(init_hard_1500.iloc[s], end_easy_1500.iloc[s],
                         marker=m, color='r')
    for s, hard_patch in enumerate(hard_patch_750):
        m = 'x' if hard_patch == 1 else 'o'
        axes[ax].scatter(init_hard_750.iloc[s], end_easy_750.iloc[s],
                         marker=m, color='g')
    for s, hard_patch in enumerate(hard_patch_187):
        m = 'x' if hard_patch == 1 else 'o'
        axes[ax].scatter(init_hard_187.iloc[s], end_easy_187.iloc[s],
                         marker=m, color='b')
    # Prettify
    #axes[ax].set_ylabel(id[-3:])
    axes[ax].set_xlim([0.45, 1.05])
    axes[ax].set_ylim([-0.1, 1.1])
    axes[ax].grid(True)
    #axes[ax].set_xticks([], [])
    axes[ax].set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))

axes[-1].set_xticks((0.5, 0.6, 0.7, 0.8, 0.9, 1))
axes[-1].legend(['1500:2', '', '', '750:1', '', '', '187'])
ax_all = fig.add_subplot(111, frameon=False)
ax_all.tick_params(labelcolor='none', top=False, bottom=False,
                   left=False, right=False)
ax_all.set_xlabel('Start Hard Patch Preference')
ax_all.set_ylabel('End Easy Patch Preference')
ax_all.set_title('Mouse Patch Preference Color-coded by Threshold Change Value')

ax17[0].set_xlim([0, 0.25])
ax17_all = fig17.add_subplot(111, frameon=False)
ax17_all.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
ax17_all.text(x=0.25, y=0.99, ha='center', va='top', s='light condition')
ax17_all.text(x=0.8, y=0.99, ha='center', va='top', s='dark condition')
ax17_all.set_ylabel('probability')
ax17_all.set_xlabel('time (s)')
ax17_all.set_title('pmfs of isis and best-fit inverse gaussian process models')
# Overlay model fits
best_fit_model_light = ( np.sqrt(shape_param_light / (2 * np.pi * (bins[1:-1]) ** 3))
                        * ( np.exp(-shape_param_light * ((bins[1:-1] - mean_param_light) ** 2)
                            / (2 * mean_param_light ** 2 * bins[1:-1])) ) ) * 0.001
best_fit_model_dark = ( np.sqrt(shape_param_dark / (2 * np.pi * (bins[1:-1]) ** 3))
                        * ( np.exp(-shape_param_dark * ((bins[1:-1] - mean_param_dark) ** 2)
                            / (2 * mean_param_dark ** 2 * bins[1:-1])) ) * 0.001 )
model_light_plot = ax17[0].plot(bins[1:-1], best_fit_model_light, 'g')
model_dark_plot = ax17[1].plot(bins[1:-1], best_fit_model_dark, 'g')
ax17[1].legend(['best fit model', 'isi pmf'])

# See if mouse's initial patch preference changes over days.
# (range of [1,2]: 1 means pure preference for patch 1, 2 means pure preference
# for patch 2)

# init_preference_score = (
#             (patch_2_pellet_ct_pre_thresh_change
#             / tot_pellet_ct_pre_thresh_change)) + 1
