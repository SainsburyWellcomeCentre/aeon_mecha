import pathlib
import datetime

from aeon.aeon_pipeline import subject, experiment, paths

# ---------------- Some constants -----------------

root = pathlib.Path('/ceph/aeon/test2/data/2021-03-25T15-05-34')
_bin_duration = datetime.timedelta(hours=3)


# ---------------- Subject -----------------
subject.Subject.insert([
    {'subject': 'Dario', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
    {'subject': 'dfs', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
    {'subject': 'f', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
    {'subject': 'BAA-1099590', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
    {'subject': 'BAA-1099591', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
    {'subject': 'BAA-1099592', 'sex': 'U', 'subject_birth_date': '2021-01-01'}
])


# ---------------- Experiment -----------------
experiment.Experiment.insert1({'experiment_name': 'exp0-r0',
                               'experiment_start_time': '2021-03-25 15-00-00',
                               'experiment_description': 'experiment 0',
                               'arena_name': 'circle-2m',
                               'lab': 'SWC',
                               'location': 'room-0'})
experiment.Experiment.Subject.insert([
    {'experiment_name': 'exp0-r0', 'subject': 'BAA-1099590'},
    {'experiment_name': 'exp0-r0', 'subject': 'BAA-1099591'},
    {'experiment_name': 'exp0-r0', 'subject': 'BAA-1099592'}
])


# ---------------- TimeBin -----------------


def generate_timebins(experiment_name, root_data_dir):
    sessiondata_files = sorted(list(root.rglob('SessionData*.csv')))

    time_bin_list, file_list = [], []
    for sessiondata_file in sessiondata_files:
        timebin_str = sessiondata_file.stem.split("_")[-1]
        date_str, time_str = timebin_str.split("T")
        time_bin_start = datetime.datetime.fromisoformat(date_str + "T" + time_str.replace("-", ":"))
        time_bin_end = time_bin_start + _bin_duration

        # --- insert to TimeBin ---
        time_bin_key = {'experiment_name': experiment_name,
                        'time_bin_start': time_bin_start}

        if time_bin_key in experiment.TimeBin.proj():
            continue

        time_bin_list.append({**time_bin_key,
                              'time_bin_end': time_bin_end})

        # -- files --
        file_datetime_str = sessiondata_file.stem.replace('SessionData_', '')
        files = list(pathlib.Path(sessiondata_file.parent).glob(f'*{file_datetime_str}*'))

        repositories = {p: n for n, p in zip(*experiment.DataRepository.fetch(
            'repository_name', 'repository_path'))}

        data_root_dir = paths.find_root_directory(list(repositories.keys()), files[0])
        repository_name = repositories[data_root_dir.as_posix()]
        file_list.extend(
            {**time_bin_key,
             'file_number': f_idx,
             'file_name': f.name,
             'data_category': experiment.DataCategory.category_mapper[f.name.split('_')[0]],
             'repository_name': repository_name,
             'file_path': f.relative_to(data_root_dir).as_posix()}
            for f_idx, f in enumerate(files))

    # insert
    print(f'Insert {len(time_bin_list)} new TimeBin')

    with experiment.TimeBin.connection.transaction:
        experiment.TimeBin.insert(time_bin_list)
        experiment.TimeBin.File.insert(file_list)


generate_timebins('exp0-r0', root)
experiment.SubjectPassageEvent.populate()
experiment.SubjectEpoch.populate()


# ============ OLD =============

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