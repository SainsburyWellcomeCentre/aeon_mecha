from aeon.aeon_pipeline import subject, experiment


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
    {'experiment_name': 'exp0-r0', 'subject': 'BAA-1099592'}])
experiment.Experiment.Directory.insert1({'experiment_name': 'exp0-r0',
                                         'repository_name': 'ceph_aeon_test2',
                                         'directory_path': 'data/2021-03-25T15-05-34'})

# ---------------- Equipment -----------------

experiment.ExperimentCamera.insert([
    {'experiment_name': 'exp0-r0', 'camera_id': 0,
     'camera_installed_time': '2021-03-25 15-00-00', 'sampling_rate': 50},
    {'experiment_name': 'exp0-r0', 'camera_id': 1,
     'camera_installed_time': '2021-03-25 15-00-00', 'sampling_rate': 125}])

experiment.ExperimentFoodPatch.insert([
    {'experiment_name': 'exp0-r0', 'food_patch_id': 0,
     'food_patch_installed_time': '2021-03-25 15-00-00'}])
experiment.ExperimentFoodPatch.RemovalTime.insert([
    {'experiment_name': 'exp0-r0', 'food_patch_id': 0,
     'food_patch_installed_time': '2021-03-25 15-00-00',
     'food_patch_removed_time': '2021-03-26 12:00:00'}])
experiment.ExperimentFoodPatch.insert([
    {'experiment_name': 'exp0-r0', 'food_patch_id': 0,
     'food_patch_installed_time': '2021-03-26 12:00:00'}])

# ---------------- Auto Ingestion -----------------


experiment.TimeBin.generate_timebins(experiment_name='exp0-r0')
experiment.SubjectCrossingEvent.populate()
experiment.SubjectEpoch.populate()
experiment.FoodPatchEvent.populate()

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


keys = (SubjectEpoch * TimeBin).fetch("KEY")
key = keys[4]

file_repo, file_path = (TimeBin.File * DataRepository
                        & 'data_category = "SessionMeta"' & key).fetch1(
    'repository_path', 'file_path')
sessiondata_file = pathlib.Path(file_repo) / file_path

root = sessiondata_file.parent

start, end = (SubjectEpoch & key).fetch1('epoch_start', 'epoch_end')
start = pd.Timestamp(start)
end = pd.Timestamp(end)

encoderdata = exp0_api.encoderdata(root.parent.as_posix(), start=start, end=end)
pelletdata = exp0_api.pelletdata(root.parent.as_posix(), start=start, end=end)
patchdata = exp0_api.patchdata(root.parent.as_posix(), start=start, end=end)
videodata = exp0_api.videodata(root.parent.as_posix(), start=start, end=end, prefix='')
