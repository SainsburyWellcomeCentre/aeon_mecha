from aeon.aeon_pipeline import subject, experiment, tracking
from aeon.aeon_pipeline.ingestion import load_arena_setup


# ============ Manuel and automatic steps to for experiment 0.1 ingestion ============


# ---------------- Subject -----------------
subject.Subject.insert([
    {'subject': 'BAA-1099790', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
    {'subject': 'BAA-1099791', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
    {'subject': 'BAA-1099792', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
    {'subject': 'BAA-1099793', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
    {'subject': 'BAA-1099794', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
    {'subject': 'BAA-1099795', 'sex': 'U', 'subject_birth_date': '2021-01-01'}
])


# ---------------- Experiment -----------------
experiment.Experiment.insert1({'experiment_name': 'exp0.1-r0',
                               'experiment_start_time': '2021-06-03 07-00-00',
                               'experiment_description': 'experiment 0.1',
                               'arena_name': 'circle-2m',
                               'lab': 'SWC',
                               'location': 'room-0'})
experiment.Experiment.Subject.insert([
    {'experiment_name': 'exp0.1-r0', 'subject': 'BAA-1099790'},
    {'experiment_name': 'exp0.1-r0', 'subject': 'BAA-1099791'},
    {'experiment_name': 'exp0.1-r0', 'subject': 'BAA-1099792'},
    {'experiment_name': 'exp0.1-r0', 'subject': 'BAA-1099793'},
    {'experiment_name': 'exp0.1-r0', 'subject': 'BAA-1099794'},
    {'experiment_name': 'exp0.1-r0', 'subject': 'BAA-1099795'}])
experiment.Experiment.Directory.insert1({'experiment_name': 'exp0.1-r0',
                                         'directory_type': 'raw',
                                         'repository_name': 'ceph_aeon',
                                         'directory_path': 'test2/experiment0.1'})

# Arena Setup - Experiment Devices
experiment_name = 'exp0.1-r0'
yml_filepath = '/nfs/nhome/live/thinh/code/ProjectAeon/aeon/aeon/aeon_pipeline/ingestion/setup_yml/Experiment0.1.yml'

load_arena_setup(yml_filepath, experiment_name)

# ---------------- Auto Ingestion -----------------
settings = {'suppress_errors': True}

experiment.TimeBin.generate_timebins(experiment_name='exp0.1-r0')
experiment.SubjectEnterExit.populate(**settings)
experiment.SubjectAnnotation.populate(**settings)
experiment.Epoch.populate(**settings)
experiment.FoodPatchEvent.populate(**settings)
tracking.SubjectPosition.populate(**settings)
tracking.EpochPosition.populate(**settings)
