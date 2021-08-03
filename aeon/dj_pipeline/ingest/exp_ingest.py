from aeon.dj_pipeline import lab, subject, experiment, tracking, analysis
from aeon.dj_pipeline.ingestion import load_arena_setup


# ============ Manual and automatic steps to for experiment 0.1 ingest ============


# ---------------- Subject -----------------
subject.Subject.insert([
    {'subject': 'BAA-1099790', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
    {'subject': 'BAA-1099791', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
    {'subject': 'BAA-1099792', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
    {'subject': 'BAA-1099793', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
    {'subject': 'BAA-1099794', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
    {'subject': 'BAA-1099795', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
    {'subject': 'BAA-1099796', 'sex': 'U', 'subject_birth_date': '2021-01-01'}
], skip_duplicates=True)


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
yml_filepath = '/nfs/nhome/live/thinh/code/ProjectAeon/aeon/aeon/dj_pipeline/ingest/setup_yml/Experiment0.1.yml'

load_arena_setup(yml_filepath, experiment_name)

# manually update coordinates of foodpatch and nest
patch_coordinates = {'Patch1': (1.13, 1.59, 0),
                     'Patch2': (1.19, 0.50, 0)}

for patch_key in experiment.ExperimentFoodPatch.fetch('KEY'):
    patch = (experiment.ExperimentFoodPatch & patch_key).fetch1('food_patch_description')
    x, y, z = patch_coordinates[patch]
    experiment.ExperimentFoodPatch.Position.update1({
        **patch_key,
        'food_patch_position_x': x,
        'food_patch_position_y': y,
        'food_patch_position_z': z})

nest_coordinates = [(0.3264, 0.864), (0.3264, 1.0368), (0.4992, 0.864), (0.4992, 1.0368)]
lab.ArenaNest.insert1({'arena_name': 'circle-2m', 'nest': 1})
lab.ArenaNest.Vertex.insert(
    ({'arena_name': 'circle-2m', 'nest': 1, 'vertex': v_id, 'vertex_x': x, 'vertex_y': y}
     for v_id, (x, y) in enumerate(nest_coordinates)))

# ---------------- Auto Ingestion -----------------
settings = {'reserve_jobs': True, 'suppress_errors': True, 'display_progress': True}

# priority-high
experiment.TimeBin.generate_timebins(experiment_name='exp0.1-r0')
experiment.SubjectEnterExit.populate(**settings)
experiment.SubjectAnnotation.populate(**settings)
experiment.SubjectWeight.populate(**settings)

experiment.FoodPatchEvent.populate(**settings)
experiment.WheelState.populate(**settings)

experiment.Session.populate(**settings)
experiment.SessionEnd.populate(**settings)
experiment.SessionEpoch.populate(**settings)

# priority-middle
tracking.SubjectPosition.populate(**settings)

analysis.SessionTimeDistribution.populate(**settings)
analysis.SessionSummary.populate(**settings)

# priority-low
experiment.FoodPatchWheel.populate(**settings)
tracking.SubjectDistance.populate(**settings)

