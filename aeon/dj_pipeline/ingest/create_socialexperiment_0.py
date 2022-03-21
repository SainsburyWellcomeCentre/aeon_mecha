import pathlib

from aeon.dj_pipeline import acquisition, lab, subject
from aeon.dj_pipeline.ingest.create_experiment_01 import ingest_exp01_metadata

# ============ Manual and automatic steps to for experiment 0.1 ingest ============
experiment_name = 'social0-r1'


def create_new_experiment():
    # ---------------- Subject -----------------
    subject_list = [
        {'subject': 'BAA-1100704', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
        {'subject': 'BAA-1100705', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
        {'subject': 'BAA-110705', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
        {'subject': 'BAA1100705', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
        {'subject': 'BAA-1100706', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
        {'subject': 'BAA-117005', 'sex': 'U', 'subject_birth_date': '2021-01-01'}
    ]
    subject.Subject.insert(subject_list, skip_duplicates=True)

    # ---------------- Experiment -----------------
    acquisition.Experiment.insert1({'experiment_name': experiment_name,
                                    'experiment_start_time': '2021-11-30 14:00:00',
                                    'experiment_description': 'social experiment 0',
                                    'arena_name': 'circle-2m',
                                    'lab': 'SWC',
                                    'location': 'room-1',
                                    'experiment_type': 'social'}, skip_duplicates=True)
    acquisition.Experiment.Subject.insert([
        {'experiment_name': experiment_name, 'subject': s['subject']}
        for s in subject_list], skip_duplicates=True)

    acquisition.Experiment.Directory.insert(
        [{'experiment_name': experiment_name,
          'repository_name': 'ceph_aeon', 'directory_type': 'raw',
          'directory_path': 'aeon/preprocessed/socialexperiment0'},
         {'experiment_name': experiment_name,
          'repository_name': 'ceph_aeon', 'directory_type': 'quality-control',
          'directory_path': 'aeon/qc/socialexperiment0'}
         ], skip_duplicates=True)

    if {'arena_name': 'circle-2m', 'nest': 1} not in lab.ArenaNest.proj():
        nest_coordinates = [(0.3264, 0.864), (0.3264, 1.0368), (0.4992, 0.864), (0.4992, 1.0368)]
        lab.ArenaNest.insert1({'arena_name': 'circle-2m', 'nest': 1})
        lab.ArenaNest.Vertex.insert(
            ({'arena_name': 'circle-2m', 'nest': 1, 'vertex': v_id, 'vertex_x': x, 'vertex_y': y}
             for v_id, (x, y) in enumerate(nest_coordinates)), skip_duplicates=True)


def add_arena_setup():
    # Arena Setup - Experiment Devices
    this_file = pathlib.Path(__file__).expanduser().absolute().resolve()
    metadata_yml_filepath = this_file.parent / "setup_yml" / "SocialExperiment0.yml"

    ingest_exp01_metadata(metadata_yml_filepath, experiment_name)

    # manually update coordinates of foodpatch and nest
    patch_coordinates = {'Patch1': (1.13, 1.59, 0),
                         'Patch2': (1.19, 0.50, 0)}

    for patch_key in (acquisition.ExperimentFoodPatch
                      & {'experiment_name': experiment_name}).fetch('KEY'):
        patch = (acquisition.ExperimentFoodPatch
                 & patch_key).fetch1('food_patch_description')
        x, y, z = patch_coordinates[patch]
        acquisition.ExperimentFoodPatch.Position.update1({
            **patch_key,
            'food_patch_position_x': x,
            'food_patch_position_y': y,
            'food_patch_position_z': z})


def main():
    create_new_experiment()
    add_arena_setup()


if __name__ == '__main__':
    main()
