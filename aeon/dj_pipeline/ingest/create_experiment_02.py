import yaml
from aeon.dj_pipeline import acquisition, lab, subject
from pathlib import Path

_weight_scale_rate = 100


# ============ Manual and automatic steps to for experiment 0.1 ingest ============
experiment_name = 'exp0.2-r0'


def create_new_experiment():
    # ---------------- Subject -----------------
    subject.Subject.insert([
        {'subject': 'BAA-1099790', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
        {'subject': 'BAA-1099791', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
        {'subject': 'BAA-1099792', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
        {'subject': 'BAA-1099793', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
        {'subject': 'BAA-1099794', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
        {'subject': 'BAA-1099795', 'sex': 'U', 'subject_birth_date': '2021-01-01'},
        {'subject': 'BAA-1099796', 'sex': 'U', 'subject_birth_date': '2021-01-01'}],
        skip_duplicates=True)

    # ---------------- Experiment -----------------
    if {'experiment_name': experiment_name} not in acquisition.Experiment.proj():
        acquisition.Experiment.insert1({'experiment_name': experiment_name,
                                        'experiment_start_time': '2021-06-03 07-00-00',
                                        'experiment_description': 'experiment 0.1',
                                        'arena_name': 'circle-2m',
                                        'lab': 'SWC',
                                        'location': 'room-0',
                                        'experiment_type': 'foraging'})
        acquisition.Experiment.Subject.insert([
            {'experiment_name': experiment_name, 'subject': 'BAA-1099790'},
            {'experiment_name': experiment_name, 'subject': 'BAA-1099791'},
            {'experiment_name': experiment_name, 'subject': 'BAA-1099792'},
            {'experiment_name': experiment_name, 'subject': 'BAA-1099793'},
            {'experiment_name': experiment_name, 'subject': 'BAA-1099794'},
            {'experiment_name': experiment_name, 'subject': 'BAA-1099795'}])
        acquisition.Experiment.Directory.insert(
            [{'experiment_name': experiment_name,
              'repository_name': 'ceph_aeon', 'directory_type': 'raw',
              'directory_path': 'test2/experiment0.1'},
             {'experiment_name': experiment_name,
              'repository_name': 'ceph_aeon', 'directory_type': 'quality-control',
              'directory_path': 'aeon/qc/experiment0.1'}
             ])

    if {'arena_name': 'circle-2m', 'nest': 1} not in lab.ArenaNest.proj():
        nest_coordinates = [(0.3264, 0.864), (0.3264, 1.0368), (0.4992, 0.864), (0.4992, 1.0368)]
        lab.ArenaNest.insert1({'arena_name': 'circle-2m', 'nest': 1})
        lab.ArenaNest.Vertex.insert(
            ({'arena_name': 'circle-2m', 'nest': 1, 'vertex': v_id, 'vertex_x': x, 'vertex_y': y}
             for v_id, (x, y) in enumerate(nest_coordinates)), skip_duplicates=True)


def add_arena_setup():
    # Arena Setup - Experiment Devices
    this_file = Path(__file__).expanduser().absolute().resolve()
    metadata_yml_filepath = this_file.parent / "setup_yml" / "Experiment0.1.yml"

    ingest_exp01_metadata(metadata_yml_filepath, experiment_name)

    # manually update coordinates of foodpatch and nest
    patch_coordinates = {'Patch1': (1.13, 1.59, 0),
                         'Patch2': (1.19, 0.50, 0)}

    for patch_key in acquisition.ExperimentFoodPatch.fetch('KEY'):
        patch = (acquisition.ExperimentFoodPatch
                 & {'experiment_name': experiment_name}
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
