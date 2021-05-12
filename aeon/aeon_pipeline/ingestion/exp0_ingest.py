import pathlib

from aeon.aeon_pipeline import subject, experiment, tracking, paths
from aeon.preprocess import exp0_api


root = pathlib.Path('/ceph/aeon/test2/data')


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
                               'experiment_start_time': '2021-03-24 15:29:37.052000046',
                               'experiment_description': 'experiment 0',
                               'arena_name': 'circle-2m',
                               'lab': 'SWC',
                               'location': 'room-0'})


# ---------------- TimeBlock -----------------

def generate_timeblocks(experiment_name, root):
    for sessiondata_file in root.rglob('SessionData*.csv'):
        sessiondata = exp0_api.sessionreader(sessiondata_file.as_posix())

        try:
            sessiondata = exp0_api.sessionduration(sessiondata)
        except ValueError:
            continue

        start = sessiondata.iloc[0].name
        end = sessiondata.iloc[0].name + sessiondata.iloc[0].duration
        subjects = sessiondata.id.values

        if start > end:
            continue

        # --- insert to TimeBlock ---
        time_block_key = {'experiment_name': experiment_name, 'time_block_start': start}

        if time_block_key in experiment.TimeBlock.proj():
            continue

        experiment.TimeBlock.insert1({**time_block_key,
                                      'time_block_end': end})
        experiment.TimeBlock.Subject.insert({**time_block_key, 'subject': subj}
                                            for subj in subjects)

        # -- files --
        file_datetime_str = sessiondata_file.stem.replace('SessionData_', '')
        files = list(pathlib.Path(sessiondata_file.parent).glob(f'*{file_datetime_str}*'))

        repositories = {p: n for n, p in zip(*experiment.DataRepository.fetch(
            'repository_name', 'repository_path'))}

        data_root_dir = paths.find_root_directory(list(repositories.keys()), files[0])
        repository_name = repositories[data_root_dir.as_posix()]
        experiment.TimeBlock.File.insert(
            {**time_block_key,
             'file_number': f_idx,
             'file_name': f.name,
             'data_category': experiment.DataCategory.category_mapper[f.name.split('_')[0]],
             'repository_name': repository_name,
             'file_path': f.relative_to(data_root_dir).as_posix()}
            for f_idx, f in enumerate(files))
