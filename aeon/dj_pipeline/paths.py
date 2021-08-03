import datajoint as dj
import pathlib


def get_repository_path(repository_name):
    """
    Find the directory's full-path corresponding to a given repository_name,
    as configured in dj.config['custom']['repository_config']
    """
    if 'repository_config' not in dj.config['custom']:
        raise ValueError('Missing repository configuration ("repository_config") in "custom"')

    repo_path = dj.config['custom']['repository_config'].get(repository_name)
    if repo_path is None:
        raise ValueError(f'Repository name not configured: {repository_name}')

    repo_path = pathlib.Path(repo_path)

    if not repo_path.exists():
        raise FileNotFoundError(f'Repository path not found: {repo_path}')

    return repo_path


def find_root_directory(root_directories, full_path):
    """
    Given multiple potential root directories and a full-path,
    search and return one directory that is the parent of the given path
        :param root_directories: potential root directories
        :param full_path: the relative path to search the root directory
        :return: full-path (pathlib.Path object)
    """
    full_path = pathlib.Path(full_path)

    if not full_path.exists():
        raise FileNotFoundError(f'{full_path} does not exist!')

    # turn to list if only a single root directory is provided
    if isinstance(root_directories, (str, pathlib.Path)):
        root_directories = [root_directories]

    try:
        return next(pathlib.Path(root_dir) for root_dir in root_directories
                    if pathlib.Path(root_dir) in set(full_path.parents))

    except StopIteration:
        raise FileNotFoundError('No valid root directory found (from {})'
                                ' for {}'.format(root_directories, full_path))