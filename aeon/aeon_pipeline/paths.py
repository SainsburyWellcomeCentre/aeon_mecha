import pathlib


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