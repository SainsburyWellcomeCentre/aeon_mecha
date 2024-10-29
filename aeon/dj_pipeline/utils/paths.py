"""
Utility functions for working with paths in the context of the DJ pipeline.
"""

from __future__ import annotations

import pathlib

from aeon.dj_pipeline import repository_config


def get_repository_path(repository_name: str) -> pathlib.Path:
    """Find the directory's full-path corresponding to a given repository_name.

    This function looks up the repository path based on the provided repository_name
    using the configuration stored in dj.config['custom']['repository_config'].

    Args:
        repository_name (str): The name of the repository to find.

    Returns:
        pathlib.Path: The full path to the directory corresponding to the repository_name.
    """
    repo_path = repository_config.get(repository_name)
    if repo_path is None:
        raise ValueError(f"Repository name not configured: {repository_name}")

    repo_path = pathlib.Path(repo_path)

    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path not found: {repo_path}")

    return repo_path


def find_root_directory(
    root_directories: str | pathlib.Path, full_path: str | pathlib.Path
) -> pathlib.Path:
    """Given multiple potential root directories and a full-path,
    search and return one directory that is the parent of the given path.

    Args:
        root_directories (str | pathlib.Path): A list of potential root directories.
        full_path (str | pathlib.Path): The full path to search for the root directory.

    Raises:
        FileNotFoundError: If the specified `full_path` does not exist.
        FileNotFoundError: If no valid root directory is found among the provided options.

    Returns:
        pathlib.Path: The full path to the discovered root directory.
    """
    full_path = pathlib.Path(full_path)

    if not full_path.exists():
        raise FileNotFoundError(f"{full_path} does not exist!")

    # turn to list if only a single root directory is provided
    if isinstance(root_directories, (str, pathlib.Path)):
        root_directories = [root_directories]

    try:
        return next(
            pathlib.Path(root_dir)
            for root_dir in root_directories
            if pathlib.Path(root_dir) in set(full_path.parents)
        )

    except StopIteration as err:
        raise FileNotFoundError(
            f"No valid root directory found (from {root_directories})"
            f" for {full_path}"
        ) from err
