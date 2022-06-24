import pathlib

from aeon.dj_pipeline import acquisition, lab, subject
from aeon.dj_pipeline.populate.create_experiment_01 import ingest_exp01_metadata

# ============ Manual and automatic steps to for experiment 0.1 populate ============
experiment_name = "social0-r1"


def create_new_experiment():
    # ---------------- Subject -----------------
    subject_list = [
        {"subject": "BAA-1100704", "sex": "U", "subject_birth_date": "2021-01-01"},
        {"subject": "BAA-1100705", "sex": "U", "subject_birth_date": "2021-01-01"},
        {"subject": "BAA-110705", "sex": "U", "subject_birth_date": "2021-01-01"},
        {"subject": "BAA1100705", "sex": "U", "subject_birth_date": "2021-01-01"},
        {"subject": "BAA-1100706", "sex": "U", "subject_birth_date": "2021-01-01"},
        {"subject": "BAA-117005", "sex": "U", "subject_birth_date": "2021-01-01"},
    ]
    subject.Subject.insert(subject_list, skip_duplicates=True)

    # ---------------- Experiment -----------------
    acquisition.Experiment.insert1(
        {
            "experiment_name": experiment_name,
            "experiment_start_time": "2021-11-30 14:00:00",
            "experiment_description": "social experiment 0",
            "arena_name": "circle-2m",
            "lab": "SWC",
            "location": "room-1",
            "experiment_type": "social",
        },
        skip_duplicates=True,
    )
    acquisition.Experiment.Subject.insert(
        [
            {"experiment_name": experiment_name, "subject": s["subject"]}
            for s in subject_list
        ],
        skip_duplicates=True,
    )

    acquisition.Experiment.Directory.insert(
        [
            {
                "experiment_name": experiment_name,
                "repository_name": "ceph_aeon",
                "directory_type": "raw",
                "directory_path": "aeon/data/raw/ARENA0/socialexperiment0",
            },
            {
                "experiment_name": experiment_name,
                "repository_name": "ceph_aeon",
                "directory_type": "quality-control",
                "directory_path": "aeon/data/qc/ARENA0/socialexperiment0",
            },
        ],
        skip_duplicates=True,
    )

    if {"arena_name": "circle-2m", "nest": 1} not in lab.ArenaNest.proj():
        nest_coordinates = [
            (0.3264, 0.864),
            (0.3264, 1.0368),
            (0.4992, 0.864),
            (0.4992, 1.0368),
        ]
        lab.ArenaNest.insert1({"arena_name": "circle-2m", "nest": 1})
        lab.ArenaNest.Vertex.insert(
            (
                {
                    "arena_name": "circle-2m",
                    "nest": 1,
                    "vertex": v_id,
                    "vertex_x": x,
                    "vertex_y": y,
                }
                for v_id, (x, y) in enumerate(nest_coordinates)
            ),
            skip_duplicates=True,
        )


def add_arena_setup():
    # Arena Setup - Experiment Devices
    this_file = pathlib.Path(__file__).expanduser().absolute().resolve()
    metadata_yml_filepath = this_file.parent / "setup_yml" / "SocialExperiment0.yml"

    ingest_exp01_metadata(metadata_yml_filepath, experiment_name)

    # manually update coordinates of foodpatch and nest
    patch_coordinates = {"Patch1": (1.13, 1.59, 0), "Patch2": (1.19, 0.50, 0)}

    for patch_key in (
        acquisition.ExperimentFoodPatch & {"experiment_name": experiment_name}
    ).fetch("KEY"):
        patch = (acquisition.ExperimentFoodPatch & patch_key).fetch1(
            "food_patch_description"
        )
        x, y, z = patch_coordinates[patch]
        acquisition.ExperimentFoodPatch.Position.update1(
            {
                **patch_key,
                "food_patch_position_x": x,
                "food_patch_position_y": y,
                "food_patch_position_z": z,
            }
        )


def main():
    create_new_experiment()
    add_arena_setup()


def fixID(subjid, valid_ids=None, valid_id_file=None):
    """
    Legacy helper function for socialexperiment0 - originaly developed by ErlichLab
    https://github.com/SainsburyWellcomeCentre/aeon_mecha/blob/ee1fa536b58e82fad01130d7689a70e68f94ec0e/aeon/util/helpers.py#L19

    Attempt to correct the id entered by the technician
    Attempt to correct the subjid entered by the technician
    Input:
    subjid              The subjid to fix
    (
    valid_ids           A list of valid_ids
    OR
    valid_id_file       A fully qualified filename of a csv file with `id`, and `roomid` columns.
    )
    """
    from os import path
    import jellyfish as jl
    import pandas as pd
    import numpy as np

    if not valid_ids:
        if not valid_id_file:
            valid_id_file = path.expanduser("~/mnt/delab/conf/valid_ids.csv")

        df = pd.read_csv(valid_id_file)
        valid_ids = df.id.values

    # The id is good
    # The subjid is good
    if subjid in valid_ids:
        return subjid

    # The id has a comment
    # The subjid has a comment
    if "/" in subjid:
        return fixID(subjid.split("/")[0], valid_ids=valid_ids)

    # The id is a combo id.
    # The id is a combo id.
    # The subjid is a combo subjid.
    if ";" in subjid:
        subjidA, subjidB = subjid.split(";")
        return f"{fixID(subjidA.strip(), valid_ids=valid_ids)};{fixID(subjidB.strip(), valid_ids=valid_ids)}"

    if "vs" in subjid:
        subjidA, tmp, subjidB = subjid.split(" ")[1:]
        return f"{fixID(subjidA.strip(), valid_ids=valid_ids)};{fixID(subjidB.strip(), valid_ids=valid_ids)}"

    try:
        ld = [jl.levenshtein_distance(subjid, x[-len(subjid) :]) for x in valid_ids]
        return valid_ids[np.argmin(ld)]
    except:
        return subjid


if __name__ == "__main__":
    main()
