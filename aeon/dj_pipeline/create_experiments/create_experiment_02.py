"""Function to create new experiments for experiment0.2."""

from aeon.dj_pipeline import acquisition, lab, subject

# ============ Manual and automatic steps to for experiment 0.2 populate ============
experiment_name = "exp0.2-r0"
_weight_scale_rate = 20


def create_new_experiment():
    """Create new experiment for experiment0.2."""
    # ---------------- Subject -----------------
    subject_list = [
        {"subject": "BAA-1100699", "sex": "U", "subject_birth_date": "2021-01-01"},
        {"subject": "BAA-1100700", "sex": "U", "subject_birth_date": "2021-01-01"},
        {"subject": "BAA-1100701", "sex": "U", "subject_birth_date": "2021-01-01"},
        {"subject": "BAA-1100702", "sex": "U", "subject_birth_date": "2021-01-01"},
        {"subject": "BAA-1100703", "sex": "U", "subject_birth_date": "2021-01-01"},
    ]
    subject.Subject.insert(subject_list, skip_duplicates=True)

    # ---------------- Experiment -----------------
    acquisition.Experiment.insert1(
        {
            "experiment_name": experiment_name,
            "experiment_start_time": "2022-02-22 09-00-00",
            "experiment_description": "experiment 0.2 - 24/7",
            "arena_name": "circle-2m",
            "lab": "SWC",
            "location": "room-0",
            "experiment_type": "foraging",
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
                "directory_path": "aeon/data/raw/AEON2/experiment0.2",
            },
            {
                "experiment_name": experiment_name,
                "repository_name": "ceph_aeon",
                "directory_type": "quality-control",
                "directory_path": "aeon/data/qc/AEON2/experiment0.2",
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


def main():
    """Main function to create a new experiment."""
    create_new_experiment()


if __name__ == "__main__":
    main()
