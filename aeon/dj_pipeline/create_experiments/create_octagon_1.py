"""Function to create new experiments for octagon1.0"""

from aeon.dj_pipeline import acquisition, subject

# ============ Manual and automatic steps to for experiment 0.2 populate ============
experiment_name = "oct1.0-r0"
_weight_scale_rate = 20


def create_new_experiment():
    """Create new experiment for octagon1.0"""
    # ---------------- Subject -----------------
    # This will get replaced by content from colony.csv
    subject_list = [
        {"subject": "A001", "sex": "U", "subject_birth_date": "2021-01-01"},
        {"subject": "A002", "sex": "U", "subject_birth_date": "2021-01-01"},
        {"subject": "A003", "sex": "U", "subject_birth_date": "2021-01-01"},
        {"subject": "A004", "sex": "U", "subject_birth_date": "2021-01-01"},
        {"subject": "A005", "sex": "U", "subject_birth_date": "2021-01-01"},
        {"subject": "A006", "sex": "U", "subject_birth_date": "2021-01-01"},
        {"subject": "A007", "sex": "U", "subject_birth_date": "2021-01-01"},
    ]
    subject.Subject.insert(subject_list, skip_duplicates=True)

    # ---------------- Experiment -----------------
    acquisition.Experiment.insert1(
        {
            "experiment_name": experiment_name,
            "experiment_start_time": "2022-02-22 09-00-00",
            "experiment_description": "octagon 1.0",
            "arena_name": "octagon-1m",
            "lab": "SWC",
            "location": "464",
            "experiment_type": "social",
        },
        skip_duplicates=True,
    )
    acquisition.Experiment.Subject.insert(
        [{"experiment_name": experiment_name, "subject": s["subject"]} for s in subject_list],
        skip_duplicates=True,
    )

    acquisition.Experiment.Directory.insert(
        [
            {
                "experiment_name": experiment_name,
                "repository_name": "ceph_aeon",
                "directory_type": "raw",
                "directory_path": "aeon/data/raw/OCTAGON01/conf1",
            },
            {
                "experiment_name": experiment_name,
                "repository_name": "ceph_aeon",
                "directory_type": "quality-control",
                "directory_path": "aeon/data/qc/OCTAGON01/conf1",
            },
        ],
        skip_duplicates=True,
    )


def main():
    """Main function to create a new experiment."""
    create_new_experiment()


if __name__ == "__main__":
    main()
