from aeon.dj_pipeline import acquisition, lab, subject

experiment_name = "presocial0.1-a2"  # AEON2 acquisition computer
location = "4th floor"


def create_new_experiment():

    lab.Location.insert1({"lab": "SWC", "location": location}, skip_duplicates=True)

    acquisition.Experiment.insert1(
        {
            "experiment_name": experiment_name,
            "experiment_start_time": "2023-02-25 18:03:49",
            "experiment_description": "presocial experiment 0.1 in aeon2",
            "arena_name": "circle-2m",
            "lab": "SWC",
            "location": location,
            "experiment_type": "presocial",
        },
        skip_duplicates=True,
    )

    acquisition.Experiment.Subject.insert(
        [
            {"experiment_name": experiment_name, "subject": s["subject"]}
            for s in subject.Subject.fetch("subject")
        ],
        skip_duplicates=True,
    )

    acquisition.Experiment.Directory.insert1(
        {
            "experiment_name": experiment_name,
            "repository_name": "ceph_aeon",
            "directory_type": "raw",
            "directory_path": "aeon/data/raw/AEON2/presocial0.1",
        },
        skip_duplicates=True,
    )


def main():
    create_new_experiment()


if __name__ == "__main__":
    main()
