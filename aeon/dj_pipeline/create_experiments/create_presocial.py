from aeon.dj_pipeline import acquisition, lab, subject

experiment_type = "presocial0.1"
experiment_names = ["presocial0.1-a2", "presocial0.1-a3", "presocial0.1-a4"]
location = "4th floor"
computers = ["AEON2", "AEON3", "AEON4"]

def create_new_experiment():

    lab.Location.insert1({"lab": "SWC", "location": location}, skip_duplicates=True)

    acquisition.ExperimentType.insert1(
        {"experiment_type": experiment_type}, skip_duplicates=True
    )

    acquisition.Experiment.insert(
        [{
            "experiment_name": experiment_name,
            "experiment_start_time": "2023-02-25 00:00:00",
            "experiment_description": "presocial experiment 0.1",
            "arena_name": "circle-2m",
            "lab": "SWC",
            "location": location,
            "experiment_type": experiment_type
        } for experiment_name in experiment_names],
        skip_duplicates=True,
    )

    acquisition.Experiment.Subject.insert(
        [
            {"experiment_name": experiment_name, "subject": s}
            for experiment_name in experiment_names
            for s in subject.Subject.fetch("subject")
        ],
        skip_duplicates=True,
    )

    acquisition.Experiment.Directory.insert(
        [
            {
                "experiment_name": experiment_name,
                "repository_name": "ceph_aeon",
                "directory_type": "raw",
                "directory_path": f"aeon/data/raw/{computer}/{experiment_type}"
            } for experiment_name, computer in zip(experiment_names, computers)
        ],
        skip_duplicates=True,
    )


def main():
    create_new_experiment()


if __name__ == "__main__":
    main()
