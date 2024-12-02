# DataJoint Pipeline for Project Aeon

This DataJoint pipeline models the data organization and data flow tailored to the project's Aeon requirements. You can access the Aeon acquisition system here: [aeon_aquisition](https://github.com/SainsburyWellcomeCentre/aeon_acquisition)


## Pipeline Architecture

The following diagrams provide a high-level overview of the pipeline's components and processes:

The diagram below illustrates the structure of the **acquisition-related tasks within the pipeline**, focusing on the most relevant subset of tables.

![datajoint_overview_acquisition_diagram](./docs/datajoint_overview_acquisition_related_diagram.svg)

The diagram below represents the **data stream flow within the pipeline**, highlighting the subset of tables critical to understanding the process.

![datajoint_overview_data_stream_diagram](./docs/datajoint_overview_data_stream_diagram.svg)

The diagram below shows the **analysis portion of the pipeline**.

![datajoint_analysis_pipeline](./docs/datajoint_analysis_diagram.svg)


The pipeline is structured into hierarchical layers of tables, which are depicted in the diagrams above. These layers include:

+ `lookup`-tier tables (gray): Define static reference information
+ `manual`-tier tables (green): Contain user-inputted data
+ `imported`-tier tables (purple): Store data ingested from external sources
+ `computed`-tier tables (red): Represent results of automated computations

Data flows through the pipeline in a top-down manner, driven by a combination of ingestion and computation routines. This layered organization facilitates efficient data processing and modular analysis.

## Core tables

#### Experiment and data acquisition

+ `Experiment` - the `aquisition.Experiment` table stores meta information about the experiments
done in Project Aeon, with secondary information such as the lab/room the experiment is carried out,
which animals participating, the directory storing the raw data, etc.

+ `Epoch` - A recording period reflecting on/off of the hardware acquisition system.
The `aquisition.Epoch` table records all acquisition epochs and their associated configuration for
any particular experiment (in the above `aquisition.Experiment` table).

+ `Chunk` - the raw data are acquired by Bonsai and stored as
a collection of files every one hour - we call this one-hour a time chunk.
The `aquisition.Chunk` table records all time chunks and their associated raw data files for
any particular experiment (in the above `aquisition.Experiment` table). A chunk must belong to one epoch.

#### Position data

+ `qc.CameraQC` - quality control procedure applied to each `ExperimentCamera` (e.g. missing frame, etc.)

+ `tracking.SLEAPTracking` - position tracking for object(s), from a particular `VideoSource` per chunk

#### Standard analyses

+ `Visit` - a `Visit` is defined as a ***period of time*** during which a particular ***animal*** remains at a specific ***place***.

+ `BlockAnalysis` - higher-level aggregation of events and metrics that occur within a defined block of time during an experiment. It integrates data from multiple subjects, positions, and interactions to enable further analysis of behavior and environmental interactions.

+ `BlockSubjectAnalysis` - A detailed analysis for each subject in a given block, partitioned into the following components:
    - `Patch`: tracks the interactions of each subject with specific patches (areas of interests)
    - `Preference`: measures a subject's preference for specific patches using various analyses, including cumulative preferences based on time spent and distance traveled in relation to each patch.

#### Data stream

