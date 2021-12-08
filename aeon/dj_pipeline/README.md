# DataJoint Pipeline for Project Aeon

This pipeline models the data organization and data flow custom-built for Project Aeon. You can find Aeon acquisition system here: [aeon_aquisition](https://github.com/SainsburyWellcomeCentre/aeon_acquisition)


## Pipeline architecture

Figure below presents an abbreviated view of the pipeline, showing the core tables
and the overall dataflow

![diagram](./docs/diagram.svg)

From the diagram above, we can see that the pipeline is organized in layers of
tables, going top down, from `lookup`-tier (in gray) and `manual`-tier (in green) tables 
to `imported`-tier (in purple) and `computed`-tier (in red) tables.

Such is also the way the data flows through the pipeline, by a combination of ingestion and 
computation routines.

## Core tables

1. `Experiment` - the `aquisition.Experiment` table stores meta information about the experiments
done in Project Aeon, with secondary information such as the lab/room the experiment is carried out, 
which animals participating, the directory storing the raw data, etc.

2. `Chunk` - the raw data are acquired by Bonsai and stored as 
a collection of files every one hour - we call this one-hour a time chunk. 
The `aquisition.Chunk` table records all time chunks and their associated raw data files for 
any particular experiment (in the above `aquisition.Experiment` table) 

3. `ExperimentCamera` - the cameras and associated specifications used for this experiment - 
e.g. camera serial number, frame rate, location, time of installation and removal, etc.

4. `ExperimentFoodPatch` - the food-patches and associated specifications used for this experiment - 
e.g. patch serial number, sampling rate of the wheel, location, time of installation and removal, etc.

5. `FoodPatchEvent` - all events (e.g. pellet triggered, pellet delivered, etc.) 
from a particular `ExperimentFoodPatch`

6. `Session` - a session is defined, for a given animal, as the time period where 
the animal enters the arena until it exits (typically 4 to 5 hours long)

7. `TimeSlice` - data for each session are stored in smaller time bins called time slices. 
Currently, a time slice is defined to be 10-minute long. Storing data in smaller time slices allows for 
more efficient searches, queries and fetches from the database.

8. `SubjectPosition` - position data (x, y, speed, area) of the subject in the time slices for 
any particular session.

9. `SessionSummary` - a table for computation and storing some summary statistics on a 
per-session level - i.e. total pellet delivered, total distance the animal travelled, total 
distance the wheel travelled (or per food-patch), etc.

10. `SessionTimeDistribution` - a table for computation and storing where the animal is at, 
for each timepoint, e.g. in the nest, in corridor, in arena, in each of the food patches. 
This can be used to produce the ethogram plot.


The diagram below shows the same architecture, with some figures 
to demonstrate which type of data is stored where.

![datajoint_pipeline](./docs/datajoint_pipeline.svg)


## Operating the pipeline - how the auto ingestion/processing work?

Some meta information about the experiment is entered - e.g. experiment name, participating 
animals, cameras, food patches setup, etc.
+ These information are either entered by hand, or parsed and inserted from configuration 
    yaml files.
+ For experiment 0.1 these info can be inserted by running 
the [exp01_insert_meta script](./ingest/exp01_insert_meta.py) (just need to do this once)

Tables in DataJoint are written with a `make()` function - 
instruction to generate and insert new records to itself, based on data from upstream tables. 
Triggering the auto ingestion and processing/computation routine is essentially 
calling the `.populate()` method for all relevant tables.

These routines are prepared in this [auto-processing script](./ingest/process.py). 
Essentially, turning on the auto-processing routine amounts to running the 
following 3 commands (in different processing threads)


    aeon_ingest high
    
    aeon_ingest mid
    
    aeon_ingest low
