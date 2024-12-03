# Pipeline Deployment (On-Premises)

This page describes the processes and required resources to deploy the Project Aeon data pipeline on-premises.

## Prerequisites

On the most basic level, in order to deploy and operate a DataJoint pipeline, you will need:

1. A MySQL database server (version 8.0) with configured to be DataJoint compatible
   - see [here](https://github.com/datajoint/mysql-docker/blob/master/config/my.cnf) for configuration of the MySQL server to be DataJoint compatible
2. If you want to use a preconfigured Docker container ([install Docker](https://docs.docker.com/engine/install/)), run the following command:
      ```bash
         docker run -d \
           --name db \
           -p 3306:3306 \
           -e MYSQL_ROOT_PASSWORD=simple \
           -v ./mysql/data:/var/lib/mysql \
           datajoint/mysql:8.0 \
           mysqld --default-authentication-plugin=mysql_native_password
      ```
   
    A new MySQL server will be launched in a Docker Container with the following credentials: 
    - host: `localhost`
    - username: `root`
    - password: `simple`
    
   To stop the container, run the following command:
   
    ```bash
       docker stop db
    ```
   
3. a GitHub repository with the [codebase](https://github.com/SainsburyWellcomeCentre/aeon_mecha) of the DataJoint pipeline
   - this repository is the codebase, no additional modifications are needed to deploy this codebase locally
4. file storage
   - the pipeline requires a location to access/store the data files (this can be a local directory or mounted network storage)
5. compute
   - you need some form of a compute environment with the right software installed to run the pipeline (this could be a laptop, local work station or an HPC cluster)

## Download the data

The released data for Project Aeon can be downloaded from the data repository [here](https://zenodo.org/records/13881885)


## Pipeline Installation & Configuration

### Installation Instructions

In order to run the pipeline, follow the instruction to install this codebase in the "Local set-up" section

### Configuration Instructions

DataJoint requires a configuration file named `dj_local_conf.json`. This file should be located in the root directory of the codebase.

1. Generate the `dj_local_conf.json` file:
   - Make a copy of the `sample_dj_local_conf.json` file with the exact name `dj_local_conf.json`.
   - Update the file with your database credentials (username, password, and database host).
   - Ensure the file is kept secure and not leaked.
2. In the `custom` section, specify the `database.prefix` - you can keep the default `aeon_`.
3. In the `custom` section, update the value of `ceph_aeon` (under `repository_config`) to the root directory of the downloaded data.
For example, if you download the data to `D:/data/project-aeon/aeon/data/raw/AEON3/...`, then update `ceph_aeon` to `D:/data/project-aeon/aeon/data`.


## Data Ingestion & Processing

Now that the pipeline is installed and configured, you can start ingesting and processing the downloaded data.

Follow the instructions in the [Data Ingestion & Processing](./notebooks/Data_Ingestion_and_Processing.ipynb) guide to learn how to ingest and process the downloaded data.
