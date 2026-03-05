"""March 2022. Cloning and archiving schemas and data for experiment 0.1."""

import os

import datajoint as dj
from datajoint_utilities.dj_data_copy import db_migration
from datajoint_utilities.dj_data_copy.pipeline_cloning import ClonedPipeline

os.environ["DJ_SUPPORT_FILEPATH_MANAGEMENT"] = "TRUE"

source_db_prefix = "aeon_"
target_db_prefix = "aeon_archived_exp01_"

schema_name_mapper = {
    source_db_prefix + schema_name: target_db_prefix + schema_name
    for schema_name in ("lab", "subject", "acquisition", "tracking", "qc", "report", "analysis")
}

restriction = {"experiment_name": "exp0.1-r0"}

table_block_list = {}

batch_size = None


def clone_pipeline():
    """Clone the pipeline for experiment 0.1."""
    diagram = None
    for orig_schema_name in schema_name_mapper:
        virtual_module = dj.create_virtual_module(orig_schema_name, orig_schema_name)
        if diagram is None:
            diagram = dj.Diagram(virtual_module)
        else:
            diagram += dj.Diagram(virtual_module)

    cloned_pipeline = ClonedPipeline(diagram, schema_name_mapper, verbose=True)
    cloned_pipeline.instantiate_pipeline(prompt=False)


def data_copy(restriction, table_block_list, batch_size=None):
    """Migrate schema."""
    for orig_schema_name, cloned_schema_name in schema_name_mapper.items():
        orig_schema = dj.create_virtual_module(orig_schema_name, orig_schema_name)
        cloned_schema = dj.create_virtual_module(cloned_schema_name, cloned_schema_name)

        db_migration.migrate_schema(
            orig_schema,
            cloned_schema,
            restriction=restriction,
            table_block_list=table_block_list.get(cloned_schema_name, []),
            allow_missing_destination_tables=True,
            force_fetch=False,
            batch_size=batch_size,
        )


if __name__ == "__main__":
    print("This is not meant to be run as a script (yet)")
