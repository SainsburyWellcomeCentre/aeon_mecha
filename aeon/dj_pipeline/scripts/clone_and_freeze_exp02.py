"""Jan 2024. Cloning and archiving schemas and data for experiment 0.2.

The pipeline code associated with this archived data pipeline is here
https://github.com/SainsburyWellcomeCentre/aeon_mecha/releases/tag/dj_exp02_stable
"""

import inspect
import os

import datajoint as dj
from datajoint_utilities.dj_data_copy import db_migration
from datajoint_utilities.dj_data_copy.pipeline_cloning import ClonedPipeline

logger = dj.logger
os.environ["DJ_SUPPORT_FILEPATH_MANAGEMENT"] = "TRUE"

source_db_prefix = "aeon_"
target_db_prefix = "aeon_archived_exp02_"

schema_name_mapper = {
    source_db_prefix + schema_name: target_db_prefix + schema_name
    for schema_name in (
        "lab",
        "subject",
        "acquisition",
        "tracking",
        "qc",
        "analysis",
        "report",
    )
}

restriction = [{"experiment_name": "exp0.2-r0"}, {"experiment_name": "social0-r1"}]

table_block_list = {}

batch_size = None


def clone_pipeline():
    """Clone the pipeline for experiment 0.2."""
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


def validate():
    """Validates schemas migration.

    1. for the provided list of schema names - validate all schemas have been migrated
    2. for each schema - validate all tables have been migrated
    3. for each table, validate all entries have been migrated
    """
    missing_schemas = []
    missing_tables = {}
    missing_entries = {}

    for orig_schema_name, cloned_schema_name in schema_name_mapper.items():
        logger.info(f"Validate schema: {orig_schema_name}")
        source_vm = dj.create_virtual_module(orig_schema_name, orig_schema_name)

        try:
            target_vm = dj.create_virtual_module(cloned_schema_name, cloned_schema_name)
        except dj.errors.DataJointError:
            missing_schemas.append(orig_schema_name)
            continue

        missing_tables[orig_schema_name] = []
        missing_entries[orig_schema_name] = {}

        for attr in dir(source_vm):
            obj = getattr(source_vm, attr)
            if isinstance(obj, dj.user_tables.UserTable) or (
                inspect.isclass(obj) and issubclass(obj, dj.user_tables.UserTable)
            ):
                source_tbl = obj
                try:
                    target_tbl = getattr(target_vm, attr)
                except AttributeError:
                    missing_tables[orig_schema_name].append(source_tbl.table_name)
                    continue
                logger.info(f"\tValidate entry count: {source_tbl.__name__}")
                source_entry_count = len(source_tbl())
                target_entry_count = len(target_tbl())
                missing_entries[orig_schema_name][source_tbl.__name__] = {
                    "entry_count_diff": source_entry_count - target_entry_count,
                    "db_size_diff": source_tbl().size_on_disk - target_tbl().size_on_disk,
                }

    return {
        "missing_schemas": missing_schemas,
        "missing_tables": missing_tables,
        "missing_entries": missing_entries,
    }


if __name__ == "__main__":
    print("This is not meant to be run as a script (yet)")
