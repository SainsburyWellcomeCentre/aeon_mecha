import datajoint as dj
from datetime import datetime

from aeon.dj_pipeline import acquisition, streams
from aeon.dj_pipeline.analysis import block_analysis

aeon_schemas = acquisition.aeon_schemas
logger = acquisition.logger

exp_key = {"experiment_name": "social0.2-aeon4"}


def find_orphaned_ingested_epochs(exp_key, delete_invalid_epochs=False):
    """
    Find ingested epochs that are no longer valid
    This is due to the raw epoch/chunk files/directories being deleted for whatever reason
    (e.g. bad data, testing, etc.)
    """
    raw_dir = acquisition.Experiment.get_data_directory(exp_key, "raw")
    epoch_dirs = [d.name for d in raw_dir.glob("*T*") if d.is_dir()]

    epoch_query = acquisition.Epoch.join(acquisition.EpochEnd, left=True) & exp_key

    valid_epochs = epoch_query & f"epoch_dir in {tuple(epoch_dirs)}"
    invalid_epochs = epoch_query - f"epoch_dir in {tuple(epoch_dirs)}"

    logger.info(f"Valid Epochs: {len(valid_epochs)} | Invalid Epochs: {len(invalid_epochs)}")

    if not invalid_epochs or not delete_invalid_epochs:
        return

    # delete blocks
    # delete streams device installations
    # delete epochs
    invalid_blocks = []
    for key in invalid_epochs.fetch("KEY"):
        epoch_start, epoch_end = (invalid_epochs & key).fetch1("epoch_start", "epoch_end")
        invalid_blocks.extend(
            (block_analysis.Block
             & exp_key
             & f"block_start BETWEEN '{epoch_start}' AND '{epoch_end}'").fetch("KEY"))

    # devices
    invalid_devices_query = acquisition.EpochConfig.DeviceType & invalid_epochs
    if invalid_devices_query:
        logger.warning("Invalid devices found - please run the rest manually to confirm deletion")
        logger.warning(invalid_devices_query)
        return

    device_types = set(invalid_devices_query.fetch("device_type"))
    device_table_invalid_query = []
    for device_type in device_types:
        device_table = getattr(streams, device_type)
        install_time_attr_name = next(n for n in device_table.primary_key if n.endswith("_install_time"))
        invalid_device_query = device_table & invalid_epochs.proj(**{install_time_attr_name: "epoch_start"})
        logger.debug(invalid_device_query)
        device_table_invalid_query.append((device_table, invalid_device_query))

    # delete
    dj.conn().start_transaction()

    with dj.config(safemode=False):
        (block_analysis.Block & invalid_blocks).delete()
        for device_table, invalid_query in device_table_invalid_query:
            (device_table & invalid_query.fetch("KEY")).delete()
        (acquisition.Epoch & invalid_epochs.fetch("KEY")).delete()

    if dj.utils.user_choice("Commit deletes?", default="no") == "yes":
        dj.conn().commit_transaction()
        logger.info("Deletes committed.")
    else:
        dj.conn().cancel_transaction()
        logger.info("Deletes cancelled")
