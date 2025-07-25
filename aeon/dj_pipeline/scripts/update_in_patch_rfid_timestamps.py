"""Script to update in_patch_rfid_timestamps for all blocks that are missing it."""

import datajoint as dj

from aeon.dj_pipeline import acquisition, fetch_stream, streams, subject
from aeon.dj_pipeline.analysis.block_analysis import Block, BlockAnalysis, BlockSubjectAnalysis

logger = dj.logger


def update_in_patch_rfid_timestamps(block_key):
    """Update in_patch_rfid_timestamps for a given block_key.

    Args:
        block_key (dict): block key
    """
    logger.info(f"Updating in_patch_rfid_timestamps for {block_key}")

    block_key = (Block & block_key).fetch1("KEY")
    block_start, block_end = (Block & block_key).fetch1("block_start", "block_end")
    chunk_restriction = acquisition.create_chunk_restriction(
        block_key["experiment_name"], block_start, block_end
    )
    patch_names = (BlockAnalysis.Patch & block_key).fetch("patch_name")
    subject_names = (BlockAnalysis.Subject & block_key).fetch("subject_name")

    rfid2subj_map = {
        int(lab_id): subj_name
        for subj_name, lab_id in zip(
            *(subject.SubjectDetail.proj("lab_id")
              & f"subject in {tuple(list(subject_names) + [''])}").fetch(
                "subject", "lab_id"
            ),
            strict=False,
        )
    }

    entries = []
    for patch_name in patch_names:
        # In patch time from RFID
        rfid_query = (
            streams.RfidReader.proj(rfid_name="REPLACE(rfid_reader_name, 'Rfid', '')")
            * streams.RfidReaderRfidEvents
            & block_key
            & {"rfid_name": patch_name}
            & chunk_restriction
        )
        rfid_df = fetch_stream(rfid_query)[block_start:block_end]
        rfid_df["subject"] = rfid_df.rfid.map(rfid2subj_map)

        for subject_name in subject_names:
            k = {
                **block_key,
                "patch_name": patch_name,
                "subject_name": subject_name,
            }
            if BlockSubjectAnalysis.Patch & k:
                entries.append(
                    {**k, "in_patch_rfid_timestamps": rfid_df[rfid_df.subject == subject_name].index.values}
                )

    with BlockSubjectAnalysis.connection.transaction:
        for e in entries:
            BlockSubjectAnalysis.Patch.update1(e)
        logger.info(f"\tUpdated {len(entries)} entries.")


def main():
    """Update in_patch_rfid_timestamps for all blocks that are missing it."""
    block_keys = (
            BlockSubjectAnalysis
            & (BlockSubjectAnalysis.Patch & "in_patch_rfid_timestamps IS NULL")
    ).fetch("KEY")
    for block_key in block_keys:
        try:
            update_in_patch_rfid_timestamps(block_key)
        except Exception as e:
            logger.error(f"Error updating {block_key}: {e}")
