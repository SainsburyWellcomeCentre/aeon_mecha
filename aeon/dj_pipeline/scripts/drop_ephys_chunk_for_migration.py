"""One-shot helper: drop EphysChunk and dependents to enable migration to link-only SyncModel.

The ONIX IMU pipeline branch refactors ``EphysChunk.SyncModel`` Part from
``(<attach> + ONIX bounds)`` to a pure FK link ``-> EphysSyncModel``. DataJoint
cannot migrate Part definitions in place. Operators must drop and re-ingest:

1. Stop ingestion processes.
2. Run this script with ``--yes`` to drop ``EphysBlockInfo``, ``EphysBlock``,
   ``EphysChunk``, and (if present) ``EphysSyncModel``.
3. Pull the new branch.
4. Re-run ingestion (``EphysSyncModel.ingest`` → ``EphysChunk.ingest_chunks``).

Model bytes for existing chunks are reproducible from the on-disk HarpSync CSVs;
no data is lost — just CPU time to refit regressions.

This is a DEV/STAGING helper. Production migration is out of scope.

Usage::

    uv run python -m aeon.dj_pipeline.scripts.drop_ephys_chunk_for_migration  # dry-run
    uv run python -m aeon.dj_pipeline.scripts.drop_ephys_chunk_for_migration --yes  # apply
"""

import argparse

from aeon.dj_pipeline import ephys


def main():
    """Parse CLI arguments and drop EphysChunk and dependents (dry-run by default)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the drop (otherwise dry-run, no DB changes).",
    )
    args = parser.parse_args()

    # Drop order matters: dependents first, masters last
    targets = [
        ephys.EphysBlockInfo,
        ephys.EphysBlock,
        ephys.EphysChunk,
    ]
    if hasattr(ephys, "EphysSyncModel"):
        targets.append(ephys.EphysSyncModel)

    for table in targets:
        n = len(table())
        print(f"Would drop {table.__name__}: {n} rows")
        if args.yes:
            table.drop_quick()
            print("  → dropped")


if __name__ == "__main__":
    main()
