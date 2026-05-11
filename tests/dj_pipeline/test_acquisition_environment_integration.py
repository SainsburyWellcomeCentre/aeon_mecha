"""Integration tests for Environment-prefixed stream tables in acquisition.py.

These run against the testcontainers MySQL fixture (`pipeline_integration`),
which activates schemas with the test_aeon_ prefix.
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("table_name", ["EnvironmentState", "MessageLog", "LightEvents"])
def test_environment_table_schema(pipeline_integration, table_name):
    """Confirm Environment table exists and has correct PK and nullable stream_df."""
    from aeon.dj_pipeline import acquisition

    table = getattr(acquisition, table_name)
    attrs = table.heading.attributes
    assert set(table.primary_key) == {"experiment_name", "chunk_start"}
    assert "sample_count" in attrs
    assert "timestamps" in attrs
    assert attrs["stream_df"].nullable


class TestEnvironmentStreamPopulate:
    """End-to-end populate against the foragingABC golden chunk.

    Skips when:
      - the golden dataset isn't on disk (require_golden_data fixture skips), or
      - the Pydantic Environment device doesn't expose @data_reader methods yet
        (upstream PR not merged).
    """

    @pytest.fixture(scope="class")
    def populated_chunks(self, full_pipeline, test_experiment, test_epochs, require_golden_data):
        """Drive Chunk + EpochConfig ingestion so the new tables have key_source rows."""
        acquisition = full_pipeline["acquisition"]
        cfg_name = test_experiment["experiment_name"]

        # Make sure the upstream Environment device exposes the @data_readers
        # we depend on; skip cleanly otherwise.
        from aeon.dj_pipeline.utils.load_metadata import (
            get_experiment_pydantic,
        )

        exp_class = get_experiment_pydantic(
            (acquisition.Experiment.DevicesSchema & {"experiment_name": cfg_name}).fetch1(
                "devices_schema_name"
            )
        )
        rig_cls = exp_class.model_fields["rig"].annotation
        env_field = rig_cls.model_fields.get("environment")
        if env_field is None:
            pytest.skip("Rig has no `environment` field — upstream PR not merged")
        env_cls = env_field.annotation
        if not all(hasattr(env_cls, name) for name in ("environment_state", "message_log")):
            pytest.skip(
                "Upstream Environment device missing @data_reader methods "
                "environment_state/message_log — pull aeon_api/foragingABC main first"
            )

        # Ingest chunks for the experiment, then run EpochConfig so the
        # rig metadata is on disk for get_stream_reader_for_epoch().
        acquisition.Chunk.ingest_chunks(cfg_name)
        acquisition.EpochConfig.populate({"experiment_name": cfg_name})

        chunks = (acquisition.Chunk & {"experiment_name": cfg_name}).fetch("KEY")
        if not chunks:
            pytest.skip("No chunks ingested from golden dataset")
        return chunks

    @pytest.mark.parametrize("table_name", ["EnvironmentState", "MessageLog", "LightEvents"])
    def test_populates(self, full_pipeline, populated_chunks, table_name):
        """Smoke test: populate the table for one chunk, confirm row exists with expected keys.

        LightEvents may not exist in all datasets but golden dataset (ForagingABC) HAS light_events.
        Sample_count may be 0 for chunks where there are no events, but the row must
        exist and stream_df must NOT be null (reader was resolved).
        """
        acquisition = full_pipeline["acquisition"]
        table = getattr(acquisition, table_name)
        table.populate(populated_chunks[:1], display_progress=False)
        row = (table & populated_chunks[0]).fetch1()
        assert row["sample_count"] >= 0
        assert isinstance(row["timestamps"], dict)
        assert row["stream_df"] is not None
        assert row["stream_df"]["stream_type"] == table_name
        assert row["stream_df"]["device_name"] == "Environment"
