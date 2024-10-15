#----                     DO NOT MODIFY                ----
#---- THIS FILE IS AUTO-GENERATED BY `streams_maker.py` ----

import re
import datajoint as dj
import pandas as pd
from uuid import UUID

import aeon
from aeon.dj_pipeline import acquisition, get_schema_name
from aeon.io import api as io_api

aeon_schemas = acquisition.aeon_schemas

schema = dj.Schema(get_schema_name("streams"))


@schema
class StreamType(dj.Lookup):
    """Catalog of all steam types for the different device types used across Project Aeon. One StreamType corresponds to one reader class in `aeon.io.reader`. The combination of `stream_reader` and `stream_reader_kwargs` should fully specify the data loading routine for a particular device, using the `aeon.io.utils`."""

    definition = """  # Catalog of all stream types used across Project Aeon
    stream_type          : varchar(20)
    ---
    stream_reader        : varchar(256)     # name of the reader class found in `aeon_mecha` package (e.g. aeon.io.reader.Video)
    stream_reader_kwargs : longblob  # keyword arguments to instantiate the reader class
    stream_description='': varchar(256)
    stream_hash          : uuid    # hash of dict(stream_reader_kwargs, stream_reader=stream_reader)
    unique index (stream_hash)
    """


@schema
class DeviceType(dj.Lookup):
    """Catalog of all device types used across Project Aeon."""

    definition = """  # Catalog of all device types used across Project Aeon
    device_type:             varchar(36)
    ---
    device_description='':   varchar(256)
    """

    class Stream(dj.Part):
        definition = """  # Data stream(s) associated with a particular device type
        -> master
        -> StreamType
        """


@schema
class Device(dj.Lookup):
    definition = """  # Physical devices, of a particular type, identified by unique serial number
    device_serial_number: varchar(12)
    ---
    -> DeviceType
    """


@schema
class RfidReader(dj.Manual):
        definition = f"""
        # rfid_reader placement and operation for a particular time period, at a certain location, for a given experiment (auto-generated with aeon_mecha-unknown)
        -> acquisition.Experiment
        -> Device
        rfid_reader_install_time  : datetime(6)   # time of the rfid_reader placed and started operation at this position
        ---
        rfid_reader_name          : varchar(36)
        """

        class Attribute(dj.Part):
            definition = """  # metadata/attributes (e.g. FPS, config, calibration, etc.) associated with this experimental device
            -> master
            attribute_name          : varchar(32)
            ---
            attribute_value=null    : longblob
            """

        class RemovalTime(dj.Part):
            definition = f"""
            -> master
            ---
            rfid_reader_removal_time: datetime(6)  # time of the rfid_reader being removed
            """


@schema
class SpinnakerVideoSource(dj.Manual):
        definition = f"""
        # spinnaker_video_source placement and operation for a particular time period, at a certain location, for a given experiment (auto-generated with aeon_mecha-unknown)
        -> acquisition.Experiment
        -> Device
        spinnaker_video_source_install_time  : datetime(6)   # time of the spinnaker_video_source placed and started operation at this position
        ---
        spinnaker_video_source_name          : varchar(36)
        """

        class Attribute(dj.Part):
            definition = """  # metadata/attributes (e.g. FPS, config, calibration, etc.) associated with this experimental device
            -> master
            attribute_name          : varchar(32)
            ---
            attribute_value=null    : longblob
            """

        class RemovalTime(dj.Part):
            definition = f"""
            -> master
            ---
            spinnaker_video_source_removal_time: datetime(6)  # time of the spinnaker_video_source being removed
            """


@schema
class UndergroundFeeder(dj.Manual):
        definition = f"""
        # underground_feeder placement and operation for a particular time period, at a certain location, for a given experiment (auto-generated with aeon_mecha-unknown)
        -> acquisition.Experiment
        -> Device
        underground_feeder_install_time  : datetime(6)   # time of the underground_feeder placed and started operation at this position
        ---
        underground_feeder_name          : varchar(36)
        """

        class Attribute(dj.Part):
            definition = """  # metadata/attributes (e.g. FPS, config, calibration, etc.) associated with this experimental device
            -> master
            attribute_name          : varchar(32)
            ---
            attribute_value=null    : longblob
            """

        class RemovalTime(dj.Part):
            definition = f"""
            -> master
            ---
            underground_feeder_removal_time: datetime(6)  # time of the underground_feeder being removed
            """


@schema
class WeightScale(dj.Manual):
        definition = f"""
        # weight_scale placement and operation for a particular time period, at a certain location, for a given experiment (auto-generated with aeon_mecha-unknown)
        -> acquisition.Experiment
        -> Device
        weight_scale_install_time  : datetime(6)   # time of the weight_scale placed and started operation at this position
        ---
        weight_scale_name          : varchar(36)
        """

        class Attribute(dj.Part):
            definition = """  # metadata/attributes (e.g. FPS, config, calibration, etc.) associated with this experimental device
            -> master
            attribute_name          : varchar(32)
            ---
            attribute_value=null    : longblob
            """

        class RemovalTime(dj.Part):
            definition = f"""
            -> master
            ---
            weight_scale_removal_time: datetime(6)  # time of the weight_scale being removed
            """


@schema
class RfidReaderRfidEvents(dj.Imported):
        definition = """  # Raw per-chunk RfidEvents data stream from RfidReader (auto-generated with aeon_mecha-unknown)
    -> RfidReader
    -> acquisition.Chunk
    ---
    sample_count: int      # number of data points acquired from this stream for a given chunk
    timestamps: longblob   # (datetime) timestamps of RfidEvents data
    rfid: longblob
    """

        @property
        def key_source(self):
            f"""
            Only the combination of Chunk and RfidReader with overlapping time
            +  Chunk(s) that started after RfidReader install time and ended before RfidReader remove time
            +  Chunk(s) that started after RfidReader install time for RfidReader that are not yet removed
            """
            return (
                acquisition.Chunk * RfidReader.join(RfidReader.RemovalTime, left=True)
                & 'chunk_start >= rfid_reader_install_time'
                & 'chunk_start < IFNULL(rfid_reader_removal_time, "2200-01-01")'
            )

        def make(self, key):
            chunk_start, chunk_end = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")

            data_dirs = acquisition.Experiment.get_data_directories(key)

            device_name = (RfidReader & key).fetch1('rfid_reader_name')

            devices_schema = getattr(
                aeon_schemas,
                (acquisition.Experiment.DevicesSchema & {"experiment_name": key["experiment_name"]}).fetch1(
                    "devices_schema_name"
                ),
            )
            stream_reader = getattr(getattr(devices_schema, device_name), "RfidEvents")

            stream_data = io_api.load(
                root=data_dirs,
                reader=stream_reader,
                start=pd.Timestamp(chunk_start),
                end=pd.Timestamp(chunk_end),
            )

            self.insert1(
                {
                    **key,
                    "sample_count": len(stream_data),
                    "timestamps": stream_data.index.values,
                    **{
                        re.sub(r"\([^)]*\)", "", c): stream_data[c].values
                        for c in stream_reader.columns
                        if not c.startswith("_")
                    },
                },
                ignore_extra_fields=True,
            )


@schema
class SpinnakerVideoSourceVideo(dj.Imported):
        definition = """  # Raw per-chunk Video data stream from SpinnakerVideoSource (auto-generated with aeon_mecha-unknown)
    -> SpinnakerVideoSource
    -> acquisition.Chunk
    ---
    sample_count: int      # number of data points acquired from this stream for a given chunk
    timestamps: longblob   # (datetime) timestamps of Video data
    hw_counter: longblob
    hw_timestamp: longblob
    """

        @property
        def key_source(self):
            f"""
            Only the combination of Chunk and SpinnakerVideoSource with overlapping time
            +  Chunk(s) that started after SpinnakerVideoSource install time and ended before SpinnakerVideoSource remove time
            +  Chunk(s) that started after SpinnakerVideoSource install time for SpinnakerVideoSource that are not yet removed
            """
            return (
                acquisition.Chunk * SpinnakerVideoSource.join(SpinnakerVideoSource.RemovalTime, left=True)
                & 'chunk_start >= spinnaker_video_source_install_time'
                & 'chunk_start < IFNULL(spinnaker_video_source_removal_time, "2200-01-01")'
            )

        def make(self, key):
            chunk_start, chunk_end = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")

            data_dirs = acquisition.Experiment.get_data_directories(key)

            device_name = (SpinnakerVideoSource & key).fetch1('spinnaker_video_source_name')

            devices_schema = getattr(
                aeon_schemas,
                (acquisition.Experiment.DevicesSchema & {"experiment_name": key["experiment_name"]}).fetch1(
                    "devices_schema_name"
                ),
            )
            stream_reader = getattr(getattr(devices_schema, device_name), "Video")

            stream_data = io_api.load(
                root=data_dirs,
                reader=stream_reader,
                start=pd.Timestamp(chunk_start),
                end=pd.Timestamp(chunk_end),
            )

            self.insert1(
                {
                    **key,
                    "sample_count": len(stream_data),
                    "timestamps": stream_data.index.values,
                    **{
                        re.sub(r"\([^)]*\)", "", c): stream_data[c].values
                        for c in stream_reader.columns
                        if not c.startswith("_")
                    },
                },
                ignore_extra_fields=True,
            )


@schema
class UndergroundFeederBeamBreak(dj.Imported):
        definition = """  # Raw per-chunk BeamBreak data stream from UndergroundFeeder (auto-generated with aeon_mecha-unknown)
    -> UndergroundFeeder
    -> acquisition.Chunk
    ---
    sample_count: int      # number of data points acquired from this stream for a given chunk
    timestamps: longblob   # (datetime) timestamps of BeamBreak data
    event: longblob
    """

        @property
        def key_source(self):
            f"""
            Only the combination of Chunk and UndergroundFeeder with overlapping time
            +  Chunk(s) that started after UndergroundFeeder install time and ended before UndergroundFeeder remove time
            +  Chunk(s) that started after UndergroundFeeder install time for UndergroundFeeder that are not yet removed
            """
            return (
                acquisition.Chunk * UndergroundFeeder.join(UndergroundFeeder.RemovalTime, left=True)
                & 'chunk_start >= underground_feeder_install_time'
                & 'chunk_start < IFNULL(underground_feeder_removal_time, "2200-01-01")'
            )

        def make(self, key):
            chunk_start, chunk_end = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")

            data_dirs = acquisition.Experiment.get_data_directories(key)

            device_name = (UndergroundFeeder & key).fetch1('underground_feeder_name')

            devices_schema = getattr(
                aeon_schemas,
                (acquisition.Experiment.DevicesSchema & {"experiment_name": key["experiment_name"]}).fetch1(
                    "devices_schema_name"
                ),
            )
            stream_reader = getattr(getattr(devices_schema, device_name), "BeamBreak")

            stream_data = io_api.load(
                root=data_dirs,
                reader=stream_reader,
                start=pd.Timestamp(chunk_start),
                end=pd.Timestamp(chunk_end),
            )

            self.insert1(
                {
                    **key,
                    "sample_count": len(stream_data),
                    "timestamps": stream_data.index.values,
                    **{
                        re.sub(r"\([^)]*\)", "", c): stream_data[c].values
                        for c in stream_reader.columns
                        if not c.startswith("_")
                    },
                },
                ignore_extra_fields=True,
            )


@schema
class UndergroundFeederDeliverPellet(dj.Imported):
        definition = """  # Raw per-chunk DeliverPellet data stream from UndergroundFeeder (auto-generated with aeon_mecha-unknown)
    -> UndergroundFeeder
    -> acquisition.Chunk
    ---
    sample_count: int      # number of data points acquired from this stream for a given chunk
    timestamps: longblob   # (datetime) timestamps of DeliverPellet data
    event: longblob
    """

        @property
        def key_source(self):
            f"""
            Only the combination of Chunk and UndergroundFeeder with overlapping time
            +  Chunk(s) that started after UndergroundFeeder install time and ended before UndergroundFeeder remove time
            +  Chunk(s) that started after UndergroundFeeder install time for UndergroundFeeder that are not yet removed
            """
            return (
                acquisition.Chunk * UndergroundFeeder.join(UndergroundFeeder.RemovalTime, left=True)
                & 'chunk_start >= underground_feeder_install_time'
                & 'chunk_start < IFNULL(underground_feeder_removal_time, "2200-01-01")'
            )

        def make(self, key):
            chunk_start, chunk_end = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")

            data_dirs = acquisition.Experiment.get_data_directories(key)

            device_name = (UndergroundFeeder & key).fetch1('underground_feeder_name')

            devices_schema = getattr(
                aeon_schemas,
                (acquisition.Experiment.DevicesSchema & {"experiment_name": key["experiment_name"]}).fetch1(
                    "devices_schema_name"
                ),
            )
            stream_reader = getattr(getattr(devices_schema, device_name), "DeliverPellet")

            stream_data = io_api.load(
                root=data_dirs,
                reader=stream_reader,
                start=pd.Timestamp(chunk_start),
                end=pd.Timestamp(chunk_end),
            )

            self.insert1(
                {
                    **key,
                    "sample_count": len(stream_data),
                    "timestamps": stream_data.index.values,
                    **{
                        re.sub(r"\([^)]*\)", "", c): stream_data[c].values
                        for c in stream_reader.columns
                        if not c.startswith("_")
                    },
                },
                ignore_extra_fields=True,
            )


@schema
class UndergroundFeederDepletionState(dj.Imported):
        definition = """  # Raw per-chunk DepletionState data stream from UndergroundFeeder (auto-generated with aeon_mecha-unknown)
    -> UndergroundFeeder
    -> acquisition.Chunk
    ---
    sample_count: int      # number of data points acquired from this stream for a given chunk
    timestamps: longblob   # (datetime) timestamps of DepletionState data
    threshold: longblob
    offset: longblob
    rate: longblob
    """

        @property
        def key_source(self):
            f"""
            Only the combination of Chunk and UndergroundFeeder with overlapping time
            +  Chunk(s) that started after UndergroundFeeder install time and ended before UndergroundFeeder remove time
            +  Chunk(s) that started after UndergroundFeeder install time for UndergroundFeeder that are not yet removed
            """
            return (
                acquisition.Chunk * UndergroundFeeder.join(UndergroundFeeder.RemovalTime, left=True)
                & 'chunk_start >= underground_feeder_install_time'
                & 'chunk_start < IFNULL(underground_feeder_removal_time, "2200-01-01")'
            )

        def make(self, key):
            chunk_start, chunk_end = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")

            data_dirs = acquisition.Experiment.get_data_directories(key)

            device_name = (UndergroundFeeder & key).fetch1('underground_feeder_name')

            devices_schema = getattr(
                aeon_schemas,
                (acquisition.Experiment.DevicesSchema & {"experiment_name": key["experiment_name"]}).fetch1(
                    "devices_schema_name"
                ),
            )
            stream_reader = getattr(getattr(devices_schema, device_name), "DepletionState")

            stream_data = io_api.load(
                root=data_dirs,
                reader=stream_reader,
                start=pd.Timestamp(chunk_start),
                end=pd.Timestamp(chunk_end),
            )

            self.insert1(
                {
                    **key,
                    "sample_count": len(stream_data),
                    "timestamps": stream_data.index.values,
                    **{
                        re.sub(r"\([^)]*\)", "", c): stream_data[c].values
                        for c in stream_reader.columns
                        if not c.startswith("_")
                    },
                },
                ignore_extra_fields=True,
            )


@schema
class UndergroundFeederEncoder(dj.Imported):
        definition = """  # Raw per-chunk Encoder data stream from UndergroundFeeder (auto-generated with aeon_mecha-unknown)
    -> UndergroundFeeder
    -> acquisition.Chunk
    ---
    sample_count: int      # number of data points acquired from this stream for a given chunk
    timestamps: longblob   # (datetime) timestamps of Encoder data
    angle: longblob
    intensity: longblob
    """

        @property
        def key_source(self):
            f"""
            Only the combination of Chunk and UndergroundFeeder with overlapping time
            +  Chunk(s) that started after UndergroundFeeder install time and ended before UndergroundFeeder remove time
            +  Chunk(s) that started after UndergroundFeeder install time for UndergroundFeeder that are not yet removed
            """
            return (
                acquisition.Chunk * UndergroundFeeder.join(UndergroundFeeder.RemovalTime, left=True)
                & 'chunk_start >= underground_feeder_install_time'
                & 'chunk_start < IFNULL(underground_feeder_removal_time, "2200-01-01")'
            )

        def make(self, key):
            chunk_start, chunk_end = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")

            data_dirs = acquisition.Experiment.get_data_directories(key)

            device_name = (UndergroundFeeder & key).fetch1('underground_feeder_name')

            devices_schema = getattr(
                aeon_schemas,
                (acquisition.Experiment.DevicesSchema & {"experiment_name": key["experiment_name"]}).fetch1(
                    "devices_schema_name"
                ),
            )
            stream_reader = getattr(getattr(devices_schema, device_name), "Encoder")

            stream_data = io_api.load(
                root=data_dirs,
                reader=stream_reader,
                start=pd.Timestamp(chunk_start),
                end=pd.Timestamp(chunk_end),
            )

            self.insert1(
                {
                    **key,
                    "sample_count": len(stream_data),
                    "timestamps": stream_data.index.values,
                    **{
                        re.sub(r"\([^)]*\)", "", c): stream_data[c].values
                        for c in stream_reader.columns
                        if not c.startswith("_")
                    },
                },
                ignore_extra_fields=True,
            )


@schema
class UndergroundFeederManualDelivery(dj.Imported):
        definition = """  # Raw per-chunk ManualDelivery data stream from UndergroundFeeder (auto-generated with aeon_mecha-unknown)
    -> UndergroundFeeder
    -> acquisition.Chunk
    ---
    sample_count: int      # number of data points acquired from this stream for a given chunk
    timestamps: longblob   # (datetime) timestamps of ManualDelivery data
    manual_delivery: longblob
    """

        @property
        def key_source(self):
            f"""
            Only the combination of Chunk and UndergroundFeeder with overlapping time
            +  Chunk(s) that started after UndergroundFeeder install time and ended before UndergroundFeeder remove time
            +  Chunk(s) that started after UndergroundFeeder install time for UndergroundFeeder that are not yet removed
            """
            return (
                acquisition.Chunk * UndergroundFeeder.join(UndergroundFeeder.RemovalTime, left=True)
                & 'chunk_start >= underground_feeder_install_time'
                & 'chunk_start < IFNULL(underground_feeder_removal_time, "2200-01-01")'
            )

        def make(self, key):
            chunk_start, chunk_end = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")

            data_dirs = acquisition.Experiment.get_data_directories(key)

            device_name = (UndergroundFeeder & key).fetch1('underground_feeder_name')

            devices_schema = getattr(
                aeon_schemas,
                (acquisition.Experiment.DevicesSchema & {"experiment_name": key["experiment_name"]}).fetch1(
                    "devices_schema_name"
                ),
            )
            stream_reader = getattr(getattr(devices_schema, device_name), "ManualDelivery")

            stream_data = io_api.load(
                root=data_dirs,
                reader=stream_reader,
                start=pd.Timestamp(chunk_start),
                end=pd.Timestamp(chunk_end),
            )

            self.insert1(
                {
                    **key,
                    "sample_count": len(stream_data),
                    "timestamps": stream_data.index.values,
                    **{
                        re.sub(r"\([^)]*\)", "", c): stream_data[c].values
                        for c in stream_reader.columns
                        if not c.startswith("_")
                    },
                },
                ignore_extra_fields=True,
            )


@schema
class UndergroundFeederMissedPellet(dj.Imported):
        definition = """  # Raw per-chunk MissedPellet data stream from UndergroundFeeder (auto-generated with aeon_mecha-unknown)
    -> UndergroundFeeder
    -> acquisition.Chunk
    ---
    sample_count: int      # number of data points acquired from this stream for a given chunk
    timestamps: longblob   # (datetime) timestamps of MissedPellet data
    missed_pellet: longblob
    """

        @property
        def key_source(self):
            f"""
            Only the combination of Chunk and UndergroundFeeder with overlapping time
            +  Chunk(s) that started after UndergroundFeeder install time and ended before UndergroundFeeder remove time
            +  Chunk(s) that started after UndergroundFeeder install time for UndergroundFeeder that are not yet removed
            """
            return (
                acquisition.Chunk * UndergroundFeeder.join(UndergroundFeeder.RemovalTime, left=True)
                & 'chunk_start >= underground_feeder_install_time'
                & 'chunk_start < IFNULL(underground_feeder_removal_time, "2200-01-01")'
            )

        def make(self, key):
            chunk_start, chunk_end = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")

            data_dirs = acquisition.Experiment.get_data_directories(key)

            device_name = (UndergroundFeeder & key).fetch1('underground_feeder_name')

            devices_schema = getattr(
                aeon_schemas,
                (acquisition.Experiment.DevicesSchema & {"experiment_name": key["experiment_name"]}).fetch1(
                    "devices_schema_name"
                ),
            )
            stream_reader = getattr(getattr(devices_schema, device_name), "MissedPellet")

            stream_data = io_api.load(
                root=data_dirs,
                reader=stream_reader,
                start=pd.Timestamp(chunk_start),
                end=pd.Timestamp(chunk_end),
            )

            self.insert1(
                {
                    **key,
                    "sample_count": len(stream_data),
                    "timestamps": stream_data.index.values,
                    **{
                        re.sub(r"\([^)]*\)", "", c): stream_data[c].values
                        for c in stream_reader.columns
                        if not c.startswith("_")
                    },
                },
                ignore_extra_fields=True,
            )


@schema
class UndergroundFeederRetriedDelivery(dj.Imported):
        definition = """  # Raw per-chunk RetriedDelivery data stream from UndergroundFeeder (auto-generated with aeon_mecha-unknown)
    -> UndergroundFeeder
    -> acquisition.Chunk
    ---
    sample_count: int      # number of data points acquired from this stream for a given chunk
    timestamps: longblob   # (datetime) timestamps of RetriedDelivery data
    retried_delivery: longblob
    """

        @property
        def key_source(self):
            f"""
            Only the combination of Chunk and UndergroundFeeder with overlapping time
            +  Chunk(s) that started after UndergroundFeeder install time and ended before UndergroundFeeder remove time
            +  Chunk(s) that started after UndergroundFeeder install time for UndergroundFeeder that are not yet removed
            """
            return (
                acquisition.Chunk * UndergroundFeeder.join(UndergroundFeeder.RemovalTime, left=True)
                & 'chunk_start >= underground_feeder_install_time'
                & 'chunk_start < IFNULL(underground_feeder_removal_time, "2200-01-01")'
            )

        def make(self, key):
            chunk_start, chunk_end = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")

            data_dirs = acquisition.Experiment.get_data_directories(key)

            device_name = (UndergroundFeeder & key).fetch1('underground_feeder_name')

            devices_schema = getattr(
                aeon_schemas,
                (acquisition.Experiment.DevicesSchema & {"experiment_name": key["experiment_name"]}).fetch1(
                    "devices_schema_name"
                ),
            )
            stream_reader = getattr(getattr(devices_schema, device_name), "RetriedDelivery")

            stream_data = io_api.load(
                root=data_dirs,
                reader=stream_reader,
                start=pd.Timestamp(chunk_start),
                end=pd.Timestamp(chunk_end),
            )

            self.insert1(
                {
                    **key,
                    "sample_count": len(stream_data),
                    "timestamps": stream_data.index.values,
                    **{
                        re.sub(r"\([^)]*\)", "", c): stream_data[c].values
                        for c in stream_reader.columns
                        if not c.startswith("_")
                    },
                },
                ignore_extra_fields=True,
            )


@schema
class WeightScaleWeightFiltered(dj.Imported):
        definition = """  # Raw per-chunk WeightFiltered data stream from WeightScale (auto-generated with aeon_mecha-unknown)
    -> WeightScale
    -> acquisition.Chunk
    ---
    sample_count: int      # number of data points acquired from this stream for a given chunk
    timestamps: longblob   # (datetime) timestamps of WeightFiltered data
    weight: longblob
    stability: longblob
    """

        @property
        def key_source(self):
            f"""
            Only the combination of Chunk and WeightScale with overlapping time
            +  Chunk(s) that started after WeightScale install time and ended before WeightScale remove time
            +  Chunk(s) that started after WeightScale install time for WeightScale that are not yet removed
            """
            return (
                acquisition.Chunk * WeightScale.join(WeightScale.RemovalTime, left=True)
                & 'chunk_start >= weight_scale_install_time'
                & 'chunk_start < IFNULL(weight_scale_removal_time, "2200-01-01")'
            )

        def make(self, key):
            chunk_start, chunk_end = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")

            data_dirs = acquisition.Experiment.get_data_directories(key)

            device_name = (WeightScale & key).fetch1('weight_scale_name')

            devices_schema = getattr(
                aeon_schemas,
                (acquisition.Experiment.DevicesSchema & {"experiment_name": key["experiment_name"]}).fetch1(
                    "devices_schema_name"
                ),
            )
            stream_reader = getattr(getattr(devices_schema, device_name), "WeightFiltered")

            stream_data = io_api.load(
                root=data_dirs,
                reader=stream_reader,
                start=pd.Timestamp(chunk_start),
                end=pd.Timestamp(chunk_end),
            )

            self.insert1(
                {
                    **key,
                    "sample_count": len(stream_data),
                    "timestamps": stream_data.index.values,
                    **{
                        re.sub(r"\([^)]*\)", "", c): stream_data[c].values
                        for c in stream_reader.columns
                        if not c.startswith("_")
                    },
                },
                ignore_extra_fields=True,
            )


@schema
class WeightScaleWeightRaw(dj.Imported):
        definition = """  # Raw per-chunk WeightRaw data stream from WeightScale (auto-generated with aeon_mecha-unknown)
    -> WeightScale
    -> acquisition.Chunk
    ---
    sample_count: int      # number of data points acquired from this stream for a given chunk
    timestamps: longblob   # (datetime) timestamps of WeightRaw data
    weight: longblob
    stability: longblob
    """

        @property
        def key_source(self):
            f"""
            Only the combination of Chunk and WeightScale with overlapping time
            +  Chunk(s) that started after WeightScale install time and ended before WeightScale remove time
            +  Chunk(s) that started after WeightScale install time for WeightScale that are not yet removed
            """
            return (
                acquisition.Chunk * WeightScale.join(WeightScale.RemovalTime, left=True)
                & 'chunk_start >= weight_scale_install_time'
                & 'chunk_start < IFNULL(weight_scale_removal_time, "2200-01-01")'
            )

        def make(self, key):
            chunk_start, chunk_end = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")

            data_dirs = acquisition.Experiment.get_data_directories(key)

            device_name = (WeightScale & key).fetch1('weight_scale_name')

            devices_schema = getattr(
                aeon_schemas,
                (acquisition.Experiment.DevicesSchema & {"experiment_name": key["experiment_name"]}).fetch1(
                    "devices_schema_name"
                ),
            )
            stream_reader = getattr(getattr(devices_schema, device_name), "WeightRaw")

            stream_data = io_api.load(
                root=data_dirs,
                reader=stream_reader,
                start=pd.Timestamp(chunk_start),
                end=pd.Timestamp(chunk_end),
            )

            self.insert1(
                {
                    **key,
                    "sample_count": len(stream_data),
                    "timestamps": stream_data.index.values,
                    **{
                        re.sub(r"\([^)]*\)", "", c): stream_data[c].values
                        for c in stream_reader.columns
                        if not c.startswith("_")
                    },
                },
                ignore_extra_fields=True,
            )
