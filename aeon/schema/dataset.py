from pathlib import Path

from dotmap import DotMap

import aeon.schema.core as stream
import aeon.schema.foraging as foraging
import aeon.schema.octagon as octagon
from aeon.io import reader
from aeon.io.device import Device

exp02 = DotMap(
    [
        Device("Metadata", stream.metadata),
        Device("ExperimentalMetadata", stream.environment, stream.messageLog),
        Device("CameraTop", stream.video, stream.position, foraging.region),
        Device("CameraEast", stream.video),
        Device("CameraNest", stream.video),
        Device("CameraNorth", stream.video),
        Device("CameraPatch1", stream.video),
        Device("CameraPatch2", stream.video),
        Device("CameraSouth", stream.video),
        Device("CameraWest", stream.video),
        Device("Nest", foraging.weight),
        Device("WeightNest", foraging.weight),
        Device("Patch1", foraging.patch),
        Device("Patch2", foraging.patch),
    ]
)

exp01 = DotMap(
    [
        Device("SessionData", foraging.session),
        Device("FrameTop", stream.video, stream.position),
        Device("FrameEast", stream.video),
        Device("FrameGate", stream.video),
        Device("FrameNorth", stream.video),
        Device("FramePatch1", stream.video),
        Device("FramePatch2", stream.video),
        Device("FrameSouth", stream.video),
        Device("FrameWest", stream.video),
        Device("Patch1", foraging.depletionFunction, stream.encoder, foraging.feeder),
        Device("Patch2", foraging.depletionFunction, stream.encoder, foraging.feeder),
    ]
)

octagon01 = DotMap(
    [
        Device("Metadata", stream.metadata),
        Device("CameraTop", stream.video, stream.position),
        Device("CameraColorTop", stream.video),
        Device("ExperimentalMetadata", stream.subject_state),
        Device("Photodiode", octagon.photodiode),
        Device("OSC", octagon.OSC),
        Device("TaskLogic", octagon.TaskLogic),
        Device("Wall1", octagon.Wall),
        Device("Wall2", octagon.Wall),
        Device("Wall3", octagon.Wall),
        Device("Wall4", octagon.Wall),
        Device("Wall5", octagon.Wall),
        Device("Wall6", octagon.Wall),
        Device("Wall7", octagon.Wall),
        Device("Wall8", octagon.Wall),
    ]
)

presocial = exp02
presocial.Patch1.BeamBreak = reader.BitmaskEvent(
    pattern="Patch1_32", value=0x22, tag="BeamBroken"
)
presocial.Patch2.BeamBreak = reader.BitmaskEvent(
    pattern="Patch2_32", value=0x22, tag="BeamBroken"
)
presocial.Patch1.DeliverPellet = reader.BitmaskEvent(
    pattern="Patch1_35", value=0x1, tag="TriggeredPellet"
)
presocial.Patch2.DeliverPellet = reader.BitmaskEvent(
    pattern="Patch2_35", value=0x1, tag="TriggeredPellet"
)


def get_device_info(schema: DotMap) -> dict[dict]:
    """
    Read from the above DotMap object and returns a device dictionary as the following.

    Args:
        schema (DotMap): DotMap object (e.g., exp02)

    e.g.   {'CameraTop':
                {'stream_type': ['Video', 'Position', 'Region'],
                    'reader': [
                                aeon.io.reader.Video,
                                aeon.io.reader.Position,
                                aeon.schema.foraging._RegionReader
                            ],
                    'pattern': ['{pattern}', '{pattern}_200', '{pattern}_201']
                }
            }
    """
    import json
    from collections import defaultdict

    from aeon.dj_pipeline import dict_to_uuid

    schema_json = json.dumps(schema, default=lambda o: o.__dict__, indent=4)
    schema_dict = json.loads(schema_json)

    device_info = {}

    for device_name in schema:
        if not device_name.startswith("_"):
            device_info[device_name] = defaultdict(list)
            if isinstance(schema[device_name], DotMap):
                for stream_type in schema[device_name].keys():
                    if schema[device_name][stream_type].__class__.__module__ in [
                        "aeon.io.reader",
                        "aeon.schema.foraging",
                        "aeon.schema.octagon",
                    ]:
                        device_info[device_name]["stream_type"].append(stream_type)
                        device_info[device_name]["reader"].append(
                            schema[device_name][stream_type].__class__
                        )
            else:
                stream_type = schema[device_name].__class__.__name__
                device_info[device_name]["stream_type"].append(stream_type)
                device_info[device_name]["reader"].append(schema[device_name].__class__)

    """Add a kwargs such as pattern, columns, extension, dtype and hash
    e.g., {'pattern': '{pattern}_SubjectState',
            'columns': ['id', 'weight', 'event'],
            'extension': 'csv',
            'dtype': None}"""
    for device_name in device_info:
        if pattern := schema_dict[device_name].get("pattern"):
            schema_dict[device_name]["pattern"] = pattern.replace(
                device_name, "{pattern}"
            )

            # Add stream_reader_kwargs
            kwargs = schema_dict[device_name]
            device_info[device_name]["stream_reader_kwargs"].append(kwargs)
            stream_reader = device_info[device_name]["stream_reader"]
            # Add hash
            device_info[device_name]["stream_hash"].append(
                dict_to_uuid({**kwargs, "stream_reader": stream_reader})
            )

        else:
            for stream_type in device_info[device_name]["stream_type"]:
                pattern = schema_dict[device_name][stream_type]["pattern"]
                schema_dict[device_name][stream_type]["pattern"] = pattern.replace(
                    device_name, "{pattern}"
                )
                # Add stream_reader_kwargs
                kwargs = schema_dict[device_name][stream_type]
                device_info[device_name]["stream_reader_kwargs"].append(kwargs)
                stream_ind = device_info[device_name]["stream_type"].index(stream_type)
                stream_reader = device_info[device_name]["stream_reader"][stream_ind]
                # Add hash
                device_info[device_name]["stream_hash"].append(
                    dict_to_uuid({**kwargs, "stream_reader": stream_reader})
                )

    return device_info


def add_device_type(schema: DotMap, metadata_yml_filepath: Path):
    """Update device_info with device_type based on metadata.yml.

    Args:
        schema (DotMap): DotMap object (e.g., exp02)
        metadata_yml_filepath (Path): Path to metadata.yml.

    Returns:
        device_info (dict): Updated device_info.
    """
    from aeon.io import api

    meta_data = (
        api.load(
            str(metadata_yml_filepath.parent),
            schema.Metadata,
        )
        .reset_index()
        .to_dict("records")[0]["metadata"]
    )

    # Get device_type_mapper based on metadata.yml {'CameraTop': 'VideoSource', 'Patch1': 'Patch'}
    device_type_mapper = {}
    for item in meta_data.Devices:
        device_type_mapper[item.Name] = item.Type

    device_info = {
        device_name: {
            "device_type": device_type_mapper.get(device_name, None),
            **device_info[device_name],
        }
        for device_name in device_info
    }

    return device_info
