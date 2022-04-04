from dotmap import DotMap
from aeon.io.reader import *

def _cameraSchema(name):
    return name, { "Video": VideoReader(name) }

def _trackingSchema(name):
    return name, { "Position": PositionReader(f"{name}_200") }

def _patchSchema(name):
    return name, {
        "DepletionState": PatchStateReader(f"{name}_State"),
        "DispenserState": HarpReader(f"{name}_200", ['count']),
        "Encoder": EncoderReader(f"{name}_90"),
        "BeamBreak": EventReader(f"{name}_32", 0x20, 'PelletDelivered'),
        "DeliverPellet": EventReader(f"{name}_35", 0x80, 'TriggerPellet')
    }

def _nestSchema(name):
    return name, {
        "WeightRaw": WeightReader(f"{name}_200"),
        "WeightFiltered": WeightReader(f"{name}_202"),
        "WeightSubject": WeightReader(f"{name}_204")
    }

def _metadataSchema(name):
    return name, {
        "EnvironmentState": CsvReader(f"{name}_EnvironmentState", ['state']),
        "SubjectState": SubjectReader(f"{name}_SubjectState"),
        "MessageLog": LogReader(f"{name}_MessageLog")
    }

def _linkedSchema(name, *schemas):
    schema = {}
    if schemas is not None:
        for sch in schemas:
            schema.update(sch(name)[1])
    return name, schema

def _dictFromItems(*items):
    return { k:v for k,v in items }

exp02 = DotMap(_dictFromItems(
    _linkedSchema("CameraTop", _cameraSchema, _trackingSchema),
    _cameraSchema("CameraEast"),
    _cameraSchema("CameraNest"),
    _cameraSchema("CameraNorth"),
    _cameraSchema("CameraPatch1"),
    _cameraSchema("CameraPatch2"),
    _cameraSchema("CameraSouth"),
    _cameraSchema("CameraWest"),
    _metadataSchema("ExperimentalMetadata"),
    _nestSchema("Nest"),
    _patchSchema("Patch1"),
    _patchSchema("Patch2")
))
