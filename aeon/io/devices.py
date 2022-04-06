from enum import Enum
from aeon.io.reader import *

class Area(Enum):
    Null = 0
    Nest = 1
    Corridor = 2
    Arena = 3
    Patch1 = 4
    Patch2 = 5

class RegionReader(HarpReader):
    def __init__(self, name):
        super().__init__(name, columns=['region'])

    def read(self, file):
        data = super().read(file)
        categorical = pd.Categorical(data.region, categories=range(len(Area._member_names_)))
        data['region'] = categorical.rename_categories(Area._member_names_)
        return data

def videoStream(name):
    return name, { "Video": VideoReader(name) }

def positionStream(name):
    return name, { "Position": PositionReader(f"{name}_200") }

def regionStream(name):
    return name, { "Region": RegionReader(f"{name}_201") }

def depletionFunctionStream(name):
    return name, { "DepletionState": PatchStateReader(f"{name}_State") }

def encoderStream(name):
    return name, { "Encoder": EncoderReader(f"{name}_90") }

def feederStream(name):
    return name,{
        "BeamBreak": EventReader(f"{name}_32", 0x20, 'PelletDelivered'),
        "DeliverPellet": EventReader(f"{name}_35", 0x80, 'TriggerPellet')
    }

def patchStream(name):
    return compositeStream(name, depletionFunctionStream, encoderStream, feederStream)

def weightStream(name):
    return name, {
        "WeightRaw": WeightReader(f"{name}_200"),
        "WeightFiltered": WeightReader(f"{name}_202"),
        "WeightSubject": WeightReader(f"{name}_204")
    }

def metadataStream(name):
    return name, {
        "EnvironmentState": CsvReader(f"{name}_EnvironmentState", ['state']),
        "SubjectState": SubjectReader(f"{name}_SubjectState")
    }

def messageLogStream(name):
    return name, { "MessageLog": LogReader(f"{name}_MessageLog") }

def compositeStream(name, *streams):
    schema = {}
    if streams is not None:
        for sch in streams:
            schema.update(sch(name)[1])
    return name, schema

class Device:
    def __init__(self, name, *schemas):
        self.name = name
        self.schema = compositeStream(name, *schemas)[1]

    def __iter__(self):
        return iter((self.name, self.schema))
