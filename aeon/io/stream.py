from enum import Enum
from aeon.io.reader import *
from aeon.io.device import compositeStream

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

def video(name):
    return name, { "Video": VideoReader(name) }

def position(name):
    return name, { "Position": PositionReader(f"{name}_200") }

def region(name):
    return name, { "Region": RegionReader(f"{name}_201") }

def depletionFunction(name):
    return name, { "DepletionState": PatchStateReader(f"{name}_State") }

def encoder(name):
    return name, { "Encoder": EncoderReader(f"{name}_90") }

def feeder(name):
    return name,{
        "BeamBreak": EventReader(f"{name}_32", 0x20, 'PelletDelivered'),
        "DeliverPellet": EventReader(f"{name}_35", 0x80, 'TriggerPellet')
    }

def patch(name):
    return compositeStream(name, depletionFunction, encoder, feeder)

def weight(name):
    return name, {
        "WeightRaw": WeightReader(f"{name}_200"),
        "WeightFiltered": WeightReader(f"{name}_202"),
        "WeightSubject": WeightReader(f"{name}_204")
    }

def metadata(name):
    return name, {
        "EnvironmentState": CsvReader(f"{name}_EnvironmentState", ['state']),
        "SubjectState": SubjectReader(f"{name}_SubjectState")
    }

def messageLog(name):
    return name, { "MessageLog": LogReader(f"{name}_MessageLog") }