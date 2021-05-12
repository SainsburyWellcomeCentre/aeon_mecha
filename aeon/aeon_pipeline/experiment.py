import datajoint as dj

from . import lab, subject
from . import get_schema_name


schema = dj.schema(get_schema_name('experiment'))


# ------------------- GENERAL INFORMATION ABOUT AN EXPERIMENT --------------------


@schema
class Experiment(dj.Manual):
    definition = """
    experiment_name: char(8)  # e.g exp0-a
    ---
    experiment_start_time: datetime(3)  # datetime of the start of this experiment
    experiment_description: varchar(1000)
    -> lab.Arena
    -> lab.Location  # lab/room where a particular experiment takes place
    """


@schema
class Camera(dj.Manual):
    definition = """
    -> Experiment
    camera_name:            varchar(24)    # device type/function
    ---
    sampling_rate:          decimal(8, 4)  # sampling rate (Hz)
    camera_position_x:      float          # (m) x-position, in the arena's coordinate frame
    camera_position_y:      float          # (m) y-position, in the arena's coordinate frame
    camera_position_z=0:    float          # (m) z-position, in the arena's coordinate frame
    camera_description='':  varchar(100)   # device description
    """


@schema
class FoodPatch(dj.Manual):
    definition = """
    -> Experiment
    food_patch_name:            varchar(24)    # device type/function
    ---
    food_patch_position_x:      float          # (m) x-position, in the arena's coordinate frame
    food_patch_position_y:      float          # (m) y-position, in the arena's coordinate frame
    food_patch_position_z=0:    float          # (m) z-position, in the arena's coordinate frame
    food_patch_description='':  varchar(100)   # device description
    """


# ------------------- DATASET ------------------------

@schema
class DataCategory(dj.Lookup):
    definition = """
    data_category: varchar(24)  
    ---
    category_description: varchar(500)  # Short description of dataset type
    """
    contents = [
        ['SessionMeta', 'Meta information of session'],
        ['VideoCamera', 'Data from camera'],
        ['VideoEvents', 'Events from video camera'],
        ['PatchEvents', 'Events from food patch'],
        ['Wheel', 'Events from wheel device'],
        ['Audio', 'Audio data']
    ]

    category_mapper = {'SessionData': 'SessionMeta',
                       'PatchEvents': 'PatchEvents',
                       'VideoEvents': 'VideoEvents',
                       'FrameSide': 'VideoCamera',
                       'FrameTop': 'VideoCamera',
                       'WheelThreshold': 'Wheel',
                       'AudioAmbient': 'Audio'}


@schema
class DataRepository(dj.Lookup):
    definition = """
    repository_name: varchar(16)
    ---
    repository_path: varchar(255)  # path to the data directory of this repository (posix path)
    """

    contents = [('ceph_aeon_test2', '/ceph/aeon/test2/data')]


@schema
class TimeBlock(dj.Manual):
    definition = """  # A recording period corresponds to an N-hour data acquisition
    -> Experiment
    time_block_start: datetime(3)  # datetime of the start of this recorded TimeBlock
    ---
    time_block_end: datetime(3)    # datetime of the end of this recorded TimeBlock
    """

    class Subject(dj.Part):
        definition = """  # the animal(s) present in the arena during this timeblock
        -> master
        -> subject.Subject
        """

    class File(dj.Part):
        definition = """
        -> master
        file_number: tinyint
        ---
        file_name: varchar(128)
        -> DataCategory
        -> DataRepository
        file_path: varchar(255)  # path of the file, relative to the data repository
        """


# ------------------- SUBJECT PERIOD --------------------


@schema
class SubjectEpoch(dj.Manual):
    definition = """
    # A short time-chunk (e.g. 30 seconds) of the recording of a given animal in the arena
    -> subject.Subject
    epoch_start: datetime(3)  # datetime of the start of this Epoch
    ---
    epoch_end: datetime(3)    # datetime of the end of this Epoch
    -> TimeBlock              # the TimeBlock containing this Epoch
    """


@schema
class EventType(dj.Lookup):
    definition = """
    event_code: smallint
    ---
    event_type: varchar(24)
    """

    contents = [(0, 'food-drop'),
                (1, 'animal-enter'),
                (2, 'animal-exit')]


@schema
class Event(dj.Imported):
    definition = """  # events associated with a given animal in a given SubjectTimeBlock
    -> SubjectEpoch
    event_number: smallint
    ---
    -> EventType
    event_time: decimal(8, 2)  # (s) event time w.r.t to the start of this TimeBlock
    """

    class FoodPatch(dj.Part):
        definition = """  # The food patch associated with a food-drop event
        -> master
        ---
        -> FoodPatch
        """
