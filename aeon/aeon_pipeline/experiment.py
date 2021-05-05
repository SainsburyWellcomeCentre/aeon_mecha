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
    experiment_start_time: datetime  # datetime of the start of this experiment
    experiment_description: varchar(1000)
    -> lab.Arena
    -> lab.Location  # lab/room where a particular experiment takes place
    """


@schema
class Camera(dj.Manual):
    definition = """
    -> experiment.Experiment
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
    -> experiment.Experiment
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
    data_category: varchar(50)  
    ---
    category_description: varchar(500)  # Short description of dataset type
    """
    contents = [
        ['SessionMeta', 'Meta information of session'],
        ['VideoCamera', 'Data from camera'],
        ['FoodPatch', 'Data from food patch']
    ]


@schema
class TimeBlock(dj.Manual):
    definition = """  # A recording period corresponds to an N-hour data acquisition
    -> Experiment
    time_block_start: datetime  # datetime of the start of this recorded TimeBlock
    ---
    time_block_end: datetime    # datetime of the end of this recorded TimeBlock
    """

    class File(dj.Part):
        definition = """
        -> master
        file_number: tinyint
        ---
        file_name: varchar(128)
        -> DataCategory
        file_path: varchar(255)  # relative path of the file
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

    contents = zip([(0, 'food-drop'),
                    (1, 'animal-enter'),
                    (2, 'animal-exit')])


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
        definition = """  # The food patch associated with a "food-drop" event 
        -> master
        ---
        -> FoodPatch
        """
