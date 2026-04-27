"""DataJoint schema for the lab pipeline."""

import datajoint as dj

from . import get_schema_name

schema = dj.Schema(get_schema_name("lab"))
logger = dj.logger


# ------------------- GENERAL LAB INFORMATION --------------------


@schema
class Colony(dj.Lookup):
    # This table will interact with Bonsai directly.
    definition = """
    subject                 : varchar(32)
    ---
    reference_weight=null   : float32
    sex='U'                 : enum('M', 'F', 'U')
    subject_birth_date=null : date  # date of birth
    note=''                 : varchar(1024)
    """


@schema
class Lab(dj.Lookup):
    definition = """
    lab             : varchar(24)  #  Abbreviated lab name
    ---
    lab_name        : varchar(255)   # full lab name
    institution     : varchar(255)
    address         : varchar(255)
    time_zone       : varchar(64)
    """

    contents = [
        {
            "lab": "SWC",
            "lab_name": "Sainsbury Wellcome Centre",
            "institution": "University College London",
            "address": "25 Howland Street London W1T 4JG",
            "time_zone": "GMT+1",
        }
    ]


@schema
class Location(dj.Lookup):
    definition = """
    # location of animal housing or experimental rigs
    -> Lab
    location            : varchar(32)
    ---
    location_description=''    : varchar(255)
    """

    contents = [
        {"lab": "SWC", "location": "room-0", "location_description": "room for experiment 0"},
        {"lab": "SWC", "location": "room-1", "location_description": "room for social experiment"},
        {
            "lab": "SWC",
            "location": "464",
            "location_description": "room for social experiment using octagon arena",
        },
        {"lab": "SWC", "location": "AEON", "location_description": "acquisition machine AEON"},
        {"lab": "SWC", "location": "AEON2", "location_description": "acquisition machine AEON2"},
        {"lab": "SWC", "location": "AEON3", "location_description": "acquisition machine AEON3"},
        {"lab": "SWC", "location": "AEON4", "location_description": "acquisition machine AEON4"},
    ]


@schema
class User(dj.Lookup):
    definition = """
    user                    : varchar(32)  # swc username
    ---
    responsible_owner=''    : varchar(32)  # pyrat username
    responsible_id=''       : varchar(32)  # pyrat `responsible_id`
    """


# ------------------- ARENA INFORMATION --------------------


@schema
class ArenaShape(dj.Lookup):
    definition = """
    arena_shape: varchar(32)
    """
    contents = [
        {"arena_shape": "square"},
        {"arena_shape": "circular"},
        {"arena_shape": "rectangular"},
        {"arena_shape": "linear"},
        {"arena_shape": "octagon"},
    ]


@schema
class Arena(dj.Lookup):
    """Coordinate frame convention as the following items.

    + x-dimension: x=0 is the left most point of the bounding box of the arena
    + y-dimension: y=0 is the top most point of the bounding box of the arena
    + z-dimension: z=0 is the lowest point of the arena (e.g. the ground)
    TODO: confirm/update this.
    """

    definition = """
    arena_name: varchar(32)  # unique name of the arena (e.g. circular_2m)
    ---
    arena_description='': varchar(1000)
    -> ArenaShape
    arena_x_dim:    float32 # (m) x-dimension of the bounding box of this arena
    arena_y_dim:    float32 # (m) y-dimension of the bounding box of this arena
    arena_z_dim=0:  float32 # (m) z-dimension of this arena (e.g. wall height)
    """

    contents = [
        {
            "arena_name": "circle-2m",
            "arena_description": "circular arena with 2-meter diameter",
            "arena_shape": "circular",
            "arena_x_dim": 2,
            "arena_y_dim": 2,
            "arena_z_dim": 0.2,
        },
        {
            "arena_name": "octagon-1m",
            "arena_description": "octagon arena with 1-m diameter",
            "arena_shape": "octagon",
            "arena_x_dim": 1,
            "arena_y_dim": 1,
            "arena_z_dim": 0.2,
        },
    ]


@schema
class ArenaNest(dj.Manual):
    definition = """
    -> Arena
    nest: int32  # nest number - e.g. 1, 2, ...
    """

    class Vertex(dj.Part):
        definition = """
        -> master
        vertex: int32
        ---
        vertex_x: float32    # (m) x-coordinate of the vertex, in the arena's coordinate frame
        vertex_y: float32    # (m) y-coordinate of the vertex, in the arena's coordinate frame
        vertex_z=0: float32  # (m) z-coordinate of the vertex, in the arena's coordinate frame
        """


@schema
class ArenaTile(dj.Manual):
    definition = """
    -> Arena
    tile: int32
    """

    class Vertex(dj.Part):
        definition = """
        -> master
        vertex: int32
        ---
        vertex_x: float32    # (m) x-coordinate of the vertex, in the arena's coordinate frame
        vertex_y: float32    # (m) y-coordinate of the vertex, in the arena's coordinate frame
        vertex_z=0: float32  # (m) z-coordinate of the vertex, in the arena's coordinate frame
        """
