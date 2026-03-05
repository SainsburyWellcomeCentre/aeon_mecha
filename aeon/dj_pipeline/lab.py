"""DataJoint schema for the lab pipeline."""

import datajoint as dj

from . import get_schema_name

schema = dj.schema(get_schema_name("lab"))
logger = dj.logger


# ------------------- GENERAL LAB INFORMATION --------------------


@schema
class Colony(dj.Lookup):
    # This table will interact with Bonsai directly.
    definition = """
    subject                 : varchar(32)
    ---
    reference_weight=null   : float
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
        (
            "SWC",
            "Sainsbury Wellcome Centre",
            "University College London",
            "25 Howland Street London W1T 4JG",
            "GMT+1",
        )
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
        ("SWC", "room-0", "room for experiment 0"),
        ("SWC", "room-1", "room for social experiment"),
        ("SWC", "464", "room for social experiment using octagon arena"),
        ("SWC", "AEON", "acquisition machine AEON"),
        ("SWC", "AEON2", "acquisition machine AEON2"),
        ("SWC", "AEON3", "acquisition machine AEON3"),
        ("SWC", "AEON4", "acquisition machine AEON4"),
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
    contents = zip(["square", "circular", "rectangular", "linear", "octagon"], strict=False)


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
    arena_x_dim:    float # (m) x-dimension of the bounding box of this arena
    arena_y_dim:    float # (m) y-dimension of the bounding box of this arena
    arena_z_dim=0:  float # (m) z-dimension of this arena (e.g. wall height)
    """

    contents = [
        ("circle-2m", "circular arena with 2-meter diameter", "circular", 2, 2, 0.2),
        ("octagon-1m", "octagon arena with 1-m diameter", "octagon", 1, 1, 0.2),
    ]


@schema
class ArenaNest(dj.Manual):
    definition = """
    -> Arena
    nest: int  # nest number - e.g. 1, 2, ...
    """

    class Vertex(dj.Part):
        definition = """
        -> master
        vertex: int
        ---
        vertex_x: float    # (m) x-coordinate of the vertex, in the arena's coordinate frame
        vertex_y: float    # (m) y-coordinate of the vertex, in the arena's coordinate frame
        vertex_z=0: float  # (m) z-coordinate of the vertex, in the arena's coordinate frame
        """


@schema
class ArenaTile(dj.Manual):
    definition = """
    -> Arena
    tile: int
    """

    class Vertex(dj.Part):
        definition = """
        -> master
        vertex: int
        ---
        vertex_x: float    # (m) x-coordinate of the vertex, in the arena's coordinate frame
        vertex_y: float    # (m) y-coordinate of the vertex, in the arena's coordinate frame
        vertex_z=0: float  # (m) z-coordinate of the vertex, in the arena's coordinate frame
        """
