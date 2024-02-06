from aeon.io import reader
from aeon.io.device import Device, register
from aeon.schema import core, foraging


"""Creating the Social 0.1 schema"""

# Above we've listed out all the streams we recorded from during Social0.1, but we won't care to analyze all
# of them. Instead, we'll create a DotMap schema from Device objects that only contains Readers for the
# streams we want to analyze.

# We'll see both examples of binder functions we saw previously: 1. "empty pattern", and
# 2. "device-name passed".

# And we'll see both examples of instantiating Device objects we saw previously: 1. from singleton binder
# functions; 2. from multiple and/or nested binder functions.

# (Note, in the simplest case, a schema can always be created from / reduced to  "empty pattern" binder
# functions as singletons in Device objects.)

# Metadata.yml (will be a singleton binder function Device object)
# ---

metadata = Device("Metadata", core.metadata)

# ---

# Environment (will be a nested, multiple binder function Device object)
# ---

# BlockState
block_state_b = lambda pattern: {
    "BlockState": reader.Csv(f"{pattern}_BlockState_*", ["pellet_ct", "pellet_ct_thresh", "due_time"])
}

# LightEvents
light_events_b = lambda pattern: {
    "LightEvents": reader.Csv(f"{pattern}_LightEvents_*", ["channel", "value"])
}

# Combine EnvironmentState, BlockState, LightEvents
environment_b = lambda pattern: register(
    pattern, core.environment_state, block_state_b, light_events_b, core.message_log
)

# SubjectState
subject_state_b = lambda pattern: {
    "SubjectState": reader.Csv(f"{pattern}_SubjectState_*", ["id", "weight", "type"])
}

# SubjectVisits
subject_visits_b = lambda pattern: {
    "SubjectVisits": reader.Csv(f"{pattern}_SubjectVisits_*", ["id", "type", "region"])
}

# SubjectWeight
subject_weight_b = lambda pattern: {
    "SubjectWeight": reader.Csv(
        f"{pattern}_SubjectWeight_*", ["weight", "confidence", "subject_id", "int_id"]
    )
}

# Separate Device object for subject-specific streams.
subject_b = lambda pattern: register(pattern, subject_state_b, subject_visits_b, subject_weight_b)
# ---

# Camera
# ---

camera_top_pos_b = lambda pattern: {"Pose": reader.Pose(f"{pattern}_test-node1*")}

# ---

# Nest
# ---

weight_raw_b = lambda pattern: {"WeightRaw": reader.Harp(f"{pattern}_200_*", ["weight(g)", "stability"])}
weight_filtered_b = lambda pattern: {
    "WeightFiltered": reader.Harp(f"{pattern}_202_*", ["weight(g)", "stability"])
}

# ---

# Patch
# ---

# Combine streams for Patch device
patch_streams_b = lambda pattern: register(
    pattern,
    foraging.pellet_depletion_state,
    core.encoder,
    foraging.feeder,
    foraging.pellet_manual_delivery,
    foraging.missed_pellet,
    foraging.pellet_retried_delivery,
)
# ---

# Rfid
# ---


def rfid_events_social01_b(pattern):
    """RFID events reader (with social0.1 specific logic)"""
    pattern = pattern.replace("Rfid", "")
    if pattern.startswith("Events"):
        pattern = pattern.replace("Events", "")
    return {"RfidEvents": reader.Harp(f"RfidEvents{pattern}_*", ["rfid"])}


def rfid_events_b(pattern):
    """RFID events reader"""
    return {"RfidEvents": reader.Harp(f"{pattern}_32*", ["rfid"])}
