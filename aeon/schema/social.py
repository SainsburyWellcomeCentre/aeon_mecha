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
# binder function: "device-name passed"; `pattern` will be set by `Device` object name: "Environment"
block_state_b = lambda pattern: {
    "BlockState": reader.Csv(f"{pattern}_BlockState*", ["pellet_ct", "pellet_ct_thresh", "due_time"])
}

# EnvironmentState

# Combine EnvironmentState and BlockState
env_block_state_b = lambda pattern: register(pattern, core.environment_state, block_state_b)

# LightEvents
cols = ["channel", "value"]
light_events_r = reader.Csv("Environment_LightEvents*", cols)
light_events_b = lambda pattern: {"LightEvents": light_events_r}  # binder function: "empty pattern"

# SubjectState
cols = ["id", "weight", "type"]
subject_state_r = reader.Csv("Environment_SubjectState*", cols)
subject_state_b = lambda pattern: {"SubjectState": subject_state_r}  # binder function: "empty pattern"

# SubjectVisits
cols = ["id", "type", "region"]
subject_visits_r = reader.Csv("Environment_SubjectVisits*", cols)
subject_visits_b = lambda pattern: {"SubjectVisits": subject_visits_r}  # binder function: "empty pattern"

# SubjectWeight
cols = ["weight", "confidence", "subject_id", "int_id"]
subject_weight_r = reader.Csv("Environment_SubjectWeight*", cols)
subject_weight_b = lambda pattern: {"SubjectWeight": subject_weight_r}  # binder function: "empty pattern"

# Nested binder fn Device object.
environment = Device("Environment", env_block_state_b, light_events_b, core.message_log)  # device name

# Separate Device object for subject-specific streams.
subject = Device("Subject", subject_state_b, subject_visits_b, subject_weight_b)

# ---

# Camera
# ---

camera_top_pos_b = lambda pattern: {"Pose": reader.Pose(f"{pattern}_test-node1*")}

# ---

# Nest
# ---

weight_raw_b = lambda pattern: {"WeightRaw": reader.Harp("Nest_200*", ["weight(g)", "stability"])}
weight_filtered_b = lambda pattern: {"WeightFiltered": reader.Harp("Nest_202*", ["weight(g)", "stability"])}

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

rfid_names = ["EventsGate", "EventsNest1", "EventsNest2", "EventsPatch1", "EventsPatch2", "EventsPatch3"]
rfid_b = lambda pattern: {"RFID": reader.Harp(f"{pattern}_*", ["rfid"])}
