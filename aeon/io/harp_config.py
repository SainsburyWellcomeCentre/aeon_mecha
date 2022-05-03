# The default temporal resolution of the Harp behavior board, in seconds.
_HARP_RES = 32e-6
# Map of Harp payload types to datatypes, including endianness (to read by `ndarray()`)
_HARP_PTYPES = DotMap(
    {
        1: "<u1",  # little-endian, unsigned int, 1-byte representation
        2: "<u2",
        4: "<u4",
        8: "<u8",
        129: "<i1",
        130: "<i2",
        132: "<i4",
        136: "<i8",  # little-endian, signed int, 8-byte representation
        68: "<f4",  # little-endian, float, 4-byte representation
    }
)
_HARP_T_PTYPES = DotMap(
    {
        17: "<u1",
        18: "<u2",
        20: "<u4",
        24: "<u8",
        145: "<i1",
        146: "<i2",
        148: "<i4",
        152: "<i8",
        84: "<f4",
    }
)
# Bitmasks that link Harp payload datatypes:
# non-timestamped -> timestamped:
# e.g. _HARP_T_PTYPES[17] == (_HARP_PTYPES[1] | _HARP_N2T_BITMASK)
_HARP_N2T_MASK = 0x10
# unsigned -> signed: e.g. _HARP_PTYPES[129] == (_HARP_PTYPES[1] | _HARP_U2I_BITMASK)
_HARP_U2I_MASK = 0x80
# unsigned -> float: e.g. _HARP_PTYPES[68] == (_HARP_PTYPES[1] | _HARP_U2F_BITMASK)
_HARP_U2F_MASK = 0x40
# harp event-bitmask map
_HARP_EVENT_BITMASK = DotMap(
    {
        "pellet_trigger": 0x80,
        "pellet_detected_in": 0x20,
        "pellet_detected_out": 0x22,
    }
)
# Map of Harp device registers used in the Experiment 1 arena.
# @todo this isn't used so should be moved out of this module, but kept as doc
HARP_REGISTERS = {
    ("Patch", 90): ["angle, intensity"],  # wheel encoder
    ("Patch", 35): ["bitmask"],  # trigger pellet delivery
    ("Patch", 32): ["bitmask"],  # pellet detected by beam break
    ("VideoController", 68): ["pwm_mask"],  # camera trigger times (top and side)
    ("PositionTracking", 200): ["x", "y", "angle", "major", "minor", "area"],
    ("WeightScale", 200): ["value", "stable"],
}