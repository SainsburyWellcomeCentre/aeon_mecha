from aeon.schema.streams import Stream, StreamGroup
import aeon.io.reader as _reader


class Heartbeat(Stream):
    """Heartbeat event for Harp devices."""

    def __init__(self, pattern):
        super().__init__(_reader.Heartbeat(f"{pattern}_8_*"))


class Video(Stream):
    """Video frame metadata."""

    def __init__(self, pattern):
        super().__init__(_reader.Video(f"{pattern}_*"))


class Position(Stream):
    """Position tracking data for the specified camera."""

    def __init__(self, pattern):
        super().__init__(_reader.Position(f"{pattern}_200_*"))


class Encoder(Stream):
    """Wheel magnetic encoder data."""

    def __init__(self, pattern):
        super().__init__(_reader.Encoder(f"{pattern}_90_*"))


class Environment(StreamGroup):
    """Metadata for environment mode and subjects."""

    def __init__(self, pattern):
        super().__init__(pattern, EnvironmentState, SubjectState)


class EnvironmentState(Stream):
    """Environment state log."""

    def __init__(self, pattern):
        super().__init__(_reader.Csv(f"{pattern}_EnvironmentState_*", ["state"]))


class SubjectState(Stream):
    """Subject state log."""

    def __init__(self, pattern):
        super().__init__(_reader.Subject(f"{pattern}_SubjectState_*"))


class MessageLog(Stream):
    """Message log data."""

    def __init__(self, pattern):
        super().__init__(_reader.Log(f"{pattern}_MessageLog_*"))


class Metadata(Stream):
    """Metadata for acquisition epochs."""

    def __init__(self, pattern):
        super().__init__(_reader.Metadata(pattern))
