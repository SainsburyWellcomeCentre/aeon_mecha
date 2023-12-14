import aeon.io.device as _device
import aeon.io.reader as _reader


def heartbeat(pattern):
    """Heartbeat event for Harp devices."""
    return {"Heartbeat": _reader.Heartbeat(f"{pattern}_8_*")}


def video(pattern):
    """Video frame metadata."""
    return {"Video": _reader.Video(f"{pattern}_*")}


def position(pattern):
    """Position tracking data for the specified camera."""
    return {"Position": _reader.Position(f"{pattern}_200_*")}


def encoder(pattern):
    """Wheel magnetic encoder data."""
    return {"Encoder": _reader.Encoder(f"{pattern}_90_*")}


def environment(pattern):
    """Metadata for environment mode and subjects."""
    return _device.register(pattern, environment_state, subject_state)


def environment_state(pattern):
    """Environment state log."""
    return {"EnvironmentState": _reader.Csv(f"{pattern}_EnvironmentState_*", ["state"])}


def subject_state(pattern):
    """Subject state log."""
    return {"SubjectState": _reader.Subject(f"{pattern}_SubjectState_*")}


def messageLog(pattern):
    """Message log data."""
    return {"MessageLog": _reader.Log(f"{pattern}_MessageLog_*")}


def metadata(pattern):
    """Metadata for acquisition epochs."""
    return {pattern: _reader.Metadata(pattern)}
