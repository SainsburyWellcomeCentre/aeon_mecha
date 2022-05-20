import aeon.io.reader as _reader
import aeon.io.device as _device

def video(pattern):
    """Video frame metadata."""
    return { "Video": _reader.Video(pattern) }

def position(pattern):
    """Position tracking data for the specified camera."""
    return { "Position": _reader.Position(f"{pattern}_200") }

def encoder(pattern):
    """Wheel magnetic encoder data."""
    return { "Encoder": _reader.Encoder(f"{pattern}_90") }

def environment(pattern):
    """Metadata for environment mode and subjects."""
    return _device.compositeStream(pattern, environment_state, subject_state)

def environment_state(pattern):
    """Environment state log."""
    return { "EnvironmentState": _reader.Csv(f"{pattern}_EnvironmentState", ['state']) }

def subject_state(pattern):
    """Subject state log."""
    return { "SubjectState": _reader.Subject(f"{pattern}_SubjectState") }

def messageLog(pattern):
    """Message log data."""
    return { "MessageLog": _reader.Csv(f"{pattern}_MessageLog", columns=['priority', 'type', 'message']) }

def metadata(pattern):
    """Metadata for acquisition epochs."""
    return { pattern: _reader.Metadata(pattern) }
