import inspect
from itertools import chain


class Stream:
    """Represents a single data stream.

    Attributes:
        reader (Reader): The reader used to retrieve the stream data.
    """

    def __init__(self, reader):
        self.reader = reader

    def __iter__(self):
        yield (self.__class__.__name__, self.reader)


class StreamGroup:
    """Represents a logical group of multiple data streams.

    Attributes:
        name (str): Name of the logical group used to find raw files.
    """

    def __init__(self, name, *args):
        self.name = name
        self._args = args

    def __iter__(self):
        for member in chain(vars(self.__class__).values(), self._args):
            if inspect.isclass(member):
                for stream in iter(member(self.name)):
                    yield stream
            elif isinstance(member, staticmethod):
                for stream in iter(member.__func__(self.name)):
                    yield stream


def compositeStream(pattern, *args):
    """Merges multiple data streams into a single composite stream."""
    composite = {}
    if args:
        for stream in args:
            composite.update(stream(pattern))
    return composite


class Device:
    """Groups multiple data streams into a logical device.

    If a device contains a single stream with the same pattern as the device
    `name`, it will be considered a singleton, and the stream reader will be
    paired directly with the device without nesting.

    Attributes:
        name (str): Name of the device.
        args (Any): Data streams collected from the device.
        pattern (str, optional): Pattern used to find raw chunk files,
            usually in the format `<Device>_<DataStream>`.
    """

    def __init__(self, name, *args, pattern=None):
        self.name = name
        self.provider = compositeStream(name if pattern is None else pattern, *args)

    def __iter__(self):
        if len(self.provider) == 1:
            singleton = self.provider.get(self.name, None)
            if singleton:
                return iter((self.name, singleton))
        return iter((self.name, self.provider))
