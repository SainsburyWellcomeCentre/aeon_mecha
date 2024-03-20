import inspect
from warnings import warn


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
        path (str): Path to the folder where stream chunks are located.
        args (Any): Data streams or data stream groups to be included in this stream group.
    """

    def __init__(self, path, *args):
        self.path = path
        self._args = args

    def __iter__(self):
        for factory in self._args:
            for stream in iter(factory(self.path)):
                yield stream


class Device:
    """Groups multiple data streams into a logical device.

    If a device contains a single stream with the same pattern as the device
    `name`, it will be considered a singleton, and the stream reader will be
    paired directly with the device without nesting.

    Attributes:
        name (str): Name of the device.
        args (Any): Data streams collected from the device.
        path (str, optional): Path to the folder where stream chunks are located.
    """

    def __init__(self, name, *args, path=None):
        if name is None:
            raise ValueError("name cannot be None.")

        self.name = name
        self._streams = Device._createStreams(name if path is None else path, args)

    @staticmethod
    def _createStreams(path, args):
        streams = {}
        for factory in args:
            if inspect.isclass(factory) and not hasattr(factory.__init__, "__code__"):
                warn(
                    f"Stream group classes with default constructors are deprecated. {factory}",
                    category=DeprecationWarning,
                )
                for method in vars(factory).values():
                    if isinstance(method, staticmethod):
                        streams.update(method.__func__(path))
            else:
                streams.update(factory(path))
        return streams

    def __iter__(self):
        if len(self._streams) == 1:
            singleton = self._streams.get(self.name, None)
            if singleton:
                return iter((self.name, singleton))
        return iter((self.name, self._streams))
