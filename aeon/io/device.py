import inspect
from typing_extensions import deprecated


@deprecated("Please use the StreamGroup class from the streams module instead.")
def compositeStream(pattern, *args):
    """Merges multiple data streams into a single composite stream."""
    composite = {}
    if args:
        for stream in args:
            if inspect.isclass(stream):
                for method in vars(stream).values():
                    if isinstance(method, staticmethod):
                        composite.update(method.__func__(pattern))
            else:
                composite.update(stream(pattern))
    return composite


@deprecated("The Device class has been moved to the streams module.")
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
        self.stream = compositeStream(name if pattern is None else pattern, *args)

    def __iter__(self):
        if len(self.stream) == 1:
            singleton = self.stream.get(self.name, None)
            if singleton:
                return iter((self.name, singleton))
        return iter((self.name, self.stream))
