import inspect

from typing_extensions import deprecated


@deprecated("Please use the StreamGroup class from the streams module instead.")
def compositeStream(pattern, *args):
    """Merges multiple data streams into a single composite stream."""
    composite = {}
    if args:
        for binder_fn in args:
            if inspect.isclass(binder_fn):
                for method in vars(binder_fn).values():
                    if isinstance(method, staticmethod):
                        registry.update(method.__func__(pattern))
            else:
                registry.update(binder_fn(pattern))
    return registry


@deprecated("The Device class has been moved to the streams module.")
class Device:
    """Groups multiple Readers into a logical device.

    If a device contains a single stream reader with the same pattern as the device `name`, it will be
    considered a singleton, and the stream reader will be paired directly with the device without nesting.

    Attributes:
        name (str): Name of the device.
        args (any): A binder function or class that returns a dictionary of Readers.
        pattern (str, optional): Pattern used to find raw chunk files,
            usually in the format `<Device>_<DataStream>`.
    """

    def __init__(self, name, *args, pattern=None):
        self.name = name
        self.registry = register(name if pattern is None else pattern, *args)

    def __iter__(self):
        if len(self.registry) == 1:
            singleton = self.registry.get(self.name, None)
            if singleton:
                return iter((self.name, singleton))
        return iter((self.name, self.registry))
